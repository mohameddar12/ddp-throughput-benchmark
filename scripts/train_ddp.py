# scripts/train_ddp.py
import os, time, yaml, logging, json, random
from pathlib import Path
from itertools import chain
from datetime import datetime, timezone

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    default_data_collator,
    get_linear_schedule_with_warmup,
)
import csv

# -------------------- utilities --------------------

def is_rank0():
    try:
        return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
    except Exception:
        return True


def setup_logging(rank: int):
    level = logging.INFO if rank == 0 else logging.WARN
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger("ddp")


class CSVLogger:
    def __init__(self, out_path: str, fieldnames: list[str]):
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        self._f = open(self.out_path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=fieldnames)
        self._w.writeheader()
    def log(self, row: dict):
        self._w.writerow(row); self._f.flush()
    def close(self):
        try: self._f.close()
        except Exception: pass


# -------------------- config helpers --------------------

def set_by_path(d, dotted_key, value):
    cur = d
    keys = dotted_key.split('.')
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def coerce(value: str):
    if value.lower() in {"true","false"}: return value.lower()=="true"
    try:
        if "." in value: return float(value)
        return int(value)
    except ValueError:
        return value


def apply_overrides(cfg: dict, unknown: list[str]):
    # accepts pairs: --a.b.c VALUE
    i = 0
    while i < len(unknown):
        tok = unknown[i]
        if tok.startswith("--") and (i+1) < len(unknown):
            key = tok[2:]
            val = coerce(unknown[i+1])
            set_by_path(cfg, key, val)
            i += 2
        else:
            i += 1


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_precision_context(device, want_precision: str):
    want = str(want_precision).lower()
    use_cuda = (device.type == 'cuda')
    if want == 'bf16' and use_cuda and torch.cuda.is_bf16_supported():
        return torch.amp.autocast('cuda', dtype=torch.bfloat16), False
    if want == 'fp16' and use_cuda:
        return torch.amp.autocast('cuda', enabled=True), True
    class NullCtx:
        def __enter__(self): pass
        def __exit__(self, *args): pass
    return NullCtx(), False


# -------------------- data prep --------------------

def prepare_dataset(config, tokenizer, logger):
    try:
        dataset = load_dataset(config['dataset']['name'], config['dataset']['subset'], split='train')
    except Exception as e:
        if is_rank0():
            logger.error(f"Error loading dataset: {e}")
            logger.info("Using fallback dataset wikitext-2-raw-v1...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split='train')

    max_samples = int(config['dataset'].get('max_samples', 0) or 0)
    if max_samples:
        dataset = dataset.select(range(min(len(dataset), max_samples)))
    if is_rank0():
        logger.info(f"Dataset (raw) size: {len(dataset)}")

    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=False, padding=False)

    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    block_size = int(config['model']['max_length'])
    def group_texts(examples):
        concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = (len(concatenated['input_ids']) // block_size) * block_size
        if total_length == 0:
            return {"input_ids": [], "attention_mask": [], "labels": []}
        input_ids = [concatenated['input_ids'][i:i+block_size] for i in range(0, total_length, block_size)]
        attention_mask = [[1]*block_size for _ in range(len(input_ids))]
        labels = [seq.copy() for seq in input_ids]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    chunked = tokenized.map(group_texts, batched=True)
    if is_rank0():
        logger.info(f"Post-chunk dataset size (sequences): {len(chunked)} (each len={block_size})")
    return chunked


# -------------------- main --------------------

def main():
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None, help="override seed")
    args, unknown = ap.parse_known_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # apply dot overrides
    apply_overrides(config, unknown)
    if args.seed is not None:
        set_by_path(config, 'training.seed', int(args.seed))

    # distributed init
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(local_rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    logger = setup_logging(rank)
    if is_rank0():
        logger.info(f"DDP init: backend={backend}, world_size={world_size}, local_rank={local_rank}")
        logger.info(f"Using device: {device}")

    # seeding
    seed = int(config.get('training', {}).get('seed', 42))
    seed_everything(seed)

    # model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    model = AutoModelForCausalLM.from_pretrained(config['model']['name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

    # data
    dataset = prepare_dataset(config, tokenizer, logger)
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)

    num_workers = int(config['dataset'].get('num_workers', 4))
    prefetch = int(config['dataset'].get('prefetch_factor', 2))
    pin = bool(config['dataset'].get('pin_memory', device.type == 'cuda'))

    dl = DataLoader(
        dataset,
        batch_size=int(config['training']['batch_size']),
        sampler=sampler,
        collate_fn=default_data_collator,
        num_workers=num_workers,
        prefetch_factor=prefetch if num_workers > 0 else None,
        pin_memory=pin,
        persistent_workers=(num_workers > 0)
    )

    # wrap model
    ddp = DDP(
        model,
        device_ids=[local_rank] if device.type == 'cuda' else None,
        output_device=local_rank if device.type == 'cuda' else None,
        find_unused_parameters=bool(config['distributed'].get('find_unused_parameters', False))
    )

    # optim & sched
    lr = float(config['training']['learning_rate'])
    optimizer = torch.optim.AdamW(ddp.parameters(), lr=lr)

    grad_accum = int(config['training'].get('gradient_accumulation_steps', 1))
    max_updates = int(config['training']['max_steps'])  # interpreter: max *optimizer* steps
    warmup = int(config['training']['warmup_steps'])

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup,
        num_training_steps=max_updates
    )

    # precision context and scaler
    amp_ctx, use_fp16_scaler = get_precision_context(device, config['training'].get('precision', 'fp32'))
    scaler = torch.amp.GradScaler('cuda', enabled=use_fp16_scaler)

    # per-run output directory
    run_meta = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
        "world_size": world_size,
        "precision": str(config['training'].get('precision', 'fp32')).lower(),
        "seq_len": int(config['model']['max_length']),
        "batch_size": int(config['training']['batch_size']),
        "grad_accum": grad_accum,
        "seed": seed,
    }
    log_root = Path(config['logging']['log_dir'])
    out_dir = log_root / f"{run_meta['run_id']}_sl{run_meta['seq_len']}_{run_meta['precision']}_seed{seed}"
    if is_rank0():
        out_dir.mkdir(parents=True, exist_ok=True)

    csv_logger = CSVLogger(out_dir / "metrics.csv", fieldnames=[
        "step","loss","step_time_s","batch_tokens","tokens_per_s","avg_tokens_per_s",
        "run_id","world_size","precision","seq_len","batch_size","grad_accum","seed"
    ]) if is_rank0() else None

    # training loop
    ddp.train()
    micro_step = 0
    update_step = 0
    total_tokens = 0
    start_time = time.time()

    if is_rank0():
        logger.info(f"Starting training for {max_updates} optimizer steps (accum={grad_accum})…")

    while update_step < max_updates:
        sampler.set_epoch(update_step)
        for batch in dl:
            t0 = time.time()
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with amp_ctx:
                out = ddp(**batch)
                loss = out.loss / grad_accum

            if use_fp16_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            micro_step += 1
            should_step = (micro_step % grad_accum) == 0
            if should_step:
                if use_fp16_scaler:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                update_step += 1

            # global tokens this micro step (sum attention_mask across ranks)
            rank_tokens = torch.as_tensor(int(batch['attention_mask'].sum().item()), device=device, dtype=torch.long)
            dist.all_reduce(rank_tokens, op=dist.ReduceOp.SUM)
            batch_tokens = int(rank_tokens.item())
            total_tokens += batch_tokens

            # timing & logging (rank 0)
            dt = time.time() - t0
            if is_rank0():
                elapsed = time.time() - start_time
                tps = batch_tokens / dt if dt > 0 else 0.0
                avg_tps = total_tokens / elapsed if elapsed > 0 else 0.0
                logging.info(
                    f"up={update_step}/{max_updates} micro={micro_step} loss={out.loss.item():.4f} "
                    f"tokens/s={tps:.1f} avg_tokens/s={avg_tps:.1f}"
                )
                if csv_logger is not None:
                    csv_logger.log({
                        "step": int(update_step),
                        "loss": float(out.loss.item()),
                        "step_time_s": round(dt, 6),
                        "batch_tokens": int(batch_tokens),
                        "tokens_per_s": round(tps, 2),
                        "avg_tokens_per_s": round(avg_tps, 2),
                        **run_meta,
                    })

            if update_step >= max_updates:
                break

    if is_rank0():
        if csv_logger is not None:
            csv_logger.close()
        logging.info(f"DONE. Logs in: {out_dir}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()