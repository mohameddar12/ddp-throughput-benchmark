# scripts/train_simple.py
# Basic training script to test on Mac before moving to distributed

# [EXPLAIN] Imports: core Python timing, PyTorch + YAML for config,
# Hugging Face Transformers for model/tokenizer, Datasets for data,
# DataLoader for batching, logging for readable console output.
import time
import torch
import yaml
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from torch.utils.data import DataLoader
import logging
from itertools import chain
from transformers import default_data_collator
import os, csv
from pathlib import Path
from datetime import datetime

class CSVLogger:
    def __init__(self, out_path, fieldnames):
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        self._f = open(self.out_path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=fieldnames)
        self._w.writeheader()

    def log(self, row: dict):
        self._w.writerow(row)
        self._f.flush()  # make sure it hits disk each step

    def close(self):
        try: self._f.close()
        except Exception: pass

# [EXPLAIN] Configure Python’s logging (INFO level + simple format).
# Returning a module logger lets us reuse it anywhere in this file.
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def is_rank0():
    try:
        import torch.distributed as dist
        return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
    except Exception:
        return True

# [EXPLAIN] Load the YAML file from configs/ (defaults to base_config.yaml).
# Safe to call at program start to centralize all knobs.
def load_config(config_path="configs/base_config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)



# [EXPLAIN] Prepare the dataset end‑to‑end:
# 1) load HuggingFace dataset split
# 2) (optionally) cut to a small subset for fast local runs
# 3) tokenize text into input IDs using your model’s tokenizer
# 4) return a tokenized Dataset object
# Note: all heavy work (like tokenization) should be outside the train loop.
def prepare_dataset(config, tokenizer):
    """Load, tokenize, and chunk dataset into fixed-size sequences."""
    logger = logging.getLogger(__name__)

    # Load dataset by name/subset from config
    try:
        dataset = load_dataset(
            config['dataset']['name'], 
            config['dataset']['subset'],
            split='train'
        )
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        logger.info("Using fallback dataset wikitext-2-raw-v1...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split='train')

    # Small subset for quick tests
    if config['dataset']['max_samples']:
        dataset = dataset.select(range(min(len(dataset), config['dataset']['max_samples'])))

    logger.info(f"Dataset (raw) size: {len(dataset)}")

    # Tokenize without truncation/padding; we’ll size in the chunking step
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=False, padding=False)

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    # Group tokens into contiguous blocks of exactly block_size
    block_size = int(config['model']['max_length'])

    def group_texts(examples):
        concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = (len(concatenated['input_ids']) // block_size) * block_size
        if total_length == 0:
            return {"input_ids": [], "attention_mask": [], "labels": []}
        input_ids = [concatenated['input_ids'][i:i+block_size] for i in range(0, total_length, block_size)]
        attention_mask = [[1] * block_size for _ in range(len(input_ids))]
        labels = [seq.copy() for seq in input_ids]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    chunked = tokenized.map(group_texts, batched=True)
    logger.info(f"Post-chunk dataset size (sequences): {len(chunked)} (each len={block_size})")
    return chunked

# [EXPLAIN] Main entry point: loads config, sets device, builds model/tokenizer,
# constructs dataloader, then runs a short training loop while timing throughput.
def main():
    logger = setup_logging()
    config = load_config()
    
    # [EXPLAIN] Print a few key config values for sanity checking.
    logger.info(f"Config loaded: max_steps = {config['training']['max_steps']}")
    logger.info(f"Batch size = {config['training']['batch_size']}")
    logger.info(f"Max length = {config['model']['max_length']}")
    
    # [EXPLAIN] Device setup. Currently **forced to CPU** for debugging.
    # On a Mac with Apple Silicon, you’ll want to use MPS later:
    #   device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    # For NVIDIA GPUs on Linux, you’d use CUDA:
    #   device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # For now, it’s CPU so everything is slow but reliable.
    device = torch.device("cpu")
    logger.info("Using CPU for debugging")
    
    # [EXPLAIN] Load a small pretrained Causal LM (GPT‑2) + its tokenizer.
    # This gives you a working model without custom architecture code.
    logger.info(f"Loading model: {config['model']['name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    model = AutoModelForCausalLM.from_pretrained(config['model']['name'])
    
    # [EXPLAIN] Some GPT‑2 checkpoints don’t define a pad token.
    # We set it to EOS so padding behaves sanely for batching.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # [EXPLAIN] Move the model weights to the selected device (CPU for now).
    model.to(device)
    
    # [EXPLAIN] Build the tokenized dataset according to config.
    dataset = prepare_dataset(config, tokenizer)
    
    # [EXPLAIN] Collator assembles a batch and pads to the longest sequence in that batch.
    # For causal LM, mlm=False (we’re not doing masked‑LM like BERT).
    data_collator = default_data_collator

    # [EXPLAIN] DataLoader handles batching + worker processes for prefetching.
    # num_workers>0 uses background workers to load/prepare batches for you.
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=data_collator,
        num_workers=2  # Adjust based on your Mac
    )
    
    # [EXPLAIN] AdamW is the common optimizer. Scheduler warms up LR then decays linearly.
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['training']['learning_rate']))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['training']['warmup_steps'],
        num_training_steps=config['training']['max_steps']
    )
    
    # [EXPLAIN] Training loop state: running averages + timers for throughput.
    model.train()
    total_tokens = 0
    start_time = time.time()
    
    # [EXPLAIN] For debugging, training is capped to 3 steps **in code**.
    # You’ll remove this and trust the YAML’s training.max_steps soon.
    max_steps = int(config["training"]["max_steps"])
    logger.info(f"Starting training for {max_steps} steps...")
    
        # Compose run metadata (handy for later analysis)
    run_meta = {
        "run_id": datetime.utcnow().strftime("%Y%m%d-%H%M%S"),
        "world_size": int(os.environ.get("WORLD_SIZE", 1)),
        "precision": str(config["training"]["precision"]).lower(),
        "seq_len": int(config["model"]["max_length"]),
        "batch_size": int(config["training"]["batch_size"]),
        "grad_accum": int(config["training"]["gradient_accumulation_steps"]),
    }

    csv_path = Path(config["logging"]["log_dir"]) / f"metrics_{run_meta['run_id']}.csv"

    # Only rank 0 writes (safe on single GPU too)
    csv_logger = CSVLogger(csv_path, fieldnames=[
        "step","loss","step_time_s","batch_tokens","tokens_per_s","avg_tokens_per_s",
        "run_id","world_size","precision","seq_len","batch_size","grad_accum"
    ]) if is_rank0() else None

    # [EXPLAIN] Enumerate over batches; we stop once we hit max_steps.
    for step, batch in enumerate(dataloader):
        if int(step) >= int(max_steps):
            logger.info(f"Reached max_steps ({max_steps}), stopping training")
            break
        
        logger.info(f"Starting step {step}")
        step_start = time.time()
        
        # [EXPLAIN] Move the whole batch to the target device.
        # (In DDP, each rank would do this to its own per‑GPU batch.)
        logger.info(f"Moving batch to device...")
        batch = {k: v.to(device) for k, v in batch.items()}
        logger.info(f"Batch moved to device")
        
        # [EXPLAIN] Basic guard: if somehow the collator produced an empty batch, skip.
        if batch['input_ids'].numel() == 0:
            logger.warning(f"Empty batch at step {step}, skipping...")
            continue
        
        # [EXPLAIN] Forward pass: run model → get loss comparing predictions vs labels.
        outputs = model(**batch)
        loss = outputs.loss
        
        # [EXPLAIN] Backward pass + optimizer step + LR schedule update.
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # [EXPLAIN] Metrics: time for this step, how many tokens processed, instantaneous
        # tokens/sec. Note: using input_ids.numel() counts **padded tokens** too; we’ll
        # switch to attention_mask.sum() in the “minimal edits” version below to count
        # only real tokens.
        step_time = time.time() - step_start
        batch_tokens = int(batch['attention_mask'].sum().item())
        total_tokens += batch_tokens
        tokens_per_second = batch_tokens / step_time
        
        # Calculate metrics for logging
        elapsed = time.time() - start_time
        avg_tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0

        # [EXPLAIN] Log a summary every 10 steps, plus a running average tokens/sec.
        if step % 10 == 0:
            logger.info(
                f"Step {step:4d} | Loss: {loss.item():.4f} | "
                f"Step time: {step_time:.3f}s | "
                f"Tokens/sec: {tokens_per_second:.1f} | "
                f"Avg tokens/sec: {avg_tokens_per_sec:.1f}"
            )

        # Log to CSV every step
        if csv_logger is not None:
            csv_logger.log({
                "step": int(step),
                "loss": float(loss.item()),
                "step_time_s": round(step_time, 6),
                "batch_tokens": int(batch_tokens),
                "tokens_per_s": round(tokens_per_second, 2),
                "avg_tokens_per_s": round(avg_tokens_per_sec, 2),
                **run_meta,
            })
    
    # [EXPLAIN] Final aggregate stats for the whole run.
    total_time = time.time() - start_time
    avg_throughput = total_tokens / total_time
    
    logger.info(f"\nTraining completed!")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Total tokens: {total_tokens}")
    logger.info(f"Average throughput: {avg_throughput:.1f} tokens/sec")

if __name__ == "__main__":
    main()