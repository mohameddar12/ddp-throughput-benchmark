# scripts/train_simple.py
# Basic training script to test on Mac before moving to distributed

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

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_config(config_path="configs/base_config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def prepare_dataset(config, tokenizer):
    """Load and tokenize dataset"""
    logger = logging.getLogger(__name__)
    
    # Load dataset with error handling
    try:
        dataset = load_dataset(
            config['dataset']['name'], 
            config['dataset']['subset'],
            split='train'
        )
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        # Fallback to a simple text dataset
        logger.info("Using fallback dataset...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split='train')
    
    # Take subset for testing
    if config['dataset']['max_samples']:
        dataset = dataset.select(range(min(len(dataset), config['dataset']['max_samples'])))
    
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding=False,
            max_length=config['model']['max_length']
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

def main():
    logger = setup_logging()
    config = load_config()
    
    # Debug: Print the actual config being used
    logger.info(f"Config loaded: max_steps = {config['training']['max_steps']}")
    logger.info(f"Batch size = {config['training']['batch_size']}")
    logger.info(f"Max length = {config['model']['max_length']}")
    
    # Device setup (force CPU for debugging)
    # Temporarily disable MPS to test
    device = torch.device("cpu")
    logger.info("Using CPU for debugging")
    
    # Load model and tokenizer
    logger.info(f"Loading model: {config['model']['name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    model = AutoModelForCausalLM.from_pretrained(config['model']['name'])
    
    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.to(device)
    
    # Prepare dataset
    dataset = prepare_dataset(config, tokenizer)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )
    
    # DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=data_collator,
        num_workers=2  # Adjust based on your Mac
    )
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['training']['learning_rate']))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['training']['warmup_steps'],
        num_training_steps=config['training']['max_steps']
    )
    
    # Training loop with metrics
    model.train()
    total_tokens = 0
    start_time = time.time()
    max_steps = 3  # Force it for debugging
    
    logger.info(f"Starting training for {max_steps} steps...")
    
    for step, batch in enumerate(dataloader):
        if int(step) >= int(max_steps):
            logger.info(f"Reached max_steps ({max_steps}), stopping training")
            break
            
        logger.info(f"Starting step {step}")
        step_start = time.time()
        
        # Move batch to device
        logger.info(f"Moving batch to device...")
        batch = {k: v.to(device) for k, v in batch.items()}
        logger.info(f"Batch moved to device")
        
        # Check for empty batch
        if batch['input_ids'].numel() == 0:
            logger.warning(f"Empty batch at step {step}, skipping...")
            continue
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # Calculate metrics
        step_time = time.time() - step_start
        batch_tokens = batch['input_ids'].numel()
        total_tokens += batch_tokens
        tokens_per_second = batch_tokens / step_time
        
        # Log every 10 steps
        if step % 10 == 0:
            elapsed = time.time() - start_time
            avg_tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
            
            logger.info(
                f"Step {step:4d} | Loss: {loss.item():.4f} | "
                f"Step time: {step_time:.3f}s | "
                f"Tokens/sec: {tokens_per_second:.1f} | "
                f"Avg tokens/sec: {avg_tokens_per_sec:.1f}"
            )
    
    total_time = time.time() - start_time
    avg_throughput = total_tokens / total_time
    
    logger.info(f"\nTraining completed!")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Total tokens: {total_tokens}")
    logger.info(f"Average throughput: {avg_throughput:.1f} tokens/sec")

if __name__ == "__main__":
    main()