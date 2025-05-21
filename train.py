#!/usr/bin/env python3

import os
import time
import json
import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import load_datasets
from model import TinyLLM
from tokenizer import Tokenizer

def main():
    parser = argparse.ArgumentParser(description='Train TinyLLM model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    args = parser.parse_args()
    num_epochs = args.epochs

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create runs directory and timestamped subdirectory for this run
    runs_dir = "runs"
    os.makedirs(runs_dir, exist_ok=True)
    run_id = time.strftime("run_%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(runs_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Saving checkpoints to {run_dir}")

    # Create or update symlink for latest run
    latest_symlink = os.path.join(runs_dir, "latest")
    try:
        if os.path.islink(latest_symlink) or os.path.exists(latest_symlink):
            os.remove(latest_symlink)
    except Exception:
        pass
    os.symlink(run_id, latest_symlink)

    # Load the datasets
    with open('./tinyshakespeare.txt', 'r') as f:
        data = f.read()
    tokenizer = Tokenizer(data=data)
    train_dataset, test_dataset = load_datasets(data, block_size=256, tokenizer=tokenizer, train_ratio=0.9)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Create the dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Create the model
    model = TinyLLM(
        vocab_size=len(tokenizer),
        block_size=train_dataset.block_size, 
        embedding_size=64,
        head_size=16
    ).to(device)
    
    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # For tracking best model and metrics
    best_val_loss = float('inf')
    best_epoch = -1
    metrics = {"train_loss": [], "val_loss": []}

    # Train the model
    print(f"Starting training for {num_epochs} epochs...\n")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        num_train_batches = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}", leave=False):
            input_seq, target = batch
            input_seq = input_seq.to(device)
            target = target.to(device)
            _, loss = model(input_seq, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            num_train_batches += 1
        avg_train_loss = train_loss / num_train_batches if num_train_batches > 0 else 0.0

        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        num_test_batches = 0
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc=f"Eval {epoch+1}", leave=False):
                input_seq, target = batch
                input_seq = input_seq.to(device)
                target = target.to(device)
                _, loss = model(input_seq, target)
                test_loss += loss.item()
                num_test_batches += 1
        avg_test_loss = test_loss / num_test_batches if num_test_batches > 0 else 0.0

        print(f"Epoch {epoch+1} Summary: Train Loss = {avg_train_loss:.4f}, Test Loss = {avg_test_loss:.4f}")

        # Save metrics
        metrics["train_loss"].append(avg_train_loss)
        metrics["val_loss"].append(avg_test_loss)
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # Save model weights after each epoch
        epoch_ckpt = os.path.join(run_dir, f"model_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), epoch_ckpt)

        # Update best model symlink if this is the best val loss so far
        if avg_test_loss < best_val_loss:
            best_val_loss = avg_test_loss
            best_epoch = epoch + 1
            best_ckpt = os.path.join(run_dir, "best_model.pt")
            # Remove existing symlink if present
            try:
                if os.path.islink(best_ckpt) or os.path.exists(best_ckpt):
                    os.remove(best_ckpt)
            except Exception:
                pass
            os.symlink(f"model_epoch_{best_epoch}.pt", best_ckpt)
    print("Training complete.")

if __name__ == '__main__':
    main()
