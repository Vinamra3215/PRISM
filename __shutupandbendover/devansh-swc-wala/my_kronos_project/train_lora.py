import sys
import torch
import torch.nn as nn
import config

repo_path = getattr(config, "KRONOS_REPO_PATH", "/home/soq/Kronos")
for p in [config.PROJECT_ROOT, repo_path]:
    if p not in sys.path:
        sys.path.insert(0, p)

from dataset import load_training_tokens
from model_utils import get_base_model, get_tokenizer, setup_lora_model, EarlyStopping

def train():
    tokenizer = get_tokenizer()
    _, train_loader, val_loader = load_training_tokens(tokenizer)
    
    base_model = get_base_model()
    model = setup_lora_model(base_model)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=config.PATIENCE, min_delta=config.MIN_DELTA)

    print(f"\n--> Starting LoRA Fine-Tuning ({config.SEQ_LEN}-Step Autoregressive Context)...")
    
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            s1_in = batch['s1_ids'].to(config.DEVICE)
            s2_in = batch['s2_ids'].to(config.DEVICE)
            s1_tgt = batch['s1_targets'].to(config.DEVICE)
            
            optimizer.zero_grad()
            
            # Explicit keyword call matching Kronos forward signature
            outputs = model(s1_ids=s1_in, s2_ids=s2_in)
            
            if isinstance(outputs, dict):
                logits = outputs.get('logits', outputs.get('s1_logits', list(outputs.values())[0]))
            elif hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif hasattr(outputs, 's1_logits'):
                logits = outputs.s1_logits
            elif isinstance(outputs, (tuple, list)):
                logits = outputs[0]
            else:
                logits = outputs

            loss = criterion(logits.reshape(-1, logits.size(-1)), s1_tgt.reshape(-1))
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / max(1, len(train_loader))

        # Validation Step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                s1_in = batch['s1_ids'].to(config.DEVICE)
                s2_in = batch['s2_ids'].to(config.DEVICE)
                s1_tgt = batch['s1_targets'].to(config.DEVICE)
                
                outputs = model(s1_ids=s1_in, s2_ids=s2_in)
                
                if isinstance(outputs, dict):
                    logits = outputs.get('logits', outputs.get('s1_logits', list(outputs.values())[0]))
                elif hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif hasattr(outputs, 's1_logits'):
                    logits = outputs.s1_logits
                elif isinstance(outputs, (tuple, list)):
                    logits = outputs[0]
                else:
                    logits = outputs

                loss = criterion(logits.reshape(-1, logits.size(-1)), s1_tgt.reshape(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        print(f"Epoch [{epoch:02d}/{config.EPOCHS:02d}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save Best Adapter Weights
        early_stopping.check(avg_val_loss, model, config.BEST_ADAPTER_DIR)
        if early_stopping.early_stop:
            print(f"--> Early stopping triggered at Epoch {epoch}.")
            break

if __name__ == "__main__":
    train()