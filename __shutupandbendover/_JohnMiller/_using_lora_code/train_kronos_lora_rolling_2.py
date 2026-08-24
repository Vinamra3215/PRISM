import sys
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

sys.path.append("./__shutupandbendover/_JohnMiller/Kronos")
from model import Kronos, KronosTokenizer
from model.kronos import calc_time_stamps

class RollingStockDataset(Dataset):
    def __init__(self, df, lookback_period=512, pred_len=1):
        self.df = df
        self.lookback_period = lookback_period
        self.pred_len = pred_len
        self.total_seq_len = lookback_period + pred_len
        
    def __len__(self):
        return max(0, len(self.df) - self.total_seq_len + 1)

    def __getitem__(self, idx):
        window_df = self.df.iloc[idx : idx + self.total_seq_len]
        return window_df

def collate_fn(batch_dfs, tokenizer, clip=5.0):
    tensors = []
    stamp_tensors = []
    price_cols = ['open', 'high', 'low', 'close']
    
    for df in batch_dfs:
        df = df.copy()
        
        if 'volume' not in df.columns:
            df['volume'] = 0.0
            df['amount'] = 0.0
        if 'amount' not in df.columns:
            df['amount'] = df['volume'] * df[price_cols].mean(axis=1)
            
        features = df[['open', 'high', 'low', 'close', 'volume', 'amount']].values.astype(np.float32)
        
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        norm_features = (features - mean) / (std + 1e-5)
        norm_features = np.clip(norm_features, -clip, clip)
        tensors.append(torch.from_numpy(norm_features))
        
        time_df = calc_time_stamps(df['timestamps'])
        stamp_tensors.append(torch.tensor(time_df.values, dtype=torch.float32))
        
    batch_features = torch.stack(tensors)
    batch_stamps = torch.stack(stamp_tensors)
    
    with torch.no_grad():
        s1_ids, s2_ids = tokenizer.encode(batch_features, half=True)
        
    return s1_ids, s2_ids, batch_stamps

def train_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device.upper()}")

    tokenizer_path = "/home/soq/__shutupandbendover/_JohnMiller/Kronos/weights/Kronos-Tokenizer-base"
    model_path = "/home/soq/__shutupandbendover/_JohnMiller/Kronos/weights/Kronos-base"
    data_file_path = "./__shutupandbendover/_JohnMiller/stock_data/infosys_5y_1d.parquet"
    output_dir = "./__shutupandbendover/_JohnMiller/Kronos/weights/Infosys_LoRA"

    # Hyperparameters
    LOOKBACK_PERIOD = 512
    PRED_LEN = 1
    BATCH_SIZE = 4
    LEARNING_RATE = 2e-5  # Low learning rate to protect pretrained weights
    MAX_EPOCHS = 20
    PATIENCE = 3

    print("Loading Parquet data...")
    df = pd.read_parquet(data_file_path)
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    df = df.sort_values('timestamps').reset_index(drop=True)
    
    train_mask = (df['timestamps'] >= '2019-06-01') & (df['timestamps'] <= '2023-11-30')
    train_df = df.loc[train_mask].reset_index(drop=True)
    
    val_mask = (df['timestamps'] >= '2021-06-01') & (df['timestamps'] <= '2023-12-31')
    val_df = df.loc[val_mask].reset_index(drop=True)

    print(f"Train Dataset size: {len(train_df)} rows | Validation Dataset size: {len(val_df)} rows")

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
    tokenizer.eval()
    
    train_dataset = RollingStockDataset(train_df, lookback_period=LOOKBACK_PERIOD, pred_len=PRED_LEN)
    val_dataset = RollingStockDataset(val_df, lookback_period=LOOKBACK_PERIOD, pred_len=PRED_LEN)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )

    print("Loading Base Model and configuring LoRA...")
    base_model = Kronos.from_pretrained(model_path).to(device)
    
    for param in base_model.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none"
    )
    
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # early stopping se loop stop kar rahe
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(MAX_EPOCHS):
        model.train()
        total_train_loss = 0
        loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS} [TRAIN]")
        
        for s1_ids, s2_ids, stamps in loop:
            s1_ids = s1_ids.to(device)
            s2_ids = s2_ids.to(device)
            stamps = stamps.to(device)
            
            s1_in, s2_in = s1_ids[:, :-1], s2_ids[:, :-1]
            stamp_in = stamps[:, :-1, :]
            s1_target, s2_target = s1_ids[:, 1:], s2_ids[:, 1:]
            
            optimizer.zero_grad()
            
            s1_logits, s2_logits = model(
                s1_in, 
                s2_in, 
                stamp=stamp_in,
                use_teacher_forcing=True, 
                s1_targets=s1_target
            )
            
            loss_s1 = F.cross_entropy(s1_logits.reshape(-1, s1_logits.size(-1)), s1_target.reshape(-1))
            loss_s2 = F.cross_entropy(s2_logits.reshape(-1, s2_logits.size(-1)), s2_target.reshape(-1))
            
            # macroevents ko jyada weight do as compared to microevents
            loss = 0.8 * loss_s1 + 0.2 * loss_s2
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_dataloader)

        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for s1_ids, s2_ids, stamps in val_dataloader:
                s1_ids = s1_ids.to(device)
                s2_ids = s2_ids.to(device)
                stamps = stamps.to(device)
                
                s1_in, s2_in = s1_ids[:, :-1], s2_ids[:, :-1]
                stamp_in = stamps[:, :-1, :]
                s1_target, s2_target = s1_ids[:, 1:], s2_ids[:, 1:]
                
                s1_logits, s2_logits = model(
                    s1_in, 
                    s2_in, 
                    stamp=stamp_in,
                    use_teacher_forcing=True, 
                    s1_targets=s1_target
                )
                
                loss_s1 = F.cross_entropy(s1_logits.reshape(-1, s1_logits.size(-1)), s1_target.reshape(-1))
                loss_s2 = F.cross_entropy(s2_logits.reshape(-1, s2_logits.size(-1)), s2_target.reshape(-1))
                val_loss = 0.8 * loss_s1 + 0.2 * loss_s2
                total_val_loss += val_loss.item()
                
        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"\nEpoch {epoch+1} Complete | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            best_model_state = copy.deepcopy(model.state_dict())
            print("  --> Validation loss improved. Checkpoint saved.")
        else:
            epochs_without_improvement += 1
            print(f"  --> No improvement. Early stopping counter: {epochs_without_improvement}/{PATIENCE}")
            
            if epochs_without_improvement >= PATIENCE:
                print("\nSTOPLOSS TRIGGERED: Validation loss increased. Overfitting prevented.")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.save_pretrained(output_dir)
    print(f"\nBest LoRA weights successfully saved to: {output_dir}")

if __name__ == "__main__":
    train_model()