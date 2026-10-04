import sys
import os
import torch
from peft import LoraConfig, get_peft_model, PeftModel
import config

KRONOS_REPO_PATH = "/home/soq/Kronos"
if KRONOS_REPO_PATH not in sys.path:
    sys.path.insert(0, KRONOS_REPO_PATH)
    
sys.path.append(config.PROJECT_ROOT)
from model import Kronos, KronosTokenizer

def get_base_model():
    model = Kronos.from_pretrained(config.BASE_MODEL_PATH)
    return model.to(config.DEVICE)

def get_tokenizer():
    tokenizer = KronosTokenizer.from_pretrained(config.TOKENIZER_PATH)
    return tokenizer

def setup_lora_model(base_model):
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.LORA_TARGET_MODULES,
        lora_dropout=config.LORA_DROPOUT,
        bias="none"
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    return model

def load_finetuned_model(adapter_dir=config.BEST_ADAPTER_DIR):
    base_model = get_base_model()
    fine_tuned_model = PeftModel.from_pretrained(base_model, adapter_dir)
    fine_tuned_model.eval()
    return fine_tuned_model

class EarlyStopping:
    def __init__(self, patience=config.PATIENCE, min_delta=config.MIN_DELTA):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def check(self, val_loss, model, save_dir):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            model.save_pretrained(save_dir)
            print(f"--> Saved lowest validation checkpoint (Val Loss: {val_loss:.6f}) to {save_dir}")
        else:
            self.counter += 1
            print(f"--> EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True