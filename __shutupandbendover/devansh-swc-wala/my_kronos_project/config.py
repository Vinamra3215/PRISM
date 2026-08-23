import os
import torch

# ==========================================
# System & Hardware Configuration
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# Core Directory & Project Paths
# ==========================================
PROJECT_ROOT = "/home/soq/__shutupandbendover/devansh-swc-wala/my_kronos_project"
KRONOS_REPO_PATH = "/home/soq/Kronos"
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "lora_adapters")

# Ensure required directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ==========================================
# Pretrained Model & Tokenizer Paths
# ==========================================
BASE_MODEL_PATH = os.path.join(PROJECT_ROOT, "models/Kronos-small")
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "models/Kronos-Tokenizer-base")
BEST_ADAPTER_DIR = os.path.join(CHECKPOINT_DIR, "best_reliance_lora")
os.makedirs(BEST_ADAPTER_DIR, exist_ok=True)

# ==========================================
# Data File Paths
# ==========================================
# Standard Reliance Datasets
DATA_TRAIN_PATH = os.path.join(DATA_DIR, "reliance_2019_2020.parquet")
DATA_EVAL_PATH = os.path.join(DATA_DIR, "reliance_2021_2025.parquet")

# NIFTY 50 (2022–2026) Dataset
NIFTY_DATA_PATH = os.path.join(DATA_DIR, "nifty50_2022_2026.parquet")

# Walk-Forward Output Artifacts
WALK_FORWARD_OUTPUT_HTML = os.path.join(OUTPUT_DIR, "nifty50_walk_forward_comparison.html")
WALK_FORWARD_CSV = os.path.join(OUTPUT_DIR, "nifty50_walk_forward_results.csv")

# ==========================================
# Context & Sequence Window Parameters
# ==========================================
SEQ_LEN = 32     
STEP_SEQ_LEN = 32           # Patch sequence length for discrete token processing
LOOKBACK_WINDOW = 400       # Standard context lookback length
MAX_CONTEXT = 512           # Maximum positional embedding context ceiling
ROLLING_TRAIN_WINDOW = 442  # 442-day rolling context for walk-forward training
 
# ==========================================
# Global Training Hyperparameters
# ==========================================
BATCH_SIZE = 16
EPOCHS = 50
ADAPT_EPOCHS = 5           # Rapid adaptation epochs per walk-forward rolling step
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4

# ==========================================
# Early Stopping Parameters
# ==========================================
PATIENCE = 12
MIN_DELTA = 1e-4

# ==========================================
# LoRA (PEFT) Hyperparameters
# ==========================================
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "out_proj", "c_attn", "c_proj"]