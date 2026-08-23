import os
from huggingface_hub import snapshot_download

DEST_BASE_DIR = "/home/soq/__shutupandbendover/devansh-swc-wala/my_kronos_project/models"

MODEL_SAVE_PATH = os.path.join(DEST_BASE_DIR, "Kronos-small")
TOKENIZER_SAVE_PATH = os.path.join(DEST_BASE_DIR, "Kronos-Tokenizer-base")

# Correct Hugging Face Repo IDs
MODEL_REPO_ID = "NeoQuasar/Kronos-small"  # or "NeoQuasar/Kronos-base" if available
TOKENIZER_REPO_ID = "NeoQuasar/Kronos-Tokenizer-base"

def download_assets():
    print(f"--> Downloading Kronos Model ({MODEL_REPO_ID}) to:\n    {MODEL_SAVE_PATH} ...")
    snapshot_download(
        repo_id=MODEL_REPO_ID,
        local_dir=MODEL_SAVE_PATH
    )
    print("✓ Model download completed.\n")

    print(f"--> Downloading Kronos Tokenizer ({TOKENIZER_REPO_ID}) to:\n    {TOKENIZER_SAVE_PATH} ...")
    snapshot_download(
        repo_id=TOKENIZER_REPO_ID,
        local_dir=TOKENIZER_SAVE_PATH
    )
    print("✓ Tokenizer download completed.\n")

if __name__ == "__main__":
    download_assets()