import os
import sys
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# KRONOS PATH
# ============================================================

KRONOS_ROOT = "/home/soq/Kronos"

sys.path.insert(0, KRONOS_ROOT)

from model import Kronos, KronosTokenizer
from model.kronos import calc_time_stamps


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "HCLTECH_clean.parquet"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

CHECKPOINT_DIR = os.path.join(
    BASE_DIR,
    "checkpoints"
)

HISTORY_PATH = os.path.join(
    RESULTS_DIR,
    "training_history.csv"
)

CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "lora_best.pt"
)


# ============================================================
# PRETRAINED KRONOS
# ============================================================

TOKENIZER_PATH = (
    "/home/soq/__shutupandbendover/"
    "het-uchiha/weights/Kronos-Tokenizer-base"
)

MODEL_PATH = (
    "/home/soq/__shutupandbendover/"
    "het-uchiha/weights/Kronos-base"
)


# ============================================================
# EXPERIMENT
# ============================================================

TRAIN_START = "2020-05-29"
TRAIN_END = "2021-12-31"

WINDOW_SIZE = 400

PRICE_COLS = [
    "open",
    "high",
    "low",
    "close"
]

VOL_COL = "volume"
AMOUNT_COL = "amount"

CLIP_VALUE = 5.0


# ============================================================
# LORA
# ============================================================

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
)


# ============================================================
# TRAINING
# ============================================================

LEARNING_RATE = 1e-4

NUM_EPOCHS = 20

DEVICE = (
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LOAD HCLTECH DATA
# ============================================================

def load_hcltech():

    print()
    print("=" * 70)
    print("LOADING HCLTECH DATA")
    print("=" * 70)

    df = pd.read_parquet(DATA_PATH)

    # --------------------------------------------------------
    # Handle Date column
    # --------------------------------------------------------

    if "Date" in df.columns:

        df["Date"] = pd.to_datetime(
            df["Date"]
        )

        df = df.set_index("Date")

    elif "date" in df.columns:

        df["date"] = pd.to_datetime(
            df["date"]
        )

        df = df.set_index("date")

    else:

        df.index = pd.to_datetime(
            df.index
        )

    # --------------------------------------------------------
    # Flatten MultiIndex columns if necessary
    # --------------------------------------------------------

    if isinstance(df.columns, pd.MultiIndex):

        df.columns = (
            df.columns
            .get_level_values(0)
        )

    df.columns = [
        str(c).lower()
        for c in df.columns
    ]

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    # --------------------------------------------------------
    # Create amount if not present
    # --------------------------------------------------------

    if "amount" not in df.columns:

        df["amount"] = (
            df["volume"]
            *
            df[
                [
                    "open",
                    "high",
                    "low",
                    "close"
                ]
            ].mean(axis=1)
        )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = df.sort_index()

    # --------------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------------

    df = df[
        PRICE_COLS
        + [VOL_COL, AMOUNT_COL]
    ].dropna()

    print(
        "Total clean rows:",
        len(df)
    )

    print(
        "Overall start:",
        df.index.min()
    )

    print(
        "Overall end:",
        df.index.max()
    )

    return df


# ============================================================
# GET EXACT 400-DAY TRAINING WINDOW
# ============================================================

def get_training_window(df):

    print()
    print("=" * 70)
    print("BUILDING 400-DAY TRAINING WINDOW")
    print("=" * 70)

    train_df = df.loc[
        TRAIN_START:TRAIN_END
    ].copy()

    print(
        "Rows available between dates:",
        len(train_df)
    )

    # --------------------------------------------------------
    # We want exactly the LAST 400 trading days
    # before prediction begins.
    # --------------------------------------------------------

    if len(train_df) < WINDOW_SIZE:

        raise RuntimeError(
            f"Only {len(train_df)} rows available "
            f"between {TRAIN_START} and {TRAIN_END}. "
            f"Need {WINDOW_SIZE}."
        )

    train_df = train_df.tail(
        WINDOW_SIZE
    )

    print()
    print(
        "TRAINING WINDOW START:",
        train_df.index.min()
    )

    print(
        "TRAINING WINDOW END:",
        train_df.index.max()
    )

    print(
        "TRAINING ROWS:",
        len(train_df)
    )

    if len(train_df) != WINDOW_SIZE:

        raise RuntimeError(
            "Training window is not exactly 400 rows."
        )

    return train_df


# ============================================================
# PREPARE INPUT TENSORS
# ============================================================

def prepare_tensors(
    window_df,
    device
):

    # --------------------------------------------------------
    # Six-dimensional OHLCVA input
    # --------------------------------------------------------

    x_raw = (
        window_df[
            PRICE_COLS
            + [VOL_COL, AMOUNT_COL]
        ]
        .values
        .astype(np.float32)
    )

    # --------------------------------------------------------
    # Per-window normalization
    # --------------------------------------------------------

    mean = x_raw.mean(
        axis=0
    )

    std = x_raw.std(
        axis=0
    )

    x_norm = (
        x_raw - mean
    ) / (
        std + 1e-5
    )

    # --------------------------------------------------------
    # Same clipping used by Kronos
    # --------------------------------------------------------

    x_norm = np.clip(
        x_norm,
        -CLIP_VALUE,
        CLIP_VALUE
    )

    # --------------------------------------------------------
    # Timestamp features
    # --------------------------------------------------------

    timestamps = pd.Series(
        pd.to_datetime(
            window_df.index
        )
    ).reset_index(
        drop=True
    )

    stamp_df = calc_time_stamps(
        timestamps
    )

    stamp = stamp_df.values.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Tensor
    # --------------------------------------------------------

    x_tensor = torch.from_numpy(
        x_norm[
            np.newaxis,
            :,
            :
        ]
    ).to(device)

    stamp_tensor = torch.from_numpy(
        stamp[
            np.newaxis,
            :,
            :
        ]
    ).to(device)

    return (
        x_tensor,
        stamp_tensor
    )


# ============================================================
# LORA LINEAR
# ============================================================

class LoRALinear(nn.Module):

    def __init__(
        self,
        base_linear,
        r=8,
        alpha=16,
        dropout=0.05
    ):

        super().__init__()

        self.base = base_linear

        # ----------------------------------------------------
        # Freeze original pretrained layer
        # ----------------------------------------------------

        for p in self.base.parameters():

            p.requires_grad = False

        self.rank = r

        self.alpha = alpha

        self.scaling = (
            alpha / r
        )

        # ----------------------------------------------------
        # LoRA A
        # ----------------------------------------------------

        self.lora_A = nn.Parameter(
            torch.empty(
                r,
                base_linear.in_features
            )
        )

        # ----------------------------------------------------
        # LoRA B
        # ----------------------------------------------------

        self.lora_B = nn.Parameter(
            torch.zeros(
                base_linear.out_features,
                r
            )
        )

        # ----------------------------------------------------
        # Kaiming Uniform
        # ----------------------------------------------------

        nn.init.kaiming_uniform_(
            self.lora_A,
            a=math.sqrt(5)
        )

        # B remains ZERO.
        #
        # Therefore:
        #
        # initial LoRA update = 0
        #
        # so initially:
        #
        # LoRA model = pretrained model
        # ----------------------------------------------------

        if dropout > 0:

            self.dropout = nn.Dropout(
                dropout
            )

        else:

            self.dropout = nn.Identity()

    def forward(self, x):

        base_output = self.base(x)

        lora_output = self.dropout(
            x
        )

        lora_output = (
            lora_output
            @ self.lora_A.t()
        )

        lora_output = (
            lora_output
            @ self.lora_B.t()
        )

        return (
            base_output
            +
            self.scaling
            *
            lora_output
        )


# ============================================================
# INJECT LORA
# ============================================================

def inject_lora(model):

    print()
    print("=" * 70)
    print("INJECTING LoRA")
    print("=" * 70)

    # --------------------------------------------------------
    # Freeze EVERYTHING first
    # --------------------------------------------------------

    for p in model.parameters():

        p.requires_grad = False

    lora_params = []

    lora_count = 0

    # --------------------------------------------------------
    # Kronos Transformer blocks
    # --------------------------------------------------------

    for block in model.transformer:

        attention = block.self_attn

        for name in LORA_TARGETS:

            original = getattr(
                attention,
                name
            )

            wrapped = LoRALinear(
                original,
                r=LORA_R,
                alpha=LORA_ALPHA,
                dropout=LORA_DROPOUT
            )

            setattr(
                attention,
                name,
                wrapped
            )

            lora_params.append(
                wrapped.lora_A
            )

            lora_params.append(
                wrapped.lora_B
            )

            lora_count += 1

    print(
        "LoRA linear modules:",
        lora_count
    )

    trainable = sum(
        p.numel()
        for p in lora_params
    )

    print(
        "Trainable LoRA parameters:",
        f"{trainable:,}"
    )

    return lora_params


# ============================================================
# TEACHER-FORCED LOSS
# ============================================================

def teacher_forced_loss(
    model,
    s1_ids,
    s2_ids,
    stamp_tensor
):

    s1_logits, s2_logits = model(
        s1_ids,
        s2_ids,
        stamp=stamp_tensor,
        use_teacher_forcing=True,
        s1_targets=s1_ids
    )

    # --------------------------------------------------------
    # S1 next-token loss
    # --------------------------------------------------------

    s1_loss = F.cross_entropy(
        s1_logits[:, :-1, :]
        .reshape(
            -1,
            s1_logits.size(-1)
        ),
        s1_ids[:, 1:]
        .reshape(-1)
    )

    # --------------------------------------------------------
    # S2 next-token loss
    # --------------------------------------------------------

    s2_loss = F.cross_entropy(
        s2_logits[:, :-1, :]
        .reshape(
            -1,
            s2_logits.size(-1)
        ),
        s2_ids[:, 1:]
        .reshape(-1)
    )

    total_loss = (
        s1_loss
        +
        s2_loss
    )

    return (
        total_loss,
        s1_loss.item(),
        s2_loss.item()
    )


# ============================================================
# SAVE LoRA CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    path,
    epoch,
    loss
):

    lora_state = {}

    for name, param in model.named_parameters():

        if param.requires_grad:

            lora_state[name] = (
                param.detach()
                .cpu()
                .clone()
            )

    checkpoint = {

        "lora_state_dict":
            lora_state,

        "r":
            LORA_R,

        "alpha":
            LORA_ALPHA,

        "dropout":
            LORA_DROPOUT,

        "target_names":
            LORA_TARGETS,

        "epoch":
            epoch,

        "loss":
            loss
    }

    torch.save(
        checkpoint,
        path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("HCLTECH KRONOS + LoRA TRAINING")
    print("=" * 70)

    print(
        "Device:",
        DEVICE
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # Directories
    # --------------------------------------------------------

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_hcltech()

    train_df = get_training_window(
        df
    )

    # --------------------------------------------------------
    # Load pretrained model/tokenizer
    #
    # IMPORTANT:
    # Load on CPU first.
    # Inject LoRA.
    # THEN move complete model to CUDA.
    #
    # This avoids the CPU/CUDA error we had earlier.
    # --------------------------------------------------------

    print()
    print(
        "Loading pretrained Kronos..."
    )

    tokenizer = (
        KronosTokenizer
        .from_pretrained(
            TOKENIZER_PATH
        )
    )

    model = (
        Kronos
        .from_pretrained(
            MODEL_PATH
        )
    )

    tokenizer.eval()

    # --------------------------------------------------------
    # Prepare training tensors
    # --------------------------------------------------------

    print()
    print(
        "Preparing 400 training bars..."
    )

    train_x, train_stamp = (
        prepare_tensors(
            train_df,
            "cpu"
        )
    )

    # --------------------------------------------------------
    # Encode with frozen tokenizer
    # --------------------------------------------------------

    print(
        "Encoding HCLTECH into Kronos tokens..."
    )

    tokenizer = tokenizer.to(
        "cpu"
    )

    model = model.to(
        "cpu"
    )

    with torch.no_grad():

        train_tokens = tokenizer.encode(
            train_x
        )

        train_s1, train_s2 = (
            model.embedding.split_token(
                train_tokens,
                model.embedding.s2_bits
            )
        )

    # --------------------------------------------------------
    # Inject LoRA WHILE MODEL IS CPU
    # --------------------------------------------------------

    lora_params = inject_lora(
        model
    )

    # --------------------------------------------------------
    # NOW move everything to CUDA
    # --------------------------------------------------------

    model = model.to(
        DEVICE
    )

    train_s1 = train_s1.to(
        DEVICE
    )

    train_s2 = train_s2.to(
        DEVICE
    )

    train_stamp = train_stamp.to(
        DEVICE
    )

    # --------------------------------------------------------
    # Verify devices
    # --------------------------------------------------------

    first_lora_param = (
        next(
            iter(lora_params)
        )
    )

    print()
    print(
        "LoRA parameter device:",
        first_lora_param.device
    )

    print(
        "Training token device:",
        train_s1.device
    )

    print(
        "Model device:",
        next(
            model.parameters()
        ).device
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        lora_params,
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    history = []

    best_loss = float(
        "inf"
    )

    print()
    print("=" * 70)
    print(
        f"STARTING TRAINING: {NUM_EPOCHS} EPOCHS"
    )
    print("=" * 70)

    try:

        for epoch in range(
            1,
            NUM_EPOCHS + 1
        ):

            model.train()

            optimizer.zero_grad(
                set_to_none=True
            )

            loss, s1_loss, s2_loss = (
                teacher_forced_loss(
                    model,
                    train_s1,
                    train_s2,
                    train_stamp
                )
            )

            loss.backward()

            optimizer.step()

            loss_value = loss.item()

            history.append({

                "epoch":
                    epoch,

                "training_loss":
                    loss_value,

                "s1_loss":
                    s1_loss,

                "s2_loss":
                    s2_loss
            })

            print(
                f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
                f"total={loss_value:.6f} | "
                f"s1={s1_loss:.6f} | "
                f"s2={s2_loss:.6f}"
            )

            # ------------------------------------------------
            # Save history after EVERY epoch
            # ------------------------------------------------

            history_df = pd.DataFrame(
                history
            )

            history_df.to_csv(
                HISTORY_PATH,
                index=False
            )

            # ------------------------------------------------
            # Save best checkpoint
            # ------------------------------------------------

            if loss_value < best_loss:

                best_loss = loss_value

                save_checkpoint(
                    model,
                    CHECKPOINT_PATH,
                    epoch,
                    best_loss
                )

                print(
                    f"  -> BEST CHECKPOINT SAVED "
                    f"(loss={best_loss:.6f})"
                )

    except KeyboardInterrupt:

        print()
        print(
            "Training interrupted."
        )

        print(
            "Latest training history was already saved."
        )

        print(
            "Best checkpoint:",
            CHECKPOINT_PATH
        )

        return

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        "Best loss:",
        f"{best_loss:.6f}"
    )

    print(
        "Checkpoint:",
        CHECKPOINT_PATH
    )

    print(
        "History:",
        HISTORY_PATH
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
# Make ~/Kronos/model importable regardless of where this script is run from
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model import Kronos, KronosTokenizer
from model.kronos import calc_time_stamps

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TOKENIZER_PATH = "/home/soq/__shutupandbendover/het-uchiha/weights/Kronos-Tokenizer-base"
MODEL_PATH     = "/home/soq/__shutupandbendover/het-uchiha/weights/Kronos-base"

DATA_PATH   = os.path.join(os.path.dirname(__file__), "data", "HCLTECH_clean.parquet")
CUTOFF_DATE = "2021-01-01"
WINDOW_SIZE = 400
TRAIN_SIZE  = 320
VAL_SIZE    = 80

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

LORA_R       = 8
LORA_ALPHA   = 16
LORA_DROPOUT = 0.05
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "out_proj")

LEARNING_RATE = 1e-4
NUM_STEPS     = 1000   # upper bound; early stopping will likely halt sooner
VAL_EVERY     = 10
PATIENCE      = 5     # stop after this many val checks in a row with no improvement
CLIP_VALUE    = 5     # matches KronosPredictor default clip

PRICE_COLS = ["open", "high", "low", "close"]
VOL_COL    = "volume"
AMT_COL    = "amount"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# LoRA layer
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable low-rank update.

    y = base(x) + scaling * dropout(x) @ A^T @ B^T
    B is zero-initialized so the wrapped layer starts identical to the
    original (delta = 0) before any training happens.
    """
    def __init__(self, base_linear: nn.Linear, r=8, alpha=16, dropout=0.05):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        in_features = base_linear.in_features
        out_features = base_linear.out_features
        self.scaling = alpha / r

        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B left at zero -> initial forward pass is unchanged

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        base_out = self.base(x)
        lora_out = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return base_out + self.scaling * lora_out


def inject_lora(model, r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT,
                 target_names=LORA_TARGETS):
    """Freezes all base model params, then replaces target attention
    Linear layers in every TransformerBlock with LoRALinear wrappers.
    Returns the list of newly created trainable LoRA parameters.
    """
    for p in model.parameters():
        p.requires_grad = False

    lora_params = []
    for block in model.transformer:
        attn = block.self_attn
        for name in target_names:
            orig = getattr(attn, name)
            wrapped = LoRALinear(orig, r=r, alpha=alpha, dropout=dropout)
            setattr(attn, name, wrapped)
            lora_params.append(wrapped.lora_A)
            lora_params.append(wrapped.lora_B)
    return lora_params


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------
def load_ohlcv(path):
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]

    if AMT_COL not in df.columns:
        df[AMT_COL] = df[VOL_COL] * df[PRICE_COLS].mean(axis=1)

    return df


def get_train_val_windows(df):
    before = df[df.index < CUTOFF_DATE]
    window = before.tail(WINDOW_SIZE)
    assert len(window) == WINDOW_SIZE, f"Expected {WINDOW_SIZE} bars, got {len(window)}"

    train = window.iloc[:TRAIN_SIZE]
    val = window.iloc[TRAIN_SIZE:]
    assert len(train) == TRAIN_SIZE and len(val) == VAL_SIZE
    return train, val


def prepare_tensors(window_df, device, clip=CLIP_VALUE):
    """Per-window normalize + clip (mirrors KronosPredictor.predict), then
    returns raw float tensors ready for tokenizer.encode()."""
    x_raw = window_df[PRICE_COLS + [VOL_COL, AMT_COL]].values.astype(np.float32)
    x_mean = x_raw.mean(axis=0)
    x_std = x_raw.std(axis=0)
    x_norm = (x_raw - x_mean) / (x_std + 1e-5)
    x_norm = np.clip(x_norm, -clip, clip)

    timestamps = pd.Series(pd.to_datetime(window_df.index)).reset_index(drop=True)
    stamp_df = calc_time_stamps(timestamps)
    stamp = stamp_df.values.astype(np.float32)

    x_tensor = torch.from_numpy(x_norm[np.newaxis, :, :]).to(device)      # (1, seq_len, 6)
    stamp_tensor = torch.from_numpy(stamp[np.newaxis, :, :]).to(device)   # (1, seq_len, 5)
    return x_tensor, stamp_tensor


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def teacher_forced_loss(model, s1_ids, s2_ids, stamp_tensor):
    """Next-token cross-entropy on both token streams."""
    s1_logits, s2_logits = model(
        s1_ids, s2_ids, stamp=stamp_tensor,
        use_teacher_forcing=True, s1_targets=s1_ids,
    )
    s1_loss = F.cross_entropy(
        s1_logits[:, :-1, :].reshape(-1, s1_logits.size(-1)),
        s1_ids[:, 1:].reshape(-1),
    )
    s2_loss = F.cross_entropy(
        s2_logits[:, :-1, :].reshape(-1, s2_logits.size(-1)),
        s2_ids[:, 1:].reshape(-1),
    )
    return s1_loss + s2_loss, s1_loss.item(), s2_loss.item()


def save_checkpoint(model, path, step, val_loss):
    """Synchronous save — completes before returning, so it's safe against
    a dropped connection immediately after this call finishes."""
    lora_state = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save({
        "lora_state_dict": lora_state,
        "r": LORA_R,
        "alpha": LORA_ALPHA,
        "dropout": LORA_DROPOUT,
        "target_names": LORA_TARGETS,
        "step": step,
        "val_loss": val_loss,
    }, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Device: {DEVICE}")

    print("Loading tokenizer and model (frozen pretrained weights)...")
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_PATH).to(DEVICE)
    model = Kronos.from_pretrained(MODEL_PATH).to(DEVICE)
    tokenizer.eval()

    print("Loading data and building 400-bar window...")
    df = load_ohlcv(DATA_PATH)
    train_df, val_df = get_train_val_windows(df)
    print(f"Train window: {train_df.index.min()} -> {train_df.index.max()} ({len(train_df)} bars)")
    print(f"Val window:   {val_df.index.min()} -> {val_df.index.max()} ({len(val_df)} bars)")

    print("Encoding train/val windows to tokens (frozen tokenizer, no grad)...")
    train_x, train_stamp = prepare_tensors(train_df, DEVICE)
    val_x, val_stamp = prepare_tensors(val_df, DEVICE)
    with torch.no_grad():
        train_tokens = tokenizer.encode(train_x)
        val_tokens = tokenizer.encode(val_x)
        train_s1, train_s2 = model.embedding.split_token(train_tokens, model.embedding.s2_bits)
        val_s1, val_s2 = model.embedding.split_token(val_tokens, model.embedding.s2_bits)

    print("Injecting LoRA into attention projections (q/k/v/out)...")
    lora_params = inject_lora(model)
    model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in lora_params)
    print(f"Total params:     {total_params:,}")
    print(f"Trainable (LoRA): {trainable_params:,} ({100*trainable_params/total_params:.3f}%)")

    optimizer = torch.optim.AdamW(lora_params, lr=LEARNING_RATE)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "lora_best.pt")
    best_val_loss = float("inf")
    checks_without_improvement = 0
    stopped_early = False
    last_completed_step = 0

    print(f"\nTraining for up to {NUM_STEPS} full-batch steps on the 320-bar window...")
    print(f"Early stopping: patience={PATIENCE} validation checks with no improvement.")
    print("If the connection drops, the best checkpoint so far is already saved to disk.\n")

    try:
        for step in range(1, NUM_STEPS + 1):
            model.train()
            optimizer.zero_grad()
            loss, s1_l, s2_l = teacher_forced_loss(model, train_s1, train_s2, train_stamp)
            loss.backward()
            optimizer.step()
            last_completed_step = step

            if step % VAL_EVERY == 0 or step == 1 or step == NUM_STEPS:
                model.eval()
                with torch.no_grad():
                    val_loss, val_s1_l, val_s2_l = teacher_forced_loss(model, val_s1, val_s2, val_stamp)
                print(f"step {step:4d} | train loss {loss.item():.4f} (s1 {s1_l:.4f} s2 {s2_l:.4f}) "
                      f"| val loss {val_loss.item():.4f} (s1 {val_s1_l:.4f} s2 {val_s2_l:.4f})")

                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    checks_without_improvement = 0
                    save_checkpoint(model, ckpt_path, step, best_val_loss)
                    print(f"  -> saved new best checkpoint to {ckpt_path} (val loss {best_val_loss:.4f})")
                else:
                    checks_without_improvement += 1
                    print(f"  -> no improvement ({checks_without_improvement}/{PATIENCE})")
                    if checks_without_improvement >= PATIENCE:
                        print(f"\nEarly stopping at step {step}: "
                              f"val loss hasn't improved in {PATIENCE} checks.")
                        stopped_early = True
                        break

    except KeyboardInterrupt:
        print(f"\nInterrupted manually at step {last_completed_step}. "
              f"Best checkpoint already on disk at {ckpt_path} (val loss {best_val_loss:.4f}).")
        return

    print(f"\n{'Stopped early' if stopped_early else 'Completed all steps'}. "
          f"Best val loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
