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

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# DATA
# ============================================================

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "HCLTECH_clean.parquet"
)


# ============================================================
# RESULTS
# ============================================================

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)


# ============================================================
# CHECKPOINT
# ============================================================

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

# Prediction begins from this date.
PREDICTION_START = "2022-01-01"

# Exactly 400 trading days immediately before
# PREDICTION_START will be used for training.
WINDOW_SIZE = 400


# ============================================================
# FEATURES
# ============================================================

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

# REQUIRED:
# 25 epochs
NUM_EPOCHS = 25


# ============================================================
# DEVICE
# ============================================================

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

    print()
    print("Data path:")
    print(DATA_PATH)

    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(
            f"\nHCLTECH data not found:\n{DATA_PATH}"
        )

    df = pd.read_parquet(
        DATA_PATH
    )

    # --------------------------------------------------------
    # Handle Date column
    # --------------------------------------------------------

    if "Date" in df.columns:

        df["Date"] = pd.to_datetime(
            df["Date"]
        )

        df = df.set_index(
            "Date"
        )

    elif "date" in df.columns:

        df["date"] = pd.to_datetime(
            df["date"]
        )

        df = df.set_index(
            "date"
        )

    else:

        df.index = pd.to_datetime(
            df.index
        )

    # --------------------------------------------------------
    # Flatten MultiIndex columns if necessary
    # --------------------------------------------------------

    if isinstance(
        df.columns,
        pd.MultiIndex
    ):

        df.columns = (
            df.columns
            .get_level_values(0)
        )

    # --------------------------------------------------------
    # Lowercase column names
    # --------------------------------------------------------

    df.columns = [
        str(c).lower()
        for c in df.columns
    ]

    # --------------------------------------------------------
    # Required columns
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
        + [
            VOL_COL,
            AMOUNT_COL
        ]
    ].dropna()

    print()
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

    # --------------------------------------------------------
    # Convert prediction date
    # --------------------------------------------------------

    prediction_start = pd.Timestamp(
        PREDICTION_START
    )

    # --------------------------------------------------------
    # Keep ONLY data before 1 Jan 2022
    # --------------------------------------------------------

    train_df = df[
        df.index < prediction_start
    ].copy()

    print()
    print(
        "Rows available before prediction:",
        len(train_df)
    )

    # --------------------------------------------------------
    # Check enough historical data
    # --------------------------------------------------------

    if len(train_df) < WINDOW_SIZE:

        raise RuntimeError(
            f"Only {len(train_df)} rows are "
            f"available before "
            f"{PREDICTION_START}. "
            f"Need {WINDOW_SIZE}."
        )

    # --------------------------------------------------------
    # Take EXACTLY last 400 trading days
    # --------------------------------------------------------

    train_df = train_df.tail(
        WINDOW_SIZE
    ).copy()

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

    # --------------------------------------------------------
    # Verify exactly 400 rows
    # --------------------------------------------------------

    if len(train_df) != WINDOW_SIZE:

        raise RuntimeError(
            "Training window is not exactly "
            "400 rows."
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
            + [
                VOL_COL,
                AMOUNT_COL
            ]
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
    # Clip values
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
    # Convert to tensors
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
        # Initialize A
        # ----------------------------------------------------

        nn.init.kaiming_uniform_(
            self.lora_A,
            a=math.sqrt(5)
        )

        # B remains zero.
        #
        # Therefore initial LoRA update = 0.
        #
        # So initially:
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

        base_output = self.base(
            x
        )

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

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Total loss
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Save only trainable LoRA parameters
    # --------------------------------------------------------

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

    print()
    print("Prediction starts:",
          PREDICTION_START)

    print(
        "Training window:",
        WINDOW_SIZE,
        "trading days"
    )

    print(
        "Number of epochs:",
        NUM_EPOCHS
    )

    print(
        "Learning rate:",
        LEARNING_RATE
    )

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
    # Print exact training dates
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL TRAINING DATA")
    print("=" * 70)

    print(
        "Training start:",
        train_df.index[0]
    )

    print(
        "Training end:",
        train_df.index[-1]
    )

    print(
        "Training rows:",
        len(train_df)
    )

    # --------------------------------------------------------
    # Load pretrained model/tokenizer
    #
    # IMPORTANT:
    # Load on CPU first.
    # Inject LoRA.
    # THEN move complete model to CUDA.
    #
    # This avoids CPU/CUDA mismatch problems.
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
    # Inject LoRA while model is on CPU
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

    # --------------------------------------------------------
    # Start training
    # --------------------------------------------------------

    print()
    print("=" * 70)

    print(
        f"STARTING TRAINING: "
        f"{NUM_EPOCHS} EPOCHS"
    )

    print("=" * 70)

    try:

        for epoch in range(
            1,
            NUM_EPOCHS + 1
        ):

            # ------------------------------------------------
            # Training mode
            # ------------------------------------------------

            model.train()

            # ------------------------------------------------
            # Clear gradients
            # ------------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )

            # ------------------------------------------------
            # Forward pass + loss
            # ------------------------------------------------

            loss, s1_loss, s2_loss = (
                teacher_forced_loss(
                    model,
                    train_s1,
                    train_s2,
                    train_stamp
                )
            )

            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            loss.backward()

            # ------------------------------------------------
            # Update LoRA parameters
            # ------------------------------------------------

            optimizer.step()

            loss_value = loss.item()

            # ------------------------------------------------
            # Store history
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Print epoch result
            # ------------------------------------------------

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
            "Latest training history "
            "was already saved."
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

    print()
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

    print()
    print(
        "Training configuration:"
    )

    print(
        "  Prediction start:",
        PREDICTION_START
    )

    print(
        "  Training days:",
        WINDOW_SIZE
    )

    print(
        "  Epochs:",
        NUM_EPOCHS
    )

    print(
        "  Training start:",
        train_df.index[0]
    )

    print(
        "  Training end:",
        train_df.index[-1]
    )

    print()
    print(
        "Training finished successfully."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()