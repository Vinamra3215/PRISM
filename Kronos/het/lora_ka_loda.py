import os
import sys
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Make ~/Kronos importable regardless of where this script is launched from.
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)

from model import Kronos, KronosTokenizer
from model.kronos import calc_time_stamps


# ============================================================
# CONFIG
# ============================================================

TOKENIZER_PATH = (
    "/home/soq/__shutupandbendover/het-uchiha/"
    "weights/Kronos-Tokenizer-base"
)

MODEL_PATH = (
    "/home/soq/__shutupandbendover/het-uchiha/"
    "weights/Kronos-base"
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "het",
    "data",
    "RIL_19-26.parquet",
)

CHECKPOINT_DIR = os.path.join(
    PROJECT_ROOT,
    "het",
    "checkpoints_ours",
)

BEST_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "lora_best.pt",
)

TRAINING_STATE = os.path.join(
    CHECKPOINT_DIR,
    "training_state.pt",
)


# -------------------------
# Dataset
# -------------------------

CUTOFF_DATE = "2021-01-01"

WINDOW_SIZE = 400
TRAIN_SIZE = 320
VAL_SIZE = 80


# -------------------------
# LoRA
# -------------------------

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
)


# -------------------------
# Training
# -------------------------

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01

MAX_EPOCHS = 100
PATIENCE = 10

GRAD_CLIP_NORM = 1.0

CLIP_VALUE = 5.0


# -------------------------
# Device
# -------------------------

DEVICE = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)


# -------------------------
# Reproducibility
# -------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# LoRA
# ============================================================

class LoRALinear(nn.Module):
    """
    Frozen pretrained Linear layer plus a trainable low-rank update.

        y = base(x) + scaling * B(A(dropout(x)))

    B is initialized to zero, meaning the model initially behaves
    exactly like the pretrained model.
    """

    def __init__(
        self,
        base_linear,
        rank,
        alpha,
        dropout,
    ):
        super().__init__()

        if not isinstance(
            base_linear,
            nn.Linear,
        ):
            raise TypeError(
                "LoRALinear requires nn.Linear"
            )

        if rank <= 0:
            raise ValueError(
                "LoRA rank must be positive"
            )

        self.base = base_linear

        # Freeze pretrained layer.
        for parameter in self.base.parameters():
            parameter.requires_grad = False

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Parameter(
            torch.empty(
                rank,
                base_linear.in_features,
            )
        )

        self.lora_B = nn.Parameter(
            torch.zeros(
                base_linear.out_features,
                rank,
            )
        )

        # Standard LoRA initialization:
        # A random, B zero.
        nn.init.kaiming_uniform_(
            self.lora_A,
            a=math.sqrt(5),
        )

        self.dropout = (
            nn.Dropout(dropout)
            if dropout > 0
            else nn.Identity()
        )

    def forward(self, x):

        base_output = self.base(x)

        lora_output = self.dropout(x)

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
            + self.scaling * lora_output
        )


def inject_lora(model):
    """
    Freeze the complete pretrained Kronos model and replace
    q/k/v/out attention projections with LoRA layers.
    """

    # Freeze absolutely everything first.
    for parameter in model.parameters():
        parameter.requires_grad = False

    trainable_parameters = []
    replaced_modules = []

    for block_index, block in enumerate(
        model.transformer
    ):

        attention = block.self_attn

        for target_name in LORA_TARGETS:

            original = getattr(
                attention,
                target_name,
            )

            if not isinstance(
                original,
                nn.Linear,
            ):
                raise TypeError(
                    f"Expected "
                    f"transformer.{block_index}"
                    f".self_attn.{target_name} "
                    f"to be nn.Linear, "
                    f"got {type(original)}"
                )

            lora_layer = LoRALinear(
                original,
                rank=LORA_R,
                alpha=LORA_ALPHA,
                dropout=LORA_DROPOUT,
            )

            setattr(
                attention,
                target_name,
                lora_layer,
            )

            trainable_parameters.extend(
                [
                    lora_layer.lora_A,
                    lora_layer.lora_B,
                ]
            )

            replaced_modules.append(
                f"transformer.{block_index}"
                f".self_attn.{target_name}"
            )

    print("\nLoRA target modules:")

    for name in replaced_modules:
        print(f"  {name}")

    return trainable_parameters


# ============================================================
# DATA LOADING
# ============================================================

PRICE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
]

VOLUME_COLUMN = "volume"
AMOUNT_COLUMN = "amount"


def load_ohlcv():

    print(
        f"\nLoading data from:\n"
        f"{DATA_PATH}"
    )

    df = pd.read_parquet(
        DATA_PATH
    )

    df.index = pd.to_datetime(
        df.index
    )

    if isinstance(
        df.columns,
        pd.MultiIndex,
    ):
        df.columns = (
            df.columns
            .get_level_values(0)
        )

    df.columns = [
        str(column).lower()
        for column in df.columns
    ]

    required_columns = (
        PRICE_COLUMNS
        + [VOLUME_COLUMN]
    )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing required columns: "
            + str(missing)
        )

    # Kronos expects amount as the sixth feature.
    if AMOUNT_COLUMN not in df.columns:

        df[AMOUNT_COLUMN] = (
            df[VOLUME_COLUMN]
            * df[PRICE_COLUMNS].mean(
                axis=1
            )
        )

    return df


def make_400_bar_split(df):

    # Only bars before the cutoff.
    before_cutoff = df[
        df.index < CUTOFF_DATE
    ]

    if len(before_cutoff) < WINDOW_SIZE:
        raise RuntimeError(
            f"Need at least "
            f"{WINDOW_SIZE} bars before "
            f"{CUTOFF_DATE}, but only "
            f"{len(before_cutoff)} exist."
        )

    # Take the most recent 400 bars before
    # the cutoff.
    window = before_cutoff.tail(
        WINDOW_SIZE
    )

    train_df = window.iloc[
        :TRAIN_SIZE
    ].copy()

    val_df = window.iloc[
        TRAIN_SIZE:
    ].copy()

    if len(train_df) != TRAIN_SIZE:
        raise RuntimeError(
            f"Expected {TRAIN_SIZE} "
            f"training bars, got "
            f"{len(train_df)}"
        )

    if len(val_df) != VAL_SIZE:
        raise RuntimeError(
            f"Expected {VAL_SIZE} "
            f"validation bars, got "
            f"{len(val_df)}"
        )

    print("\n400-bar split:")

    print(
        f"Train: "
        f"{train_df.index.min()} "
        f"-> "
        f"{train_df.index.max()} "
        f"({len(train_df)} bars)"
    )

    print(
        f"Validation: "
        f"{val_df.index.min()} "
        f"-> "
        f"{val_df.index.max()} "
        f"({len(val_df)} bars)"
    )

    return train_df, val_df


# ============================================================
# NORMALIZATION / TIMESTAMPS
# ============================================================

def prepare_window(
    window_df,
):

    columns = (
        PRICE_COLUMNS
        + [
            VOLUME_COLUMN,
            AMOUNT_COLUMN,
        ]
    )

    values = window_df[
        columns
    ].values.astype(
        np.float32
    )

    mean = values.mean(
        axis=0
    )

    std = values.std(
        axis=0
    )

    normalized = (
        values - mean
    ) / (
        std + 1e-5
    )

    normalized = np.clip(
        normalized,
        -CLIP_VALUE,
        CLIP_VALUE,
    )

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

    x_tensor = torch.from_numpy(
        normalized[
            np.newaxis,
            :,
            :,
        ]
    ).to(DEVICE)

    stamp_tensor = torch.from_numpy(
        stamp_df.values.astype(
            np.float32
        )[
            np.newaxis,
            :,
            :,
        ]
    ).to(DEVICE)

    return (
        x_tensor,
        stamp_tensor,
    )


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_window(
    tokenizer,
    model,
    window_df,
):

    x, stamp = prepare_window(
        window_df
    )

    with torch.no_grad():

        composite_tokens = (
            tokenizer.encode(x)
        )

        s1_ids, s2_ids = (
            model.embedding.split_token(
                composite_tokens,
                model.embedding.s2_bits,
            )
        )

    return (
        s1_ids,
        s2_ids,
        stamp,
    )


# ============================================================
# LOSS
# ============================================================

def calculate_loss(
    model,
    s1_ids,
    s2_ids,
    stamp,
):

    s1_logits, s2_logits = model(
        s1_ids,
        s2_ids,
        stamp=stamp,
        use_teacher_forcing=True,
        s1_targets=s1_ids,
    )

    # Position t predicts token t+1.
    s1_loss = F.cross_entropy(
        s1_logits[:, :-1, :].reshape(
            -1,
            s1_logits.size(-1),
        ),
        s1_ids[:, 1:].reshape(-1),
    )

    s2_loss = F.cross_entropy(
        s2_logits[:, :-1, :].reshape(
            -1,
            s2_logits.size(-1),
        ),
        s2_ids[:, 1:].reshape(-1),
    )

    # Equal weighting of the two token streams.
    total_loss = (
        s1_loss + s2_loss
    ) / 2.0

    return (
        total_loss,
        s1_loss,
        s2_loss,
    )


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def get_lora_state_dict(
    model,
):

    return {
        name: parameter.detach()
        .cpu()
        .clone()

        for name, parameter
        in model.named_parameters()

        if parameter.requires_grad
    }


def atomic_save(
    object_to_save,
    path,
):

    temporary_path = (
        path + ".tmp"
    )

    torch.save(
        object_to_save,
        temporary_path,
    )

    # Only replace the real checkpoint after
    # torch.save has completed successfully.
    os.replace(
        temporary_path,
        path,
    )


def save_best_checkpoint(
    model,
    epoch,
    val_loss,
):

    checkpoint = {

        "lora_state_dict":
            get_lora_state_dict(
                model
            ),

        "epoch": epoch,

        "val_loss": val_loss,

        "lora_r": LORA_R,

        "lora_alpha":
            LORA_ALPHA,

        "lora_dropout":
            LORA_DROPOUT,

        "target_modules":
            LORA_TARGETS,
    }

    atomic_save(
        checkpoint,
        BEST_CHECKPOINT,
    )

    print(
        f"  BEST checkpoint saved "
        f"(epoch {epoch}, "
        f"val={val_loss:.6f})"
    )


def save_training_state(
    model,
    optimizer,
    epoch,
    best_val_loss,
    patience_counter,
):

    state = {

        "epoch": epoch,

        "best_val_loss":
            best_val_loss,

        "patience_counter":
            patience_counter,

        "lora_state_dict":
            get_lora_state_dict(
                model
            ),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "lora_r": LORA_R,

        "lora_alpha":
            LORA_ALPHA,

        "lora_dropout":
            LORA_DROPOUT,

        "target_modules":
            LORA_TARGETS,
    }

    atomic_save(
        state,
        TRAINING_STATE,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True,
    )

    print("=" * 72)
    print("KRONOS LoRA FINE-TUNING")
    print("=" * 72)

    print(
        f"Device: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"VRAM: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    print(
        "\n[1/6] Loading tokenizer..."
    )

    tokenizer = (
        KronosTokenizer
        .from_pretrained(
            TOKENIZER_PATH
        )
        .to(DEVICE)
    )

    tokenizer.eval()

    # --------------------------------------------------------
    # Load pretrained Kronos
    # --------------------------------------------------------

    print(
        "[2/6] Loading pretrained Kronos..."
    )

    model = (
        Kronos
        .from_pretrained(
            MODEL_PATH
        )
        .to(DEVICE)
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    print(
        "[3/6] Loading Reliance data..."
    )

    df = load_ohlcv()

    train_df, val_df = (
        make_400_bar_split(df)
    )

    # --------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------

    print(
        "\n[4/6] Tokenizing train/validation..."
    )

    train_s1, train_s2, train_stamp = (
        tokenize_window(
            tokenizer,
            model,
            train_df,
        )
    )

    val_s1, val_s2, val_stamp = (
        tokenize_window(
            tokenizer,
            model,
            val_df,
        )
    )

    print(
        f"\nTrain s1 shape: "
        f"{tuple(train_s1.shape)}"
    )

    print(
        f"Train s2 shape: "
        f"{tuple(train_s2.shape)}"
    )

    print(
        f"Validation s1 shape: "
        f"{tuple(val_s1.shape)}"
    )

    print(
        f"Validation s2 shape: "
        f"{tuple(val_s2.shape)}"
    )

    # --------------------------------------------------------
    # LoRA
    # --------------------------------------------------------

    print(
        "\n[5/6] Installing LoRA..."
    )

    lora_parameters = inject_lora(
        model
    )

    model.to(DEVICE)

    total_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter
        in lora_parameters
    )

    print(
        "\nParameter summary:"
    )

    print(
        f"Total:     "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable: "
        f"{trainable_parameters:,}"
    )

    print(
        f"Percentage: "
        f"{100.0 * trainable_parameters / total_parameters:.4f}%"
    )

    # Safety assertion.
    non_lora_trainable = [
        name
        for name, parameter
        in model.named_parameters()
        if parameter.requires_grad
        and "lora_" not in name
    ]

    if non_lora_trainable:

        raise RuntimeError(
            "Found non-LoRA trainable "
            "parameters:\n"
            + "\n".join(
                non_lora_trainable
            )
        )

    print(
        "Trainable-parameter safety "
        "check: PASSED"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        lora_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    best_val_loss = float(
        "inf"
    )

    patience_counter = 0

    start_epoch = 1

    if os.path.exists(
        TRAINING_STATE
    ):

        print(
            "\nExisting training state "
            "found:"
        )

        print(
            TRAINING_STATE
        )

        answer = input(
            "Resume it? [y/N]: "
        ).strip().lower()

        if answer == "y":

            state = torch.load(
                TRAINING_STATE,
                map_location=DEVICE,
            )

            model.load_state_dict(
                state[
                    "lora_state_dict"
                ],
                strict=False,
            )

            optimizer.load_state_dict(
                state[
                    "optimizer_state_dict"
                ]
            )

            start_epoch = (
                state["epoch"] + 1
            )

            best_val_loss = (
                state["best_val_loss"]
            )

            patience_counter = (
                state[
                    "patience_counter"
                ]
            )

            print(
                f"Resuming at epoch "
                f"{start_epoch}"
            )

            print(
                f"Previous best validation "
                f"loss: {best_val_loss:.6f}"
            )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print(
        "\n[6/6] Starting LoRA training..."
    )

    print(
        f"Maximum epochs: {MAX_EPOCHS}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print(
        f"Early stopping patience: "
        f"{PATIENCE}"
    )

    print(
        "\nTraining will use only the "
        "320-bar training window."
    )

    print(
        "The 80-bar validation window "
        "is used only for validation loss."
    )

    print()

    for epoch in range(
        start_epoch,
        MAX_EPOCHS + 1,
    ):

        # ====================================================
        # TRAIN
        # ====================================================

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        train_loss, train_s1_loss, train_s2_loss = (
            calculate_loss(
                model,
                train_s1,
                train_s2,
                train_stamp,
            )
        )

        train_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            lora_parameters,
            GRAD_CLIP_NORM,
        )

        optimizer.step()

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        with torch.no_grad():

            val_loss, val_s1_loss, val_s2_loss = (
                calculate_loss(
                    model,
                    val_s1,
                    val_s2,
                    val_stamp,
                )
            )

        train_value = (
            train_loss.item()
        )

        val_value = (
            val_loss.item()
        )

        print(
            f"Epoch {epoch:3d} | "
            f"Train {train_value:.6f} "
            f"(s1={train_s1_loss.item():.6f}, "
            f"s2={train_s2_loss.item():.6f}) | "
            f"Val {val_value:.6f} "
            f"(s1={val_s1_loss.item():.6f}, "
            f"s2={val_s2_loss.item():.6f})"
        )

        # ====================================================
        # BEST CHECKPOINT
        # ====================================================

        if val_value < best_val_loss:

            best_val_loss = val_value

            patience_counter = 0

            save_best_checkpoint(
                model,
                epoch,
                val_value,
            )

        else:

            patience_counter += 1

        # ====================================================
        # RESUME CHECKPOINT
        # ====================================================

        save_training_state(
            model,
            optimizer,
            epoch,
            best_val_loss,
            patience_counter,
        )

        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if (
            patience_counter
            >= PATIENCE
        ):

            print(
                "\nEarly stopping triggered."
            )

            print(
                f"Validation loss did not "
                f"improve for "
                f"{PATIENCE} epochs."
            )

            break

    # ========================================================
    # RESTORE BEST MODEL
    # ========================================================

    if not os.path.exists(
        BEST_CHECKPOINT
    ):

        raise RuntimeError(
            "Training finished but "
            "no best checkpoint exists."
        )

    print(
        "\nRestoring best LoRA weights..."
    )

    best_checkpoint = torch.load(
        BEST_CHECKPOINT,
        map_location=DEVICE,
    )

    model.load_state_dict(
        best_checkpoint[
            "lora_state_dict"
        ],
        strict=False,
    )

    print(
        f"Best epoch: "
        f"{best_checkpoint['epoch']}"
    )

    print(
        f"Best validation loss: "
        f"{best_checkpoint['val_loss']:.6f}"
    )

    print(
        "\n=================================================="
    )

    print(
        "LoRA TRAINING COMPLETE"
    )

    print(
        "=================================================="
    )

    print(
        f"Best checkpoint:\n"
        f"{BEST_CHECKPOINT}"
    )

    print(
        f"\nTraining-state checkpoint:\n"
        f"{TRAINING_STATE}"
    )


if __name__ == "__main__":
    main()