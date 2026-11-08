from pathlib import Path
import gc
import json
import logging
import warnings

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl

from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_forecasting import NaNLabelEncoder, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE, MAPE, RMSE, SMAPE, QuantileLoss
from torchmetrics.regression import R2Score

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)


PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_VAL_FILE = PROJECT_DIR / "Data_2016_2023.xlsx"
TEST_FILE = PROJECT_DIR / "Data_2024.xlsx"
MODEL_DIR = PROJECT_DIR / "models"

MAX_ENCODER_LENGTH = 500
MAX_PREDICTION_LENGTH = 72
BATCH_SIZE = 64

LEARNING_RATE = 1e-3
MAX_EPOCHS = 1  # Configured for testing

MODEL_SUFFIX = "72h_test"
TARGET_VARIABLES = ["NO2", "SO2", "CO", "NOX", "RH", "TMP", "Global", "WSP"]

def load_and_prepare_data(series_name):
    df_dev = pd.read_excel(TRAIN_VAL_FILE)
    df_test = pd.read_excel(TEST_FILE)

    for frame in (df_dev, df_test):
        frame["datetime"] = pd.to_datetime(frame["date"])
        frame.sort_values("datetime", inplace=True)
        frame.reset_index(drop=True, inplace=True)

        frame["hour"] = pd.to_numeric(frame["hour"])
        frame["serie"] = series_name

    df_train = df_dev.loc[df_dev["year"] <= 2022].copy().reset_index(drop=True)
    df_val = df_dev.loc[df_dev["year"] == 2023].copy().reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    for frame in (df_train, df_val, df_test):
        frame["time_idx"] = np.arange(len(frame), dtype=np.int64)

    return df_train, df_val, df_test


def standardize_from_training(df_train, df_val, df_test, variable):
    """Apply training-period Z-score statistics to all partitions."""
    mean = df_train[variable].mean()
    std = df_train[variable].std()

    for frame in (df_train, df_val, df_test):
        frame[variable] = (frame[variable] - mean) / std

    return {
        variable: {
            "mean": float(mean),
            "std": float(std),
        }
    }



class PrintMetricsCallback(Callback):
    """Print validation metrics at the end of each validation epoch."""

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        metrics = []
        for name, value in trainer.callback_metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            if isinstance(value, (float, int)):
                metrics.append(f"{name}: {value:.4f}")

        if metrics:
            print(
                f"Epoch {trainer.current_epoch:02d} | "
                + " | ".join(metrics)
            )


def make_trainer(checkpoint_dir, checkpoint_name, patience):
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=checkpoint_name,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=patience,
        verbose=True,
        mode="min",
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=MAX_EPOCHS,
        precision=32,
        gradient_clip_val=0.1,
        callbacks=[
            checkpoint_callback,
            PrintMetricsCallback(),
            early_stop_callback,
        ],
        accumulate_grad_batches=2,
        enable_progress_bar=True,
        log_every_n_steps=10,
        logger=False,
    )

    return trainer, checkpoint_callback


def make_dataset(training_frame, target):
    return TimeSeriesDataSet(
        training_frame,
        time_idx="time_idx",
        target=target,
        group_ids=["serie"],
        min_encoder_length=MAX_ENCODER_LENGTH,
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        min_prediction_length=MAX_PREDICTION_LENGTH,
        static_categoricals=[],
        static_reals=[],
        time_varying_known_categoricals=[],
        time_varying_unknown_reals=[target],
        time_varying_known_reals=["time_idx", "hour"],
        add_relative_time_idx=True,
        add_target_scales=True,
        categorical_encoders={
            "serie": NaNLabelEncoder(add_nan=True),
        },
    )


def make_dataloaders(training_dataset, validation_dataset):
    train_loader = training_dataset.to_dataloader(
        train=True,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )
    val_loader = validation_dataset.to_dataloader(
        train=False,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )
    return train_loader, val_loader



def train_variable_tft(target):
    model_name = f"{target}_{MODEL_SUFFIX}"
    print(f"Training TFT: {target}")

    df_train, df_val, df_test = load_and_prepare_data(series_name=target)

    normalization_stats = standardize_from_training(
        df_train,
        df_val,
        df_test,
        target,
    )

    ts_train = make_dataset(
        df_train,
        target=target,
    )

    ts_val = TimeSeriesDataSet.from_dataset(
        ts_train,
        df_val,
        predict=True,
        stop_randomization=True,
    )

    ts_test = TimeSeriesDataSet.from_dataset(
        ts_train,
        df_test,
        predict=True,
        stop_randomization=True,
    )

    train_loader, val_loader = make_dataloaders(ts_train, ts_val)

    model_dir = MODEL_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = model_dir / f"tsd_{model_name}.pth"
    ts_train.save(dataset_path)
    with (model_dir / "normalization_stats.json").open("w", encoding="utf-8") as file:
        json.dump(normalization_stats, file, indent=4)

    logging_metrics = [
        SMAPE(),
        MAE(),
        RMSE(),
        MAPE(),
        R2Score(),
    ]

    tft = TemporalFusionTransformer.from_dataset(
        ts_train,
        learning_rate=LEARNING_RATE,
        hidden_size=128,
        lstm_layers=2,
        attention_head_size=4,
        dropout=0.4,
        hidden_continuous_size=4,
        output_size=7,
        loss=QuantileLoss(),
        log_interval=100,
        log_val_interval=4,
        reduce_on_plateau_patience=10,
        logging_metrics=logging_metrics,
    )

    print(f"Number of parameters: {tft.size() / 1e6:.2f} M")
    print(f"Training windows:   {len(ts_train):,}")
    print(f"Validation windows: {len(ts_val):,}")
    print(f"Encoder:            {MAX_ENCODER_LENGTH} h")
    print(f"Forecast horizon:   {MAX_PREDICTION_LENGTH} h")
    print(f"Training period:    {df_train['year'].min()}–{df_train['year'].max()}")
    print(f"Validation period:  {df_val['year'].min()}–{df_val['year'].max()}")
    print(f"Held-out test:      {df_test['year'].min()}–{df_test['year'].max()}")

    trainer, checkpoint_callback = make_trainer(
        checkpoint_dir=model_dir,
        checkpoint_name=f"tft_{model_name}",
        patience=20,
    )

    gc.collect()
    torch.cuda.empty_cache()

    trainer.fit(
        tft,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Training dataset: {dataset_path}")

    del trainer, tft, train_loader, val_loader, ts_train, ts_val, ts_test
    gc.collect()
    torch.cuda.empty_cache()


def main():
    for target in TARGET_VARIABLES:
        train_variable_tft(target)


if __name__ == "__main__":
    main()
