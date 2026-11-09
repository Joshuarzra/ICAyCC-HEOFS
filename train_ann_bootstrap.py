import os
from pathlib import Path
import sys
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from scipy.stats import pearsonr


PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from red_neuronal import RedNeuronalSimple
ANN_EXCEDANCE_DATA = PROJECT_DIR / "Data_excedence_2016_2023.xlsx"
OUTPUT_DIR = PROJECT_DIR / "C-ANN results"
BOOTSTRAP_DIR = OUTPUT_DIR / "Bootstrap datasets"
MODEL_DIR = PROJECT_DIR / "models"

ANN_FEATURES = [
    "month",
    "hour",
    "NO2",
    "SO2",
    "CO",
    "NOX",
    "RH",
    "TMP",
    "Global",
    "WSP"
]

ANN_TARGET = "O3"
ANN_SPECIAL_HIDDEN = 60 # Set for testing purposes; can be adjusted for final training
ANN_LEARNING_RATE = 0.001
ANN_MAX_EPOCHS = 10000
ANN_BATCH_SIZE = 64
ANN_PATIENCE = 500
ANN_MIN_DELTA = 0.0001
BOOTSTRAP_SIZES = [2000] 
GLOBAL_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BOOTSTRAP_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)



def set_seed(seed=GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_ann_data(file_path, features=ANN_FEATURES, target=ANN_TARGET):
    df = pd.read_excel(file_path, engine="openpyxl")
    df_model = df[features + [target]].copy()
    df_model = df_model.dropna().reset_index(drop=True)

    for col in features + [target]:
        df_model[col] = pd.to_numeric(df_model[col], errors="coerce")

    df_model = df_model.dropna().reset_index(drop=True)
    X = df_model[features].values.astype(np.float32)
    Y = df_model[target].values.astype(np.float32).reshape(-1, 1)

    print(f"File: {file_path}")
    print(f"Samples: {len(X):,}")
    print(f"Features: {X.shape[1]}")
    print(f"Target: {Y.shape[1]}")

    return df_model, X, Y


def create_bootstrap_dataframe(df, n_samples, seed=GLOBAL_SEED):
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(df), size=n_samples, replace=True)
    return df.iloc[indices].copy().reset_index(drop=True)


def generate_contingency_bootstraps(file_path=ANN_EXCEDANCE_DATA, sample_sizes=BOOTSTRAP_SIZES, output_dir=BOOTSTRAP_DIR, seed=GLOBAL_SEED):
    bootstrap_results = {}
    df = pd.read_excel(file_path, engine="openpyxl")

    print(f"Original exceedance dataset: {len(df):,} rows")

    for j, n_samples in enumerate(sample_sizes):
        df_boot = create_bootstrap_dataframe(df, n_samples, seed=seed + j)
        file_name = f"Data_excedence_bootstrap_{n_samples}.xlsx"
        file_path = output_dir / file_name
        df_boot.to_excel(file_path, index=False, engine="openpyxl")
        bootstrap_results[n_samples] = df_boot
        print(f"Bootstrap {n_samples:>5,}: {len(df_boot):,} rows -> {file_path}")

    return bootstrap_results


def train_ann_dataframe(
    df,
    hidden_neurons,
    model_save_path,
    model_name="ANN",
    seed=GLOBAL_SEED,
    train_ratio=0.70,
    val_ratio=0.15
):
    set_seed(seed)

    data = df[ANN_FEATURES + [ANN_TARGET]].copy().dropna().reset_index(drop=True)

    for col in ANN_FEATURES + [ANN_TARGET]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna().reset_index(drop=True)

    X_np = data[ANN_FEATURES].values.astype(np.float32)
    T_np = data[ANN_TARGET].values.astype(np.float32).reshape(-1, 1)

    n_samples = len(X_np)
    input_size = X_np.shape[1]
    output_size = 1

    print(f"\nTRAINING: {model_name}")
    print(f"Samples: {n_samples:,}")
    print(f"Inputs: {input_size}")
    print(f"Hidden neurons: {hidden_neurons}")

    X_tensor = torch.from_numpy(X_np).float()
    T_tensor = torch.from_numpy(T_np).float()
    dataset = TensorDataset(X_tensor, T_tensor)

    net = RedNeuronalSimple(input_size, hidden_neurons, output_size).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=ANN_LEARNING_RATE)

    train_size = int(train_ratio * n_samples)
    val_size = int(val_ratio * n_samples)
    test_size = n_samples - train_size - val_size

    indices = np.random.permutation(n_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=ANN_BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=ANN_BATCH_SIZE, sampler=val_sampler)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    start_time = time.time()

    for epoch in range(ANN_MAX_EPOCHS):
        net.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / train_size

        net.eval()
        val_running_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                val_running_loss += loss.item() * inputs.size(0)

        val_epoch_loss = val_running_loss / val_size

        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch {epoch + 1:5d}/{ANN_MAX_EPOCHS} | "
                f"Train MSE: {epoch_loss:.6f} | "
                f"Val MSE: {val_epoch_loss:.6f}"
            )

        if val_epoch_loss < best_val_loss - ANN_MIN_DELTA:
            best_val_loss = val_epoch_loss
            epochs_no_improve = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in net.state_dict().items()
            }
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= ANN_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    training_time = time.time() - start_time

    if "best_state" in locals():
        net.load_state_dict(best_state)

    net.eval()

    with torch.no_grad():
        X_eval = X_tensor.to(DEVICE)
        Y_pred = net(X_eval).cpu().numpy()

    Y_pred = np.maximum(Y_pred, 0)
    Y_true = T_np

    R, _ = pearsonr(Y_pred.flatten(), Y_true.flatten())
    RMSE = np.sqrt(np.mean((Y_pred - Y_true) ** 2))
    MAE = np.mean(np.abs(Y_pred - Y_true))
    R2 = 1 - np.sum((Y_true - Y_pred) ** 2) / np.sum((Y_true - Y_true.mean()) ** 2)

    net_cpu = net.to("cpu")
    torch.save(net_cpu.state_dict(), model_save_path)

    print(f"Training completed: {training_time:.2f} s")
    print(f"R = {R:.4f}")
    print(f"RMSE = {RMSE:.4f}")
    print(f"MAE = {MAE:.4f}")
    print(f"R2 = {R2:.4f}")
    print(f"Model saved: {model_save_path}")

    return {
        "model": net_cpu,
        "model_name": model_name,
        "n_samples": n_samples,
        "hidden_neurons": hidden_neurons,
        "best_val_mse": best_val_loss,
        "R": R,
        "RMSE": RMSE,
        "MAE": MAE,
        "R2": R2,
        "training_time_s": training_time,
        "model_path": model_save_path
    }


def main():
    set_seed()

    df_exceedance, X_exceedance, Y_exceedance = load_ann_data(ANN_EXCEDANCE_DATA)
    original_result = train_ann_dataframe(
        df=df_exceedance,
        hidden_neurons=ANN_SPECIAL_HIDDEN,
        model_save_path=MODEL_DIR / "NN_especial_excedance.pth",
        model_name="C-ANN - original exceedances",
        seed=GLOBAL_SEED
    )

    bootstrap_datasets = generate_contingency_bootstraps()

    for j, n_bootstrap in enumerate(BOOTSTRAP_SIZES):
        print(f"\nBOOTSTRAP {n_bootstrap:,}")
        df_bootstrap = bootstrap_datasets[n_bootstrap]

        bootstrap_result = train_ann_dataframe(
            df=df_bootstrap,
            hidden_neurons=ANN_SPECIAL_HIDDEN,
            model_save_path=MODEL_DIR / f"NN_especial_bootstrap_{n_bootstrap}.pth",
            model_name=f"C-ANN - bootstrap {n_bootstrap}",
            seed=GLOBAL_SEED + j + 1
        )

        bootstrap_result["bootstrap_size"] = n_bootstrap

    print(f"Bootstrap datasets saved to: {BOOTSTRAP_DIR}")
    print(f"Models saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
