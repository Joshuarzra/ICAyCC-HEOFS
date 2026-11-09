import os
import sys
from pathlib import Path
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import pearsonr

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from red_neuronal import RedNeuronalSimple
TRAIN_VAL_FILE = PROJECT_DIR / "Data_2016_2023.xlsx"
TEST_FILE = PROJECT_DIR / "Data_2024.xlsx"
OUTPUT_DIR = PROJECT_DIR / "C-ANN results"
FIGURES_DIR = OUTPUT_DIR / "Figures"
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
HIDDEN_NEURONS = 60 # Set for testing purposes; can be adjusted for final training
LEARNING_RATE = 0.001
MAX_EPOCHS = 10000
BATCH_SIZE = 64
PATIENCE = 500
MIN_DELTA = 0.0001
GLOBAL_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


import matplotlib.pyplot as plt

PLOT_ORDER = ["Train", "Validation", "Test", "Overall"]
PLOT_COLORS = {
    "Train": "#0072BD",
    "Validation": "#008000",
    "Test": "#D62728",
    "Overall": "#444444"
}


def generate_regression_plot(results, model_label, figures_dir):
    plot_data = {}

    for partition in PLOT_ORDER:
        if partition not in results:
            continue

        target = np.asarray(results[partition]["target"], dtype=float).ravel()
        output = np.asarray(results[partition]["output"], dtype=float).ravel()

        valid = np.isfinite(target) & np.isfinite(output)
        target = target[valid]
        output = output[valid]

        if len(target):
            plot_data[partition] = {"target": target, "output": output}

    maximum = max(
        max(np.max(data["target"]), np.max(data["output"]))
        for data in plot_data.values()
    )
    minimum = min(
        min(np.min(data["target"]), np.min(data["output"]))
        for data in plot_data.values()
    )

    value_range = maximum - minimum
    if value_range <= 0:
        value_range = max(abs(maximum), 1.0)

    margin = 0.035 * value_range
    lower_limit = min(0.0, minimum) - margin
    upper_limit = maximum + margin

    fig, axes = plt.subplots(1, 4, figsize=(24, 7), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    for ax, partition in zip(axes, PLOT_ORDER):
        if partition not in plot_data:
            ax.set_visible(False)
            continue

        target = plot_data[partition]["target"]
        output = plot_data[partition]["output"]

        ax.scatter(
            target,
            output,
            s=30,
            marker="o",
            facecolors="none",
            edgecolors=(0, 0, 0, 0.50),
            linewidths=0.70,
            label="Data",
            zorder=2,
            rasterized=True
        )

        if len(target) >= 2 and np.ptp(target) > 0:
            slope, intercept = np.polyfit(target, output, 1)
            x_fit = np.linspace(lower_limit, upper_limit, 300)
            y_fit = slope * x_fit + intercept
            ax.plot(
                x_fit,
                y_fit,
                color=PLOT_COLORS[partition],
                linewidth=4.0,
                linestyle="-",
                label="Fit",
                zorder=5
            )

        ax.plot(
            [lower_limit, upper_limit],
            [lower_limit, upper_limit],
            color="black",
            linestyle="--",
            linewidth=3.5,
            label=r"$Y=T$",
            zorder=6
        )

        if len(target) >= 2 and np.std(target) > 0 and np.std(output) > 0:
            correlation = np.corrcoef(target, output)[0, 1]
            correlation_text = f"R = {correlation:.2f}"
        else:
            correlation_text = "R = N/A"

        ax.set_title(
            f"{partition}\n{correlation_text}",
            fontsize=25,
            fontweight="bold",
            pad=12
        )
        ax.set_xlabel("Target (ppb)", fontsize=24, fontweight="bold", labelpad=9)

        if partition == "Train":
            ax.set_ylabel("Output (ppb)", fontsize=24, fontweight="bold", labelpad=10)

        ax.set_xlim(lower_limit, upper_limit)
        ax.set_ylim(lower_limit, upper_limit)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=15,
            direction="out",
            length=5,
            width=1.3
        )

        for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
            tick_label.set_fontweight("bold")

        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

        legend = ax.legend(
            loc="lower right",
            fontsize=17,
            frameon=True,
            fancybox=False,
            framealpha=0.92,
            facecolor="#D6D5D5",
            edgecolor="#555555",
            borderpad=0.7,
            labelspacing=0.55,
            handlelength=2.2,
            handletextpad=0.7,
            borderaxespad=0.6
        )
        legend.set_zorder(10)
        legend.get_frame().set_linewidth(1.2)

        for legend_text in legend.get_texts():
            legend_text.set_fontweight("bold")

    title = f"{model_label}: regression plots"
    fig.suptitle(title, fontsize=25, fontweight="bold", y=1.04)
    fig.tight_layout(rect=[0, 0, 1, 0.94], w_pad=1.5)

    file_path = figures_dir / f"{model_label.lower().replace('-', '_')}_regression_plots.png"
    fig.savefig(file_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Figure saved: {file_path}")


def generate_regression_figures(results, model_label, figures_dir):
    generate_regression_plot(results, model_label, figures_dir)



def set_seed(seed=GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data():
    df = pd.read_excel(TRAIN_VAL_FILE, engine="openpyxl")
    df["datetime"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    df = df[(df["datetime"].dt.year >= 2016) & (df["datetime"].dt.year <= 2023)].copy()
    df = df[["datetime"] + ANN_FEATURES + [ANN_TARGET]].dropna().reset_index(drop=True)

    df_train = df[df["datetime"].dt.year <= 2022].copy()
    df_val = df[df["datetime"].dt.year == 2023].copy()

    df_test = pd.read_excel(TEST_FILE, engine="openpyxl")
    df_test["datetime"] = pd.to_datetime(df_test["date"], errors="coerce")
    df_test = df_test.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    df_test = df_test[df_test["datetime"].dt.year == 2024].copy()
    df_test = df_test[["datetime"] + ANN_FEATURES + [ANN_TARGET]].dropna().reset_index(drop=True)

    X_train = df_train[ANN_FEATURES].values.astype(np.float32)
    y_train = df_train[ANN_TARGET].values.astype(np.float32).reshape(-1, 1)
    X_val = df_val[ANN_FEATURES].values.astype(np.float32)
    y_val = df_val[ANN_TARGET].values.astype(np.float32).reshape(-1, 1)
    X_test = df_test[ANN_FEATURES].values.astype(np.float32)
    y_test = df_test[ANN_TARGET].values.astype(np.float32).reshape(-1, 1)

    print(f"TRAIN 2016-2022: {len(df_train):,} observations")
    print(f"VALIDATION 2023: {len(df_val):,} observations")
    print(f"TEST 2024: {len(df_test):,} observations")

    return X_train, y_train, X_val, y_val, X_test, y_test


def create_loader(X, y, shuffle):
    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def calculate_metrics(y_real, y_pred):
    y_real = np.asarray(y_real).flatten()
    y_pred = np.asarray(y_pred).flatten()
    r, _ = pearsonr(y_pred, y_real)
    rmse = np.sqrt(np.mean((y_pred - y_real) ** 2))
    mae = np.mean(np.abs(y_pred - y_real))
    bias = np.mean(y_pred - y_real)
    r2 = 1 - np.sum((y_real - y_pred) ** 2) / np.sum((y_real - y_real.mean()) ** 2)
    return {"R": r, "RMSE": rmse, "MAE": mae, "Bias": bias, "R2": r2}


def predict(model, X):
    model.eval()
    with torch.no_grad():
        prediction = model(torch.from_numpy(X).float().to(DEVICE)).cpu().numpy().reshape(-1)
    return np.maximum(prediction, 0)


def train_normal_ann():
    set_seed()

    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    train_loader = create_loader(X_train, y_train, True)
    val_loader = create_loader(X_val, y_val, False)

    net = RedNeuronalSimple(len(ANN_FEATURES), HIDDEN_NEURONS, 1).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    best_val_loss = np.inf
    best_state = None
    epochs_without_improvement = 0
    start_time = time.time()

    print(f"Device: {DEVICE}")
    print(f"Hidden neurons: {HIDDEN_NEURONS}")

    for epoch in range(MAX_EPOCHS):
        net.train()
        train_loss_sum = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            optimizer.zero_grad()
            y_pred = net(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * X_batch.size(0)

        train_loss = train_loss_sum / len(train_loader.dataset)

        net.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                y_pred = net(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss_sum += loss.item() * X_batch.size(0)

        val_loss = val_loss_sum / len(val_loader.dataset)

        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch [{epoch + 1:5d}/{MAX_EPOCHS}] | "
                f"Train MSE: {train_loss:.6f} | "
                f"Val MSE: {val_loss:.6f}"
            )

        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in net.state_dict().items()
            }
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    training_time = time.time() - start_time
    net.load_state_dict(best_state)

    pred_train = np.maximum(net(torch.from_numpy(X_train).float().to(DEVICE)).detach().cpu().numpy().reshape(-1), 0)
    pred_val = predict(net, X_val)
    pred_test = predict(net, X_test)

    metrics = {
        "Train_2016_2022": calculate_metrics(y_train, pred_train),
        "Validation_2023": calculate_metrics(y_val, pred_val),
        "Test_2024": calculate_metrics(y_test, pred_test)
    }

    model_path = MODEL_DIR / f"NN_normal_2016_2022_{HIDDEN_NEURONS}.pth"
    torch.save(net.cpu().state_dict(), model_path)

    results = {
        "Train": {"target": y_train.flatten(), "output": pred_train},
        "Validation": {"target": y_val.flatten(), "output": pred_val},
        "Test": {"target": y_test.flatten(), "output": pred_test}
    }
    results["Overall"] = {
        "target": np.concatenate([results[key]["target"] for key in ["Train", "Validation", "Test"]]),
        "output": np.concatenate([results[key]["output"] for key in ["Train", "Validation", "Test"]])
    }
    generate_regression_figures(results, "ANN_normal", FIGURES_DIR)

    print(f"Training time: {training_time:.2f} s")
    print(f"Best validation MSE: {best_val_loss:.6f}")
    print(f"Model saved: {model_path}")

    for partition, values in metrics.items():
        print(
            f"{partition}: R={values['R']:.4f}, "
            f"RMSE={values['RMSE']:.4f}, "
            f"MAE={values['MAE']:.4f}, "
            f"Bias={values['Bias']:.4f}, "
            f"R2={values['R2']:.4f}"
        )


if __name__ == "__main__":
    train_normal_ann()
