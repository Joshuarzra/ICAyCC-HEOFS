from pathlib import Path
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from torch.utils.data import TensorDataset, DataLoader
from red_neuronal import RedNeuronalSimple


PROJECT_DIR = Path(__file__).resolve().parent
EXCEEDANCE_FILE = PROJECT_DIR / "Data_excedence_2016_2023.xlsx"
TRAIN_VALIDATION_FILE = PROJECT_DIR / "Data_2016_2023.xlsx"
TEST_FILE = PROJECT_DIR / "Data_2024.xlsx"
OUTPUT_DIR = PROJECT_DIR / "ANN_BOOTSTRAP_results"
BOOTSTRAP_DIR = OUTPUT_DIR / "bootstrap_datasets"
MODEL_DIR = OUTPUT_DIR / "models"
FIGURE_DIR = OUTPUT_DIR / "figures"

FEATURES = [
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

TARGET = "O3"
CONTINGENCY_THRESHOLD = 155
HIDDEN_NEURONS = 60
BOOTSTRAP_SIZES = [500, 1000, 1500, 2000, 3000, 5000]
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MAX_EPOCHS = 10000
PATIENCE = 500
MIN_DELTA = 0.0001
GLOBAL_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BOOTSTRAP_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def fijar_semilla(seed=GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cargar_particiones():
    df_exceedance = pd.read_excel(EXCEEDANCE_FILE, engine="openpyxl")
    df_exceedance["year"] = pd.to_numeric(df_exceedance["year"], errors="coerce")
    df_exceedance = df_exceedance[FEATURES + [TARGET, "year"]].dropna().reset_index(drop=True)

    df_train_pool = df_exceedance[df_exceedance["year"] <= 2022].copy()
    df_validation = pd.read_excel(TRAIN_VALIDATION_FILE, engine="openpyxl")
    df_validation["datetime"] = pd.to_datetime(df_validation["date"], errors="coerce")
    df_validation = df_validation.dropna(subset=["datetime"]).copy()
    df_val = df_validation[df_validation["datetime"].dt.year == 2023].copy()
    df_val = df_val[FEATURES + [TARGET]].dropna().reset_index(drop=True)

    df_test = pd.read_excel(TEST_FILE, engine="openpyxl")
    df_test["datetime"] = pd.to_datetime(df_test["date"], errors="coerce")
    df_test = df_test.dropna(subset=["datetime"]).copy()
    df_test = df_test[df_test["datetime"].dt.year == 2024].copy()
    df_test = df_test[FEATURES + [TARGET]].dropna().reset_index(drop=True)

    X_val = df_val[FEATURES].to_numpy(dtype=np.float32)
    y_val = df_val[TARGET].to_numpy(dtype=np.float32).reshape(-1, 1)
    X_test = df_test[FEATURES].to_numpy(dtype=np.float32)
    y_test = df_test[TARGET].to_numpy(dtype=np.float32).reshape(-1, 1)

    print(f"Pool de excedencias para entrenamiento 2016-2022: {len(df_train_pool):,}")
    print(f"Validation 2023: {len(df_val):,}")
    print(f"Test 2024: {len(df_test):,}")

    return df_train_pool, X_val, y_val, X_test, y_test


def calcular_metricas(y_real, y_pred):
    y_real = np.asarray(y_real).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    r, _ = pearsonr(y_real, y_pred)
    rmse = np.sqrt(np.mean((y_pred - y_real) ** 2))
    mae = np.mean(np.abs(y_pred - y_real))
    bias = np.mean(y_pred - y_real)
    r2 = 1 - np.sum((y_real - y_pred) ** 2) / np.sum((y_real - np.mean(y_real)) ** 2)
    return {"R": r, "RMSE": rmse, "MAE": mae, "Bias": bias, "R2": r2}


def crear_bootstraps(df_train_pool):
    datasets = {}

    for index, size in enumerate(BOOTSTRAP_SIZES):
        rng = np.random.default_rng(GLOBAL_SEED + index)
        indices = rng.choice(len(df_train_pool), size=size, replace=True)
        bootstrap_df = df_train_pool.iloc[indices].copy().reset_index(drop=True)
        bootstrap_path = BOOTSTRAP_DIR / f"Data_excedence_bootstrap_{size}.xlsx"
        bootstrap_df.to_excel(bootstrap_path, index=False, engine="openpyxl")
        datasets[size] = bootstrap_df
        print(f"Bootstrap {size}: {bootstrap_path}")

    return datasets


def crear_loader(X, y, shuffle):
    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def predecir(model, X):
    model.eval()
    with torch.no_grad():
        prediction = model(torch.from_numpy(X).float().to(DEVICE)).cpu().numpy().reshape(-1)
    return np.maximum(prediction, 0.0)


def graficar_regresion(particiones, output_path, title):
    fig, axes = plt.subplots(1, len(particiones), figsize=(5 * len(particiones), 4.8))
    axes = np.atleast_1d(axes)
    values = []

    for target, output in particiones.values():
        valid = np.isfinite(target) & np.isfinite(output) & (target >= 0) & (output >= 0)
        values.extend(target[valid].tolist())
        values.extend(output[valid].tolist())

    upper = max(values) * 1.03 if values else 1.0
    upper = max(upper, 1.0)

    for ax, (name, (target, output)) in zip(axes, particiones.items()):
        valid = np.isfinite(target) & np.isfinite(output) & (target >= 0) & (output >= 0)
        target = target[valid]
        output = output[valid]
        r, _ = pearsonr(target, output)
        ax.scatter(target, output, s=10, alpha=0.45)
        ax.plot([0, upper], [0, upper], linestyle="--", linewidth=1.5, label="Y = T")
        ax.set_title(f"{name} (R = {r:.3f})")
        ax.set_xlabel("Target")
        ax.set_ylabel("Output")
        ax.set_xlim(0, upper)
        ax.set_ylim(0, upper)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def entrenar_bootstrap(df_bootstrap, X_val, y_val, X_test, y_test, size, seed):
    fijar_semilla(seed)

    X_train = df_bootstrap[FEATURES].to_numpy(dtype=np.float32)
    y_train = df_bootstrap[TARGET].to_numpy(dtype=np.float32).reshape(-1, 1)

    train_loader = crear_loader(X_train, y_train, True)
    val_loader = crear_loader(X_val, y_val, False)

    model = RedNeuronalSimple(len(FEATURES), HIDDEN_NEURONS, 1).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = np.inf
    best_state = None
    epochs_without_improvement = 0
    start_time = time.time()

    print(f"\nANN especial bootstrap {size}")
    print(f"Dispositivo: {DEVICE}")
    print(f"Train bootstrap: {len(X_train):,}")
    print(f"Validation 2023: {len(X_val):,}")
    print(f"Test 2024: {len(X_test):,}")
    print(f"Neuronas ocultas: {HIDDEN_NEURONS}")

    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss_sum = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            optimizer.zero_grad()
            prediction = model(X_batch)
            loss = criterion(prediction, y_batch)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * X_batch.size(0)

        train_loss = train_loss_sum / len(X_train)

        model.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                prediction = model(X_batch)
                loss = criterion(prediction, y_batch)
                val_loss_sum += loss.item() * X_batch.size(0)

        val_loss = val_loss_sum / len(X_val)

        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch [{epoch + 1}/{MAX_EPOCHS}] | "
                f"Train MSE: {train_loss:.6f} | "
                f"Val MSE: {val_loss:.6f}"
            )

        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping en época {epoch + 1}")
                break

    training_time = time.time() - start_time
    model.load_state_dict(best_state)

    pred_train = predecir(model, X_train)
    pred_val = predecir(model, X_val)
    pred_test = predecir(model, X_test)

    metrics = {
        "Train_bootstrap": calcular_metricas(y_train, pred_train),
        "Validation_2023": calcular_metricas(y_val, pred_val),
        "Test_2024": calcular_metricas(y_test, pred_test)
    }

    model_path = MODEL_DIR / f"NN_especial_bootstrap_{size}.pth"
    torch.save(model.cpu().state_dict(), model_path)

    regression_data = {
        "Train": (y_train.reshape(-1), pred_train),
        "Validation": (y_val.reshape(-1), pred_val),
        "Test": (y_test.reshape(-1), pred_test),
        "Overall": (
            np.concatenate([y_train.reshape(-1), y_val.reshape(-1), y_test.reshape(-1)]),
            np.concatenate([pred_train, pred_val, pred_test])
        )
    }

    graficar_regresion(
        regression_data,
        FIGURE_DIR / f"NN_especial_bootstrap_{size}_regression_plots.png",
        f"ANN bootstrap {size}: observed versus predicted O$_3$"
    )

    rows = []
    for split, values in metrics.items():
        rows.append({
            "bootstrap_size": size,
            "Partition": split,
            **values
        })

    print(f"Tiempo de entrenamiento: {training_time:.2f} s")
    print(f"Mejor validation MSE: {best_val_loss:.6f}")
    print(f"Modelo guardado: {model_path}")

    for split, values in metrics.items():
        print(
            f"{split}: R={values['R']:.4f}, "
            f"RMSE={values['RMSE']:.4f}, "
            f"MAE={values['MAE']:.4f}, "
            f"Bias={values['Bias']:.4f}, "
            f"R2={values['R2']:.4f}"
        )

    return rows


def main():
    df_train_pool, X_val, y_val, X_test, y_test = cargar_particiones()
    bootstrap_datasets = crear_bootstraps(df_train_pool)

    all_metrics = []

    for index, size in enumerate(BOOTSTRAP_SIZES):
        rows = entrenar_bootstrap(
            bootstrap_datasets[size],
            X_val,
            y_val,
            X_test,
            y_test,
            size,
            GLOBAL_SEED + index + 1
        )
        all_metrics.extend(rows)

    comparison = pd.DataFrame(all_metrics)
    comparison.to_csv(OUTPUT_DIR / "ANN_bootstrap_metrics.csv", index=False)
    print(f"\nResultados guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
