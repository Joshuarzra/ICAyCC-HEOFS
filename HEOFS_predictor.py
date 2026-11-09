from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from red_neuronal import RedNeuronalSimple

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"

INPUT_FILE = PROJECT_DIR / "Data_2024.xlsx"
TRAINING_DATA_FILE = PROJECT_DIR / "Data_2016_2023.xlsx"
OUTPUT_FILE = PROJECT_DIR / "HEOFS_predictions.xlsx"

ENCODER_LENGTH = 500
PREDICTION_HORIZON = 72
ANN_NORMAL_HIDDEN_NEURONS = 60
ANN_SPECIAL_HIDDEN_NEURONS = 60


ANN_NORMAL_MODEL = f"NN_normal_2016_2022_{ANN_NORMAL_HIDDEN_NEURONS}.pth"
ANN_SPECIAL_MODEL = f"NN_especial_bootstrap_2000_{ANN_SPECIAL_HIDDEN_NEURONS}.pth"

TFT_MODELS = {
    "O3": {
        "directory": "O3_72h_test",
        "checkpoint": None,
        "dataset": "tsd_O3_72h_test.pth",
    },
    "NO2": {
        "directory": "NO2_72h_test",
        "checkpoint": None,
        "dataset": "tsd_NO2_72h_test.pth",
    },
    "SO2": {
        "directory": "SO2_72h_test",
        "checkpoint": None,
        "dataset": "tsd_SO2_72h_test.pth",
    },
    "CO": {
        "directory": "CO_72h_test",
        "checkpoint": None,
        "dataset": "tsd_CO_72h_test.pth",
    },
    "NOX": {
        "directory": "NOX_72h_test",
        "checkpoint": None,
        "dataset": "tsd_NOX_72h_test.pth",
    },
    "RH": {
        "directory": "RH_72h_test",
        "checkpoint": None,
        "dataset": "tsd_RH_72h_test.pth",
    },
    "TMP": {
        "directory": "TMP_72h_test",
        "checkpoint": None,
        "dataset": "tsd_TMP_72h_test.pth",
    },
    "Global": {
        "directory": "Global_72h_test",
        "checkpoint": None,
        "dataset": "tsd_Global_72h_test.pth",
    },
    "WSP": {
        "directory": "WSP_72h_test",
        "checkpoint": None,
        "dataset": "tsd_WSP_72h_test.pth",
    },
}

PRECURSORS = ["NO2", "SO2", "CO", "NOX", "RH", "TMP", "Global", "WSP"]
TFT_VARIABLES = ["O3"] + PRECURSORS
ANN_FEATURES = ["month", "hour"] + PRECURSORS
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENFORCE_GLOBAL_NIGHT_ZERO = True


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_DIR / path


def find_checkpoint(model_config):
    directory = resolve_path(MODEL_DIR / model_config["directory"])
    selected = model_config["checkpoint"]

    if selected:
        selected_path = Path(selected)
        if not selected_path.is_absolute():
            selected_path = directory / selected_path
        if selected_path.is_file():
            return selected_path
        raise FileNotFoundError(f"No se encontró el checkpoint: {selected_path}")

    checkpoints = list(directory.glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No hay archivos .ckpt en {directory}")

    def validation_loss(path):
        try:
            return float(path.stem.split("val_loss=")[-1])
        except ValueError:
            return float("inf")

    checkpoints.sort(key=validation_loss)
    if not np.isfinite(validation_loss(checkpoints[0])):
        if len(checkpoints) == 1:
            return checkpoints[0]
        raise ValueError(
            f"No se pudo identificar el mejor checkpoint en {directory}. "
            "Especifica su nombre en TFT_MODELS."
        )
    return checkpoints[0]


def load_normalization_statistics():
    frame = pd.read_excel(resolve_path(TRAINING_DATA_FILE), engine="openpyxl")
    if "year" not in frame.columns:
        frame["year"] = pd.to_datetime(frame["date"], errors="coerce").dt.year

    frame = frame.loc[frame["year"].between(2016, 2022)].copy()
    statistics = {}

    for variable in TFT_VARIABLES:
        values = pd.to_numeric(frame[variable], errors="coerce").dropna()
        statistics[variable] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
        }

    return statistics


def load_history():
    frame = pd.read_excel(resolve_path(INPUT_FILE), engine="openpyxl")

    if "datetime" in frame.columns:
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    elif "date" in frame.columns:
        frame["datetime"] = pd.to_datetime(frame["date"], errors="coerce")
    else:
        raise ValueError("El archivo de entrada debe contener 'datetime' o 'date'.")

    frame = frame.dropna(subset=["datetime"]).sort_values("datetime")
    frame = frame.drop_duplicates("datetime", keep="last").reset_index(drop=True)

    required = TFT_VARIABLES
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el archivo de entrada: {missing}")

    frame = frame.tail(ENCODER_LENGTH).copy().reset_index(drop=True)
    if len(frame) != ENCODER_LENGTH:
        raise ValueError(
            f"Se requieren exactamente {ENCODER_LENGTH} horas históricas; "
            f"se encontraron {len(frame)}."
        )

    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if frame[required].isna().any().any():
        raise ValueError("Hay valores faltantes o no numéricos en las variables atmosféricas.")

    differences = frame["datetime"].diff().dropna()
    if not (differences == pd.Timedelta(hours=1)).all():
        raise ValueError("Las 500 observaciones deben ser consecutivas, con frecuencia horaria.")

    return frame


def load_tft_models():
    models = {}
    datasets = {}
    parameters = {}

    for variable in TFT_VARIABLES:
        config = TFT_MODELS[variable]
        model_dir = resolve_path(MODEL_DIR / config["directory"])
        dataset_path = resolve_path(model_dir / config["dataset"])
        checkpoint_path = find_checkpoint(config)
        
        training_dataset = torch.load(
            dataset_path,
            map_location="cpu",
            weights_only=False
        )
        dataset_parameters = training_dataset.get_parameters()

        model = TemporalFusionTransformer.load_from_checkpoint(
            str(checkpoint_path),
            map_location=DEVICE,
        )
        model.to(DEVICE)
        model.eval()

        models[variable] = model
        datasets[variable] = training_dataset
        parameters[variable] = dataset_parameters

    return models, datasets, parameters


def cast_calendar_columns(frame, dataset_parameters):
    encoders = dataset_parameters.get("categorical_encoders", {})
    for column in ("hour", "day", "month"):
        encoder = encoders.get(column)
        classes = getattr(encoder, "classes_", None)
        if isinstance(classes, dict) and classes:
            sample = next(iter(classes.keys()))
            if isinstance(sample, str):
                frame[column] = frame[column].astype(str)
            elif isinstance(sample, (int, np.integer)):
                frame[column] = frame[column].astype(int)
            elif isinstance(sample, (float, np.floating)):
                frame[column] = frame[column].astype(float)
    return frame


def predict_tft(variable, history, future_dates, models, parameters, statistics, future_precursors=None, series_id="ozono"):
    model = models[variable]
    dataset_parameters = parameters[variable]

    if variable != "O3":
        series_id = variable

    historical = history.copy()
    historical["serie"] = series_id
    historical["time_idx"] = np.arange(len(historical), dtype=np.int64)
    historical["year"] = historical["datetime"].dt.year
    historical["month"] = historical["datetime"].dt.month
    historical["day"] = historical["datetime"].dt.day
    historical["hour"] = historical["datetime"].dt.hour

    unknown_reals = dataset_parameters.get("time_varying_unknown_reals", [])
    for column in unknown_reals:
        if column in statistics:
            historical[column] = (
                historical[column] - statistics[column]["mean"]
            ) / statistics[column]["std"]

    future = pd.concat(
        [historical.iloc[[-1]].copy()] * len(future_dates),
        ignore_index=True,
    )
    future["datetime"] = future_dates
    future["year"] = future_dates.year
    future["month"] = future_dates.month
    future["day"] = future_dates.day
    future["hour"] = future_dates.hour
    future["time_idx"] = np.arange(
        len(historical), len(historical) + len(future_dates), dtype=np.int64
    )
    future["serie"] = series_id

    if variable == "O3":
        for precursor in PRECURSORS:
            future[precursor] = (
                np.asarray(future_precursors[precursor], dtype=float)
                - statistics[precursor]["mean"]
            ) / statistics[precursor]["std"]
        future["O3"] = historical["O3"].iloc[-1]
    else:
        future[variable] = historical[variable].iloc[-1]

    historical = cast_calendar_columns(historical, dataset_parameters)
    future = cast_calendar_columns(future, dataset_parameters)
    combined = pd.concat([historical, future], ignore_index=True)

    prediction_dataset = TimeSeriesDataSet.from_parameters(
        dataset_parameters,
        combined,
        predict=True,
        stop_randomization=True,
    )
    loader = prediction_dataset.to_dataloader(
        train=False,
        batch_size=1,
        num_workers=0,
    )

    x, _ = next(iter(loader))
    x = {
        key: value.to(DEVICE) if torch.is_tensor(value) else value
        for key, value in x.items()
    }
    with torch.inference_mode():
        output = model(x)

    prediction = output["prediction"].detach().cpu().numpy()
    if prediction.ndim == 3:
        quantiles = getattr(model.loss, "quantiles", None)
        median_index = 3 if quantiles is None else int(
            np.argmin(np.abs(np.asarray(quantiles) - 0.5))
        )
        prediction = prediction[0, :, median_index]
    else:
        prediction = prediction.reshape(-1)

    prediction = prediction[:len(future_dates)]
    prediction = prediction * statistics[variable]["std"] + statistics[variable]["mean"]
    prediction = np.maximum(prediction, 0.0)

    if variable == "Global" and ENFORCE_GLOBAL_NIGHT_ZERO:
        prediction[(future_dates.hour < 6) | (future_dates.hour > 20)] = 0.0

    return prediction


def load_ann(path, hidden_neurons):
    model = RedNeuronalSimple(len(ANN_FEATURES), hidden_neurons, 1)
    state = torch.load(resolve_path(MODEL_DIR / path), map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def main():
    statistics = load_normalization_statistics()
    history = load_history()
    future_dates = pd.date_range(
        start=history["datetime"].iloc[-1] + pd.Timedelta(hours=1),
        periods=PREDICTION_HORIZON,
        freq="h",
    )

    models, _, parameters = load_tft_models()
    forecasts = pd.DataFrame({
        "datetime": future_dates,
        "year": future_dates.year,
        "month": future_dates.month,
        "day": future_dates.day,
        "hour": future_dates.hour,
    })

    for variable in PRECURSORS:
        forecasts[variable] = predict_tft(
            variable, history, future_dates, models, parameters, statistics
        )

    forecasts["O3_TFT"] = predict_tft(
        "O3",
        history,
        future_dates,
        models,
        parameters,
        statistics,
        future_precursors=forecasts,
    )

    ann_normal = load_ann(ANN_NORMAL_MODEL, ANN_NORMAL_HIDDEN_NEURONS)
    ann_special = load_ann(ANN_SPECIAL_MODEL, ANN_SPECIAL_HIDDEN_NEURONS)

    # Los ANN solo se evalúan entre las 13:00 y las 19:00, inclusive.
    # Fuera de esos horarios, cada variante HEOFS conserva el pronóstico TFT de O3.
    ann_hours_mask = forecasts["hour"].between(13, 19, inclusive="both").to_numpy()
    heofs_c_ann = forecasts["O3_TFT"].to_numpy(dtype=float).copy()
    heofs_n_ann = forecasts["O3_TFT"].to_numpy(dtype=float).copy()

    if ann_hours_mask.any():
        X = forecasts.loc[ann_hours_mask, ANN_FEATURES].to_numpy(dtype=np.float32)
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)

        with torch.inference_mode():
            normal_prediction = ann_normal(X_tensor).squeeze(-1).cpu().numpy()
            special_prediction = ann_special(X_tensor).squeeze(-1).cpu().numpy()

        heofs_n_ann[ann_hours_mask] = np.maximum(normal_prediction, 0.0)
        heofs_c_ann[ann_hours_mask] = np.maximum(special_prediction, 0.0)

    forecasts["HEOFS - C_ANN"] = heofs_c_ann
    forecasts["HEOFS - N-ANN"] = heofs_n_ann

    prediction_sheet = forecasts[
        ["datetime", "O3_TFT", "HEOFS - C_ANN", "HEOFS - N-ANN"]
    ].copy()

    historical_sheet = history.copy()
    historical_sheet["TFT_O3"] = np.nan
    historical_sheet["HEOFS - C_ANN"] = np.nan
    historical_sheet["HEOFS - N-ANN"] = np.nan
    future_sheet = forecasts.copy()
    future_sheet["O3"] = np.nan
    future_sheet = future_sheet.rename(columns={"O3_TFT": "TFT_O3"})

    common_columns = [
        "datetime", "O3", "TFT_O3", "HEOFS - C_ANN", "HEOFS - N-ANN",
        "NO2", "SO2", "CO", "NOX", "RH", "TMP", "Global", "WSP",
    ]

    historical_output = historical_sheet.reindex(columns=common_columns)
    future_output = future_sheet.reindex(columns=common_columns)
    continuous_sheet = pd.concat(
        [historical_output, future_output],
        ignore_index=True,
    )

    output_path = resolve_path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        prediction_sheet.to_excel(writer, sheet_name="Predicciones", index=False)
        continuous_sheet.to_excel(writer, sheet_name="Ventana_historica", index=False)

    print(f"Predicción completada: {output_path}")
    print(f"Dispositivo: {DEVICE}")
    print(f"Horizonte: {PREDICTION_HORIZON} horas")


if __name__ == "__main__":
    main()
