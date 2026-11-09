# HEOFS: Hybrid Extreme Ozone Forecasting System

**HEOFS** is a hybrid deep learning framework designed to forecast hourly tropospheric ozone (O₃) concentrations and anticipate extreme ozone episodes associated with air quality contingencies in Mexico City.

The system combines Temporal Fusion Transformers (TFTs) and Artificial Neural Networks (ANNs) to generate ozone forecasts up to 72 hours ahead.

## Architecture

HEOFS integrates the following components:

- **TFT models:** One model forecasts O₃, while eight additional models forecast atmospheric precursor and meteorological variables.
- **N-ANN:** Conventional Artificial Neural Network.
- **C-ANN:** Contingency-oriented Artificial Neural Network.

The ANN components use forecasts from the eight atmospheric TFT models to generate complementary ozone estimates. These estimates replace the main TFT ozone forecast between 13:00 and 19:00, inclusive. Outside this interval, the original TFT predictions are retained.

The system uses a 500-hour historical input window.

## Repository Structure

```text
ICAyCC-HEOFS/
├── C-ANN results/
│   └── Figures/
├── models/
├── .gitignore
├── HEOFS_predictor.py
├── README.md
├── red_neuronal.py
├── train_ann_bootstrap.py
├── train_ann_normal.py
├── train_tft_o3.py
└── train_tft_precursors.py
```

| File | Description |
|---|---|
| `HEOFS_predictor.py` | Main hybrid forecasting pipeline. |
| `red_neuronal.py` | ANN implementation. |
| `train_tft_o3.py` | Training of the ozone TFT model. |
| `train_tft_precursors.py` | Training of the eight atmospheric TFT models. |
| `train_ann_normal.py` | Training of N-ANN. |
| `train_ann_bootstrap.py` | Training of C-ANN. |
| `models/` | Trained models and checkpoints. |

## Input Variables

The system uses hourly observations of:

| Variable | Description | Unit |
|---|---|---|
| O₃ | Ozone | ppb |
| NO₂ | Nitrogen dioxide | ppb |
| SO₂ | Sulfur dioxide | ppb |
| CO | Carbon monoxide | ppm |
| NOX | Nitrogen oxides | ppb |
| RH | Relative humidity | % |
| TMP | Air temperature | °C |
| Global | Global solar irradiance | W/m² |
| WSP | Wind speed | m/s |

The ANN components also use month and hour as temporal inputs.

## Data Availability

Atmospheric pollutant and meteorological data are publicly available through the Mexico City Atmospheric Monitoring System (SIMAT).

Solar radiation data were obtained from the Solar Radiation Observatory at the Institute of Geophysics, UNAM, through an institutional collaboration agreement and cannot be publicly redistributed. Therefore, the complete dataset is not included in this repository.

## Installation

HEOFS was developed using Python 3.12.10 and the PyTorch deep learning ecosystem.

Install the required dependencies using:

```bash
python -m pip install -r requirements.txt
```

The exact dependency versions will be specified after verification of the development environment.

## Model Training and Forecasting

The training scripts correspond to the following components:

```bash
python train_tft_o3.py
python train_tft_precursors.py
python train_ann_normal.py
python train_ann_bootstrap.py
```

After placing the required trained models in the `models/` directory, run the forecasting pipeline:

```bash
python HEOFS_predictor.py
```

The associated study uses the following data periods:

- **Training:** 2016–2022.
- **Validation:** 2023.
- **Independent testing:** 2024.

