# Time Series Forecasting & GenAI Trend Analysis using TFT

This repository contains a Streamlit app that trains a Temporal Fusion Transformer (TFT) on a single time series using a sliding‑window forecasting approach, produces a multi‑step forecast, visualizes history + forecast, reports an in‑sample RMSE, and generates a short natural‑language trend summary using a Hugging Face LLM (Mistral) via LangChain.

---

## Features

* Upload CSV with a date column and a target column (optionally additional time‑varying features).
* Configure look‑back window and forecast horizon via sliders.
* Train a Temporal Fusion Transformer (PyTorch Forecasting) through PyTorch Lightning.
* Produce an iterative (sliding‑window) multi‑step forecast.
* Plot historical series + forecast using Matplotlib (rendered in Streamlit).
* Compute in‑sample RMSE for quick model quality assessment.
* Summarize the recent trend using a HuggingFace LLM via LangChain.

---

## Important notes about the implementation

* The app disables TorchDynamo optimizations at the top with `os.environ["PL_DISABLE_DYNAMO"] = "1"` to avoid runtime compilation issues on some environments.
* The example is built for a **single group / single time series** (the code sets `id = 0` for the whole dataset). For multiple time series, you'd need to provide group IDs and modify the dataset and data loaders.
* The GenAI summary step uses `st.secrets["HUGGINGFACEHUB_API_TOKEN"]`. You must store your Hugging Face token in Streamlit secrets (see below).

---

## Requirements

This project was developed with Python 3.8+. The primary dependencies include:

* streamlit
* pandas
* numpy
* matplotlib
* torch
* pytorch-lightning
* pytorch-forecasting
* torchmetrics
* langchain
* huggingface-hub

```

> Note: exact versions depend on your environment (CUDA vs CPU). If you use GPU, install a matching `torch` binary from pytorch.org.

---

## Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd <repo-folder>
```

2. (Recommended) Create and activate a virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
# or
pip install streamlit pandas numpy matplotlib torch pytorch-lightning pytorch-forecasting torchmetrics langchain huggingface-hub
```

4. Provide a Hugging Face API token (for the GenAI summary):

Create a file at `.streamlit/secrets.toml` with the following content (replace the placeholder with your token):

```toml
[HUGGINGFACEHUB]
HUGGINGFACEHUB_API_TOKEN = "hf_XXXXXXXXXXXXXXXXXXXXXXXX"
```

Alternatively, you can use the Streamlit cloud secrets UI if deploying there. The app reads `st.secrets["HUGGINGFACEHUB_API_TOKEN"]`.

---

## Usage

Start the app:

```bash
streamlit run main.py
```

1. Upload a CSV file. The CSV must have a date column and a numeric target column. The app will parse and let you select which columns are date, target, and optional extra features.
2. Choose the **Look‑back window size** and the **Forecast horizon** using the sliders.
3. Click **Train Model** to fit the TFT (the app uses `max_epochs=10` by default).
4. When training completes, the app will perform iterative sliding‑window forecasting for the specified horizon, plot the results, show an in‑sample RMSE, and run a short GenAI trend summary using the Hugging Face LLM.

---

## CSV expectations & preprocessing

* The app converts the selected date column to `datetime` and sorts the data by date.
* It creates a monotonically increasing `time_idx` (dense rank) used by `TimeSeriesDataSet`.
* The code is written for a *single sequence* (it sets `id = 0` for all rows). To support multiple series, add a column with group IDs and pass it into `group_ids`.
* The code attempts to infer the calendar frequency using `pd.infer_freq`. If inference fails, it defaults to daily frequency.

---

## Configuration and tuning

Top-of-file and dataset parameters you might want to change:

```python
# model / dataset
lookback = ...        # slider in Streamlit
horizon = ...         # slider in Streamlit
batch_size = 16       # in the code: ts_data.to_dataloader(... batch_size=16)
max_epochs = 10       # Trainer max_epochs
hidden_size = 16      # TemporalFusionTransformer.from_dataset(...)
learning_rate = 0.03
```

* Increasing `max_epochs`, `hidden_size`, or batch size can improve performance but increases training time and memory usage.
* Training is significantly faster on a GPU. If you have CUDA available, configure the PyTorch Lightning `Trainer` to use `accelerator='gpu'` and set `devices`.

---

## Performance & resource considerations

* TFTs and PyTorch Forecasting can be memory‑ and compute‑intensive. On large datasets or long lookback windows, use GPU and adjust batch sizes.
* The app uses an iterative single‑step prediction loop to construct the multi‑step forecast. This is simple but can be slower than vectorized multi‑step prediction for long horizons.

---

## Troubleshooting

* **Model training fails with CUDA/torch errors**: ensure `torch` version matches your CUDA runtime, or switch to CPU build if you don't have CUDA.
* **`pd.infer_freq` returns `None`**: the app falls back to daily frequency, but you may want to explicitly pass frequency (e.g. `freq='D'`) if your timestamps are irregular.
* **Hugging Face LLM errors**: ensure your token is correct and your environment has internet access. Some large models may require additional configuration or account permissions on Hugging Face.

---

## Security & secrets

* Do **not** commit `.streamlit/secrets.toml` to a public repository. Use Streamlit Cloud secrets or environment variables for deployment.

---
