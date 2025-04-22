import os
os.environ["PL_DISABLE_DYNAMO"] = "1"  # Disable TorchDynamo compilation

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import torch
from torchmetrics import MeanSquaredError

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer
from pytorch_lightning import Trainer, LightningModule

from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ————————————————————————————————
# Streamlit UI
# ————————————————————————————————
st.title("Sliding‑Window Forecasting + Future Visualization")

# 1) Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if not uploaded_file:
    st.stop()

df = pd.read_csv(uploaded_file)
df.columns = df.columns.str.strip()
date_col   = st.selectbox("Date column",   df.columns)
target_col = st.selectbox("Target column", df.columns)
feature_cols = st.multiselect(
    "Optional extra features", 
    [c for c in df.columns if c not in [date_col, target_col]]
)

# Preview original data
st.write("## Data preview")
st.dataframe(df[[date_col, target_col] + feature_cols].head())

# Rename and preprocess
df[date_col] = pd.to_datetime(df[date_col])
df.rename(columns={date_col: "date", target_col: "target"}, inplace=True)
df = df.sort_values("date").reset_index(drop=True)
df["time_idx"] = df["date"].rank(method="dense").astype(int)
df["id"] = 0  # single‑group

# user‑selectable window & horizon
lookback = st.slider("Look‑back window size", min_value=5, max_value=50, value=12)
horizon  = st.slider("Forecast horizon (steps)",    min_value=1, max_value=252, value=30)

# 2) Build TimeSeriesDataSet with GroupNormalizer(softplus)
target_normalizer = GroupNormalizer(
    groups=["id"],
    transformation="softplus"    # smooth, positive-only scaling
)
ts_data = TimeSeriesDataSet(
    df,
    time_idx="time_idx",
    target="target",
    group_ids=["id"],
    max_encoder_length=lookback,
    max_prediction_length=horizon,
    time_varying_known_reals=["time_idx"] + feature_cols,
    time_varying_unknown_reals=["target"],
    target_normalizer=target_normalizer,
)
train_loader = ts_data.to_dataloader(train=True, batch_size=16, num_workers=0)

# 3) Define TFT & Trainer
tft = TemporalFusionTransformer.from_dataset(
    ts_data,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=2,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=1,
    loss=MeanSquaredError(),
    log_interval=10,
    reduce_on_plateau_patience=4,
)

class TFTWrapper(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x)
    def training_step(self, batch, idx):
        self.model.log = lambda *a, **k: None
        if self.model.trainer is None:
            self.model.trainer = self.trainer
        return self.model.training_step(batch, idx)
    def validation_step(self, batch, idx):
        self.model.log = lambda *a, **k: None
        if self.model.trainer is None:
            self.model.trainer = self.trainer
        return self.model.validation_step(batch, idx)
    def configure_optimizers(self):
        return self.model.configure_optimizers()
    def on_fit_start(self):
        self.model.trainer = self.trainer

trainer = Trainer(
    max_epochs=10,
    enable_model_summary=False,
    logger=False,
    enable_checkpointing=False
)

# 4) Train Button & Control Flow
if "trained" not in st.session_state:
    st.session_state.trained = False

if not st.session_state.trained:
    if st.button("Train Model"):
        st.session_state.trained = True
    else:
        st.info("Click **Train Model** to start training, then the app will continue.")
        st.stop()

with st.spinner("Training TFT model…"):
    trainer.fit(model=TFTWrapper(tft), train_dataloaders=train_loader)
st.success("Training complete!")

# 5) Sliding‑window iterative forecasting
window_df = df.tail(lookback).copy().reset_index(drop=True)
preds = []
for _ in range(horizon):
    # build next row
    new_time_idx = int(window_df["time_idx"].iloc[-1] + 1)
    row = { "time_idx": new_time_idx, "id": 0 }
    for feat in feature_cols:
        row[feat] = window_df[feat].iloc[-1]
    row["target"] = 0.0  # dummy

    tmp = pd.concat([window_df, pd.DataFrame([row])], ignore_index=True)
    ds = TimeSeriesDataSet(
        tmp,
        time_idx="time_idx",
        target="target",
        group_ids=["id"],
        max_encoder_length=lookback,
        max_prediction_length=1,
        time_varying_known_reals=["time_idx"] + feature_cols,
        time_varying_unknown_reals=["target"],
        target_normalizer=target_normalizer,
    )
    dl = ds.to_dataloader(train=False, batch_size=1)
    preds.append(tft.predict(dl)[0].item())

    # slide window
    window_df = tmp.tail(lookback).reset_index(drop=True)
    window_df.at[lookback-1, "target"] = preds[-1]
    window_df.at[lookback-1, "time_idx"] = new_time_idx

# Build future date index
last_date = df["date"].iloc[-1]
freq = pd.infer_freq(df["date"]) or "D"
future_dates = pd.date_range(start=last_date + pd.Timedelta(1, unit=freq[0]),
                             periods=horizon, freq=freq)

# 6) Plot history + forecast
plt.figure(figsize=(10, 5))
plt.plot(df["date"], df["target"], label="Historical")
plt.plot(future_dates, preds, "--", label="Forecast")
plt.xlabel("Date")
plt.ylabel(target_col)
plt.legend()
st.pyplot(plt.gcf())

# 7) In‑sample RMSE
test_dl = ts_data.to_dataloader(train=False, batch_size=1)
all_preds, all_targs = [], []
for batch in test_dl:
    batch = batch[0] if isinstance(batch, tuple) else batch
    with torch.no_grad():
        out = tft(batch)
    all_preds.extend(out.prediction.cpu().numpy().flatten())
    all_targs.extend(batch["decoder_target"].cpu().numpy().flatten())
rmse = np.sqrt(np.mean((np.array(all_preds) - np.array(all_targs))**2))
st.write(f"**In‑sample RMSE**: {rmse:.4f}")

# 8) GenAI Trend Summary with Mistral on HF
st.subheader("GenAI Trend Summary")
HUGGINGFACEHUB_API_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
hf_llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    model_kwargs={"temperature":0.7, "max_new_tokens":200},
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)
prompt = PromptTemplate(
    input_variables=["last_actual","first_forecast"],
    template=(
        "Last observed value: {last_actual:.2f}. "
        "First forecast step: {first_forecast:.2f}. "
        "Provide a concise, plain‑English summary of the trend."
    )
)
chain = LLMChain(prompt=prompt, llm=hf_llm)
summary = chain.run(
    last_actual    = df["target"].iloc[-1],
    first_forecast = preds[0]
)
st.write(summary)
