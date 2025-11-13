### Before running the code, make sure to convert the original signals to HI indexes. 
### The merged.csv contains the necessary HI columns.
### You can also see the individual HI index data at data folder for reference.

import warnings, random, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from scipy.signal import savgol_filter

# ---------- Reproducibility ----------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
set_seed(42)

# ---------- Config ----------
SEQ_LEN   = 20     # sequence window length
PRED_STEP = 5      # prediction step ahead
EPOCHS    = 10
LR        = 1e-3
BATCH     = 32
SPIKE_K   = 2.0    # mean + 2*std as spike threshold
DEVICE    = "cpu"  # TCN is not in demand of GPU，you can deploy it on CPU directly
DO_PLOTS  = True   # whether to plot the comparison results --> Choice by yourself

# ---------- 1) Load data ----------
needed = ["FFT_bin_2_H","Kurtosis_V","Skewness_H","RMS_V","RMS_H",
          "FFT_bin_1_V","FFT_bin_1_H","FFT_peak_V"] # possible features needed (HI)
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"merged.csv 缺少必要列：{missing}")

df = df.dropna(subset=["FFT_bin_2_H"]).reset_index(drop=True)
df["RUL"] = range(len(df)-1, -1, -1)

# Train / Test split 80% - 20%
train_size = int(0.8 * len(df))
train_df = df.iloc[:train_size].reset_index(drop=True)
test_df  = df.iloc[train_size:].reset_index(drop=True)

# ---------- 2) Dataset ----------
class SequenceDataset(Dataset):
    """把一维时间序列切成 [seq_len]->预测 pred_step 的监督样本"""
    def __init__(self, series, seq_len, pred_step):
        series = np.asarray(series, dtype=float)
        X, y = [], []
        for i in range(len(series) - seq_len - pred_step):
            X.append(series[i:i+seq_len])
            y.append(series[i+seq_len+pred_step-1])
        self.X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(1)  # [N,1,seq_len]
        self.y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)  # [N,1]
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ---------- 3) Modern TCN ----------
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.down = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    def forward(self, x):
        y = self.relu(self.conv(x))
        y = y[:, :, :x.size(2)]           # Trim off the padding at the end to ensure the residuals are aligned.
        return self.relu(y + self.down(x))

class ModernTCN(nn.Module):
    def __init__(self, input_channels, output_size, num_channels=(32,32,16), kernel_size=3):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch    = input_channels if i == 0 else num_channels[i-1]
            dilation = 2 ** i
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation))
        self.net = nn.Sequential(*layers)
        self.fc  = nn.Linear(num_channels[-1], output_size)
    def forward(self, x):
        y = self.net(x)                  # [B,C,T]
        return self.fc(y[:, :, -1])     

# ---------- 4) Train TCN ----------
train_ds = SequenceDataset(train_df["FFT_bin_2_H"].values, SEQ_LEN, PRED_STEP)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)

model = ModernTCN(1, 1, (32,32,16)).to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), lr=LR)
lossf = nn.MSELoss()

model.train()
for ep in range(EPOCHS):
    total = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        pred = model(xb)
        loss = lossf(pred, yb)
        loss.backward(); opt.step()
        total += float(loss.item()) * len(xb)
    # print(f"Epoch {ep+1}/{EPOCHS} | loss={total/len(train_ds):.6f}") --> print the loss if needed

# ---------- 5) Predict on test ----------
test_ds = SequenceDataset(test_df["FFT_bin_2_H"].values, SEQ_LEN, PRED_STEP)
preds, trues = [], []
model.eval()
with torch.no_grad():
    for xb, yb in DataLoader(test_ds, batch_size=64):
        yhat = model(xb.to(DEVICE)).cpu().numpy().ravel()
        preds.extend(yhat.tolist())
        trues.extend(yb.numpy().ravel().tolist())

# Align preds with test_df
base_idx = SEQ_LEN + PRED_STEP - 1
df_pred = test_df.iloc[base_idx : base_idx + len(preds)].copy()
df_pred["FFT_bin_2_H_pred"] = preds
df_pred["FFT_bin_2_H_true"] = trues

# ---------- 6) Spike helper & features ----------
def _safe_savgol(x, win=5, poly=2):
    x = np.asarray(x, dtype=float)
    if len(x) < 3: return x
    win = min(win, len(x) if len(x)%2==1 else len(x)-1)
    if win < 3: win = 3
    return savgol_filter(x, window_length=win, polyorder=min(poly, win-1))

def compute_slope(x):
    return np.gradient(_safe_savgol(x, 5, 2))

FEATURE_COLS = [
    "FFT_bin_1_V", "FFT_bin_1_H", "FFT_peak_V", "FFT_bin_2_H_pred",
    "fft_slope", "kurt_slope", "skew_slope", "rms_slope", "SE_energy"
]

def make_feature_table(use_spike=True, spike_k=SPIKE_K):
    if use_spike:
        thr = df_pred["FFT_bin_2_H_pred"].mean() + spike_k * df_pred["FFT_bin_2_H_pred"].std()
        sub = df_pred.loc[df_pred["FFT_bin_2_H_pred"] > thr].copy()
        if len(sub) < 5:  # protect against too few samples after spike filtering
            sub = df_pred.copy()
    else:
        sub = df_pred.copy()

    sub["fft_slope"]  = compute_slope(sub["FFT_bin_2_H_pred"])
    sub["kurt_slope"] = compute_slope(sub["Kurtosis_V"])
    sub["skew_slope"] = compute_slope(sub["Skewness_H"])
    sub["rms_slope"]  = compute_slope(sub["RMS_V"])
    sub["SE_energy"]  = sub["RMS_H"] / (sub["RMS_H"] + sub["RMS_V"] + 1e-9)
    sub["Time_to_Failure"] = sub["RUL"].max() - sub["RUL"]
    sub = sub.dropna()

    X = sub[FEATURE_COLS].astype(float).copy()
    y = sub["Time_to_Failure"].astype(float).values
    return sub, X, y

# ---------- 7) Learners ----------
def fit_predict(learner, X, y):
    if learner == 'rf':
        rf = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
        rf.fit(X,y)
        return rf.predict(X), rf, None
    elif learner == 'lgbm':
        lgb = LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=7, random_state=42)
        lgb.fit(X,y)
        return lgb.predict(X), None, lgb
    elif learner == 'ens':
        rf  = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42).fit(X,y)
        lgb = LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=7, random_state=42).fit(X,y)
        return (0.5*rf.predict(X) + 0.5*lgb.predict(X)), rf, lgb
    else:
        raise ValueError("learner must be one of {'rf','lgbm','ens'}")

# ---------- 8) Metrics ----------
def eval_metrics(df_used, pred_ttf):
    """employed metrics: RMSE, MAE, R², MAPE (all normalized and consistent with baselines)"""
    df_used = df_used.copy()
    df_used["Predicted_RUL"] = df_used["RUL"].max() - pred_ttf
    true_norm = df_used["RUL"].values / df_used["RUL"].max()
    pred_norm = df_used["Predicted_RUL"].values / df_used["RUL"].max()

    rmse = float(np.sqrt(mean_squared_error(true_norm, pred_norm)))
    mae  = float(mean_absolute_error(true_norm, pred_norm))
    r2   = float(r2_score(true_norm, pred_norm))

    eps = 1e-5
    mape = float(np.mean(np.abs((true_norm - pred_norm) / (np.maximum(np.abs(true_norm), eps)))) * 100.0)
    return rmse, mae, r2, mape, int(len(df_used))

# ---------- 9) Unified experiment runner ----------
def run_experiment(name, learner='ens', use_spike=True, spike_k=SPIKE_K):
    sub, X, y = make_feature_table(use_spike=use_spike, spike_k=spike_k)
    yhat, rf, lgb = fit_predict(learner, X, y)
    rmse, mae, r2, mape, cov = eval_metrics(sub, yhat)
    return {
        "name": name,
        "RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape,
        "coverage": cov, "use_spike": use_spike,
        "learner": learner, "n_features": X.shape[1]
    }

# ---------- 10) Spike(Yes/No) comparison ----------
def compare_spike_effect(spike_k=2.0):
    rows = []
    for learner in ('rf', 'lgbm'):
        rows.append(run_experiment(name=f"MTCN + {learner.upper()}",
                                   learner=learner, use_spike=False, spike_k=spike_k))
        rows.append(run_experiment(name=f"MTCN + {learner.upper()} + Spike",
                                   learner=learner, use_spike=True,  spike_k=spike_k))
    df_out = pd.DataFrame(rows)[['name','RMSE','MAE','R2','MAPE','coverage','use_spike','learner','n_features']] \
               .sort_values(['learner','use_spike']).reset_index(drop=True)

    # comparison（Spike − No-spike）
    deltas = []
    for L in ('rf','lgbm'):
        no  = df_out[(df_out.learner==L) & (~df_out.use_spike)].iloc[0]
        yes = df_out[(df_out.learner==L) & (df_out.use_spike)].iloc[0]
        deltas.append({
            'learner': L.upper(),
            'RMSE Δ (spike - no)': yes.RMSE - no.RMSE,
            'MAE Δ'              : yes.MAE  - no.MAE,
            'R² Δ'               : yes.R2   - no.R2,
            'MAPE Δ'             : yes.MAPE - no.MAPE,
            'Coverage Δ'         : int(yes.coverage) - int(no.coverage),
        })
    df_delta = pd.DataFrame(deltas)

    print("\n== Spike 对比（RF & LGBM）==")
    print(df_out.to_string(index=False))
    print("\n== Δ（Spike − No-spike）==")
    print(df_delta.to_string(index=False))

    if DO_PLOTS:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for i, L in enumerate(('rf','lgbm')):
            sub = df_out[df_out.learner==L].sort_values('use_spike')
            ax[0].bar([i-0.15, i+0.15], sub['RMSE'].values, width=0.3)
            ax[1].bar([i-0.15, i+0.15], sub['coverage'].values, width=0.3)
        ax[0].set_xticks([0,1]); ax[0].set_xticklabels(['RF','LGBM'])
        ax[1].set_xticks([0,1]); ax[1].set_xticklabels(['RF','LGBM'])
        ax[0].set_ylabel('RMSE'); ax[1].set_ylabel('Coverage')
        ax[0].legend(['No spike','Spike']); ax[1].legend(['No spike','Spike'])
        fig.suptitle('Spike vs No-spike (RF / LGBM)')
        plt.tight_layout(); plt.show()

    return df_out, df_delta

# ablation test for spike effect
# ---- run the comparison ----
df_spike_cmp, df_spike_delta = compare_spike_effect(spike_k=SPIKE_K)

# df_spike_cmp.to_csv("spike_comparison.csv", index=False)
# df_spike_delta.to_csv("spike_comparison_deltas.csv", index=False)