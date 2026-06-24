import random
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import gdown    
import torch
import torch.nn as nn
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, RobustScaler, MinMaxScaler

st.set_page_config(page_title="F1 Lap Time Forecasting", layout="wide")
st.title("🏎️ F1 Lap Time Sequence-to-Sequence Forecasting")

# ==========================================
# 1. Model Architecture
# ==========================================
class F1LapSeq2Seq(nn.Module):
    def __init__(self, cat_dims, cont_dim, hidden_dim=128, embed_dim=6, 
                 forecast_len=3, num_layers=3, dropout=0.3, target_col_idx=-1):
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.forecast_len = forecast_len
        self.target_col_idx = target_col_idx
        
        self.embeddings = nn.ModuleList([nn.Embedding(d, embed_dim) for d in cat_dims])
        self.total_cat_dim = len(cat_dims) * embed_dim
        self.input_dim = self.total_cat_dim + cont_dim
        
        self.encoder = nn.LSTM(self.input_dim, hidden_dim, num_layers=num_layers,
                               batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.decoder = nn.LSTM(self.input_dim, hidden_dim, num_layers=num_layers,
                               batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
    
    def forward(self, x_cat, x_cont, x_cat_fut, x_cont_fut, y_target=None, teacher_forcing_ratio=0.5):
        cat_embs = [emb(x_cat[:, :, i]) for i, emb in enumerate(self.embeddings)]
        cat_emb = torch.cat(cat_embs, dim=-1)
        x = torch.cat([cat_emb, x_cont], dim=-1)
        
        _, (hidden, cell) = self.encoder(x)
        
        outputs = []
        next_val = x_cont[:, -1, self.target_col_idx]
        
        for i in range(self.forecast_len):
            step_cat = x_cat_fut[:, i:i+1, :]
            step_cont = x_cont_fut[:, i:i+1, :].clone()
            
            step_cont[:, 0, self.target_col_idx] = next_val
            
            step_cat_embs = [emb(step_cat[:, :, j]) for j, emb in enumerate(self.embeddings)]
            step_cat_emb = torch.cat(step_cat_embs, dim=-1)
            
            decoder_input = torch.cat([step_cat_emb, step_cont], dim=-1)
            
            out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            pred = self.fc(self.dropout(out))
            outputs.append(pred)
            
            if i < self.forecast_len - 1:
                if y_target is not None and random.random() < teacher_forcing_ratio:
                    next_val = y_target[:, i, 0]
                else:
                    next_val = pred.view(-1)
                    
        return torch.cat(outputs, dim=1)


# Feature Groups / Categorization for Preprocessing
embedding_features = ['Year', 'Event', 'Driver', 'Team']          

onehot_features = ['Compound']                   

robust_features = [
    'LapTimeDelta', 'LapTimeSeconds', 'Sector1Time', 'Sector2Time', 'Sector3Time',
    'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST','Relative_Gap',
    'TrackTemp', 'AirTemp'
]

minmax_features = ['LapNumber', 'Stint', 'Stint_Age', 'TyreLife', 'TyreLifeSquared']

bool_features = [
    'IsPersonalBest', 'FreshTyre', 'DRS_Active',
    'IsSafetyCar', 'IsRedFlag', 'IsVSC', 'IsPitClosed', 'IsSCEnding',
    'Rainfall',
]

class F1Preprocessor:
    def __init__(self, embedding_cols=embedding_features,
                       onehot_cols=onehot_features,
                       robust_cols=robust_features,
                       minmax_cols=minmax_features,
                       bool_cols=bool_features):
        # Initialize the F1Preprocessor with specified feature groups
        self.embedding_cols = embedding_cols
        self.onehot_cols = onehot_cols
        self.robust_cols = robust_cols
        self.minmax_cols = minmax_cols
        self.bool_cols = bool_cols

        # Initialize the encoders and scalers
        self.label_encoders = {}
        self.onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.robust_scaler = RobustScaler()
        self.minmax_scaler = MinMaxScaler()

        # Initialize lists to keep track of valid columns for each transformation
        self.valid_robust = []
        self.valid_minmax = []
        self.valid_onehot = []
        self.feature_names_out = None

    def fit(self, df):
        df = df.copy()
        
        # 1. Fit Embeddings (1-indexed, 0 is saved for <UNK>)
        for col in self.embedding_cols:
            if col in df.columns:
                unique = sorted(df[col].dropna().unique())
                self.label_encoders[col] = {val: i + 1 for i, val in enumerate(unique)}
        
        # 2. Fit One-Hot 
        self.valid_onehot = [c for c in self.onehot_cols if c in df.columns]
        if self.valid_onehot:
            self.onehot_encoder.fit(df[self.valid_onehot].fillna('missing'))
        
        # 3. Fit Scalers (Robust & MinMax)
        self.valid_robust = [c for c in self.robust_cols if c in df.columns]
        if self.valid_robust:
            self.robust_scaler.fit(df[self.valid_robust].fillna(0))
            
        self.valid_minmax = [c for c in self.minmax_cols if c in df.columns]
        if self.valid_minmax:
            self.minmax_scaler.fit(df[self.valid_minmax].fillna(0))
        
        # 4. Create Feature Names
        self._build_feature_names(df)
        return self

    # Build the output feature names after transformation
    def _build_feature_names(self, df):
        self.feature_names_out = []
        
        for col in df.columns:
            if col in self.embedding_cols:
                self.feature_names_out.append(col)
            
            elif col in self.onehot_cols and col in self.valid_onehot:
                oh_names = self.onehot_encoder.get_feature_names_out([col])
                self.feature_names_out.extend(oh_names)
            
            elif col in self.robust_cols or col in self.minmax_cols or col in self.bool_cols:
                self.feature_names_out.append(col)
        
        return self.feature_names_out

    def transform(self, df):
        df = df.copy()
        processed = []
        
        # 1. Transform One-Hot Encoded Features
        oh_transformed = {}
        if self.valid_onehot:
            oh_array = self.onehot_encoder.transform(df[self.valid_onehot].fillna('missing'))
            # Split the transformed array back into individual columns
            start_idx = 0
            for col in self.valid_onehot:
                cats_len = len(self.onehot_encoder.categories_[self.valid_onehot.index(col)])
                oh_transformed[col] = oh_array[:, start_idx:start_idx + cats_len]
                start_idx += cats_len

        # 2. Transform Robust Scaled Features
        robust_transformed = {}
        if self.valid_robust:
            rob_array = self.robust_scaler.transform(df[self.valid_robust].fillna(0))
            for i, col in enumerate(self.valid_robust):
                robust_transformed[col] = rob_array[:, i:i+1]

        # 3. Transform Min-Max Scaled Features
        minmax_transformed = {}
        if self.valid_minmax:
            mm_array = self.minmax_scaler.transform(df[self.valid_minmax].fillna(0))
            for i, col in enumerate(self.valid_minmax):
                minmax_transformed[col] = mm_array[:, i:i+1]

        # 4. Combine Features in Original Column Order
        for col in df.columns:
            if col in self.embedding_cols:
                # Map categorical values to their corresponding integer indices, 
                # fill NaN with 0 (for <UNK>), and reshape to a column vector
                vals = df[col].map(self.label_encoders.get(col, {})).fillna(0).astype(int).values.reshape(-1, 1)
                processed.append(vals)
                
            elif col in self.onehot_cols and col in oh_transformed:
                processed.append(oh_transformed[col])
                
            elif col in self.robust_cols and col in robust_transformed:
                processed.append(robust_transformed[col])
                
            elif col in self.minmax_cols and col in minmax_transformed:
                processed.append(minmax_transformed[col])
                
            elif col in self.bool_cols:
                vals = df[col].fillna(0).astype(int).values.reshape(-1, 1)
                processed.append(vals)
        
        return np.hstack(processed)

    def get_feature_names(self):
        return self.feature_names_out.copy()

    def inverse_transform_field(self, scaled_values, col_name):
        # Make sure the input is 2D shape (N, 1)
        scaled_values_2d = np.array(scaled_values).reshape(-1, 1)
        
        # Case A: The column belongs to Robust Scaler
        if col_name in self.valid_robust:
            col_idx = self.valid_robust.index(col_name)
            center = self.robust_scaler.center_[col_idx]
            scale = self.robust_scaler.scale_[col_idx]
            # RobustScaler inverse formula: original = scaled * scale + center
            return (scaled_values_2d * scale + center).flatten()
            
        # Case B: The column belongs to MinMax Scaler
        elif col_name in self.valid_minmax:
            col_idx = self.valid_minmax.index(col_name)
            min_val = self.minmax_scaler.data_min_[col_idx]
            max_val = self.minmax_scaler.data_max_[col_idx]
            # MinMaxScaler inverse transform: original = scaled * (max - min) + min
            return (scaled_values_2d * (max_val - min_val) + min_val).flatten()
            
        else:
            raise ValueError(f"Column '{col_name}' is not in either robust_cols or minmax_cols!")


# ==========================================
# 2. Resource Loaders
# ==========================================
@st.cache_resource
def load_f1_model(model_path="./models/best_model.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = F1LapSeq2Seq(
        cat_dims=[5, 33, 34, 14], 
        cont_dim=31,
        hidden_dim=128,
        embed_dim=6, 
        forecast_len=3,
        num_layers=3, 
        dropout=0.2, 
        target_col_idx=-1
    )
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    return model, device

@st.cache_resource
def load_preprocessor(preprocessor_path="models/preprocessor.joblib"):
    return joblib.load(preprocessor_path)

@st.cache_resource
def load_dataset_tensors():
    file_id = "1SEyNyL02peJvzO7UGMo-ShPoDEqM787d" 
    output_path = Path("models/val_tensors.pt")

    if not output_path.exists():
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(output_path), quiet=False)

    all_tensors = torch.load(output_path, weights_only=False)
    return all_tensors

# ==========================================
# 3. Helper Functions & Pipeline
# ==========================================
def predict_and_unscale(model, preprocessor, sample_dict, sample_idx, device="cpu"):
    """
    Perform a single prediction and unscale the results.
    """
    # 1. Extract Sample Data and Move to Device
    x_cat = sample_dict["x_cat"][sample_idx : sample_idx + 1].to(device)
    x_cont = sample_dict["x_cont"][sample_idx : sample_idx + 1].to(device)
    x_cat_fut = sample_dict["x_cat_fut"][sample_idx : sample_idx + 1].to(device)
    x_cont_fut = sample_dict["x_cont_fut"][sample_idx : sample_idx + 1].to(device)
    
    # 2. Model Forward Pass (Inference)
    with torch.no_grad():
        scaled_preds = model(
            x_cat, x_cont, x_cat_fut, x_cont_fut, 
            y_target=None, teacher_forcing_ratio=0.0
        )
    
    # 3. Convert to 1D Numpy Array
    scaled_preds = scaled_preds.squeeze(0).cpu().numpy().flatten()
    scaled_y_actual = sample_dict["y"][sample_idx].cpu().numpy().flatten()
    
    # 3. Unscale for Delta (Seconds)
    pred_deltas = preprocessor.inverse_transform_field(scaled_preds, "LapTimeDelta")
    actual_deltas = preprocessor.inverse_transform_field(scaled_y_actual, "LapTimeDelta")
    
    # 4.  Lap Time (Unscale Continuous Features
    lap_time_col_idx = -2 
    scaled_history = x_cont[0, :, lap_time_col_idx].cpu().numpy()
    history_lap_times = preprocessor.inverse_transform_field(scaled_history, "LapTimeSeconds")
    
    # 5. Get Last Known Lap Time
    last_known_lap_time = history_lap_times[-1]
    
    # 6. Compute Predicted and Actual Lap Times
    pred_lap_times = last_known_lap_time + np.cumsum(pred_deltas)
    actual_lap_times = last_known_lap_time + np.cumsum(actual_deltas)
    
    return history_lap_times, pred_lap_times, actual_lap_times


def plot_lap_time_predictions(history_laps, pred_laps, actual_laps=None, title="Lap Time Forecasting"):
    """
    Generate a plot comparing historical lap times, predicted lap times, and actual lap times.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))
    
    hist_len = len(history_laps)
    fut_len = len(pred_laps)
    
    # X-axis Sequence Steps
    hist_x = np.arange(1, hist_len + 1)
    fut_x = np.arange(hist_len, hist_len + fut_len + 1) # 連接歷史最後一點
    
    # 1. Plot Historical Lap Times
    ax.plot(hist_x, history_laps, 'k-o', label="History (Observed)", markersize=4, linewidth=1.5)
    
    # 2. Plot Actual Ground Truth
    if actual_laps is not None:
        full_actual = np.append(history_laps[-1], actual_laps)
        ax.plot(fut_x, full_actual, 's-', color='#1f77b4', label="Actual Ground Truth", markersize=5, linewidth=2)
    
    # 3. Plot Model Predictions
    full_pred = np.append(history_laps[-1], pred_laps)
    ax.plot(fut_x, full_pred, 'x--', color='#d62728', label="Model Prediction", markersize=6, linewidth=1.8)
    
    # 4. Add Vertical Line to Separate History and Forecast
    ax.axvline(x=hist_len, color='gray', linestyle=':', alpha=0.7, linewidth=1.5)

    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_xlabel("Sequence Step (Laps)", fontsize=10)
    ax.set_ylabel("Lap Time (Seconds)", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc="best")
    ax.set_xticks(np.arange(1, hist_len + fut_len + 1))
    
    plt.tight_layout()
    return fig

# ==========================================
# 4. Instantiate Artifacts
# ==========================================
model, device = load_f1_model()
preprocessor = load_preprocessor()
all_tensors = load_dataset_tensors()


# =========================================================
# 5: Main UI
# =========================================================
# Sidebar: Mode Selection
st.sidebar.title("🎮 Mode Selection")
mode = st.sidebar.radio("Choose Mode:", ["Mode 1: Historical Evaluation", "Mode 2: Custom Strategy Simulator"])

meta_df = all_tensors["meta"]

# =========================================================
# Mode 1: Historical Evaluation
# =========================================================
if mode == "Mode 1: Historical Evaluation":
    st.subheader("📊 Single Sequence Prediction & Evaluation")

    meta_df = all_tensors["meta"]

    # 1. 4 Dropdowns for Year, Event, Driver, Lap Number
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        years = sorted(meta_df["Year"].unique())
        selected_year = st.selectbox("Year", years)
    filtered_df = meta_df[meta_df["Year"] == selected_year]

    with col2:
        events = sorted(filtered_df["Event"].unique())
        selected_event = st.selectbox("Event", events)
    filtered_df = filtered_df[filtered_df["Event"] == selected_event]

    with col3:
        drivers = sorted(filtered_df["Driver"].unique())
        selected_driver = st.selectbox("Driver", drivers)
    filtered_df = filtered_df[filtered_df["Driver"] == selected_driver]

    with col4:
        laps = sorted(filtered_df["LapNumber"].unique())
        selected_lap = st.selectbox("Lap Number", laps)

    # 2. Find the Matching Sample Index
    matched = meta_df[
        (meta_df["Year"] == selected_year) &
        (meta_df["Event"] == selected_event) &
        (meta_df["Driver"] == selected_driver) &
        (meta_df["LapNumber"] == selected_lap)
    ]

    if not matched.empty:
        sample_idx = matched.index[0]
        data_split = matched.iloc[0]["Split"]  # 👈 自動取得 'Train', 'Validation', 或 'Test'
        
        # 3. Notice the Data Split Source
        split_colors = {
            "Train": "🔵 Notice: This sequence is from the **Training Set**.",
            "Validation": "🟡 Notice: This sequence is from the **Validation Set**.",
            "Test": "🟢 Notice: This sequence is from the **Test Set** (Unseen Data)."
        }
        st.info(split_colors.get(data_split, f"Notice: Data source is {data_split}."))

        # 4. Perform Prediction and Unscale
        history_laps, pred_laps, actual_laps = predict_and_unscale(
            model, preprocessor, all_tensors, sample_idx, device=device
        )
        
        # Calculate MAE
        mae = np.mean(np.abs(pred_laps - actual_laps))
        
        # KPI Display
        k1, k2, k3 = st.columns(3)
        k1.metric("Sample Index", f"#{sample_idx}")
        k2.metric("Prediction MAE", f"{mae:.3f} s")
        k3.metric("Data Split Source", data_split)

        # Plotting the Results
        fig = plot_lap_time_predictions(
            history_laps, pred_laps, actual_laps,
            title=f"Sample #{sample_idx} | {selected_driver} - {selected_event} ({selected_year})"
        )
        st.pyplot(fig)

    else:
        st.warning("No sequence found for the selected parameters.")

# =========================================================
# Mode 2: Custom Strategy Simulator (Placeholder)
# =========================================================
else:
    st.subheader("🛠️ Custom Future Feature Simulator")
    st.info("Mode 2 is under construction. Future Strategy Widgets will be added here.")