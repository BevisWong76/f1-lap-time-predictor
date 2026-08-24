import json
import numpy as np
import pandas as pd
import joblib
import torch
from src.model import F1LapSeq2Seq
from src.preprocessor import F1Preprocessor
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------
# PAGE & STYLES SETUP
# ---------------------------------------------------------
st.set_page_config(
    page_title="F1 Lap Time Predictor", 
    page_icon="🏎️", 
    layout="wide"
)

# Global Custom Metric CSS
METRIC_CARD_CSS = """
<style>
.metric-card {
    background-color: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 10px;
    padding: 16px 20px;
    margin-top: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02);
}
.metric-label {
    font-size: 13px;
    color: #555555;
    margin-bottom: 6px;
    font-weight: 500;
}
.metric-value {
    font-size: 28px;
    font-weight: 600;
    color: #1a1d20;
    line-height: 1.2;
}
.metric-delta {
    display: inline-block;
    margin-top: 8px;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
}
.delta-good { background-color: #e6f4ea; color: #137333; }
.delta-bad { background-color: #fce8e6; color: #c5221f; }
</style>
"""

st.title("🏎️ F1 Lap Time Sequence Predictor")

# ---------------------------------------------------------
# DATA LOADERS (CACHED)
# ---------------------------------------------------------
@st.cache_data
def load_historical_data():
    splits = {
        "Train": "results/best_train_predictions.parquet",
        "Validation": "results/best_val_predictions.parquet",
        "Test": "results/best_test_predictions.parquet",
    }

    dfs = []
    for split_name, filepath in splits.items():
        try:
            df = pd.read_parquet(filepath)
            df["Data_Split"] = split_name
            dfs.append(df)
        except FileNotFoundError:
            st.warning(f"File not found: {filepath}")

    if not dfs:
        st.error("No prediction parquet files found.")
        return pd.DataFrame()

    full_df = pd.concat(dfs, ignore_index=True)
    full_df["Start_Target_Lap"] = full_df["Target_Laps"].apply(lambda x: x[0] if len(x) > 0 else 0)
    full_df["Target_Laps_Str"] = full_df["Target_Laps"].apply(
        lambda x: f"Laps {x[0]} to {x[-1]}" if len(x) > 0 else str(x)
    )
    return full_df


@st.cache_data
def load_metadata():
    try:
        with open("src/model_metadata.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Metadata file not found.")
        return {}

@st.cache_resource
def load_preprocessor():
    """Load pre-fitted F1Preprocessor instance."""
    try:
        return joblib.load("models/f1_preprocessor.joblib")
    except Exception as e:
        st.error(f"Failed to load preprocessor: {e}")
        return None


@st.cache_data
def load_model_config():
    """Load model architecture hyperparameters and configuration."""
    try:
        with open("models/best_model_config.json", "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load model config: {e}")
        return {}


@st.cache_resource
def load_f1_model(config):
    """Instantiate and load PyTorch model state dict."""
    if not config:
        return None

    try:
        # Instantiate model architecture using config parameters
        model = F1LapSeq2Seq(
            cat_dims=config.get("cat_dims"),
            cont_dim=config.get("cont_dim"),
            hidden_dim=config.get("hidden_dim", 128), 
            embed_dim=config.get("embed_dim", 2),
            forecast_len=config.get("forecast_len", 3),
            num_layers=config.get("num_layers", 3),
            dropout=config.get("dropout", 0.5),
            target_cont_idx=config.get("target_cont_idx")
        )


        # Load trained weights
        state_dict = torch.load("models/best_model.pth", map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.eval()  # Set to evaluation mode for inference
        return model
    except Exception as e:
        st.error(f"Failed to load PyTorch model weights: {e}")
        return None


# Load artifacts into global scope at app launch
df_history = load_historical_data()
meta = load_metadata()

preprocessor = load_preprocessor()
model_config = load_model_config()
model = load_f1_model(model_config)

# ---------------------------------------------------------
# Helper function
# ---------------------------------------------------------

# For mode 2: Apply track status flags based on user selection
def apply_track_status_flags(
    df: pd.DataFrame, event_type: str, start_lap: int, end_lap: int
) -> pd.DataFrame:
    """Applies race control boolean flags based on chosen event type and lap range."""
    df = df.copy()

    # Reset core track status flags
    flag_cols = ["IsYellowFlag", "IsSafetyCar", "IsVSC", "IsRedFlag"]
    df[flag_cols] = 0

    if event_type == "Clean Track (Normal Racing)":
        return df

    mask = (df["Lap"] >= start_lap) & (df["Lap"] <= end_lap)

    # Flag column mapping
    flag_map = {
        "Yellow Flag": "IsYellowFlag",
        "Safety Car (SC)": "IsSafetyCar",
        "Virtual Safety Car (VSC)": "IsVSC",
        "Red Flag": "IsRedFlag",
    }

    if event_type in flag_map:
        df.loc[mask, flag_map[event_type]] = 1

    return df


# ------------------------------------
# NAVIGATION
# ---------------------------------------------------------
mode = st.sidebar.radio(
    "Select Mode", ["Mode 1: Historical Predictions", "Mode 2: Custom Inference"]
)


# =========================================================
# MODE 1: HISTORICAL DATA EVALUATION
# =========================================================
if mode == "Mode 1: Historical Predictions":
    st.subheader("Historical Prediction Explorer")

    if df_history.empty:
        st.stop()

    # Dynamic Cascading Filters
    col_year, col_event, col_driver, col_sample = st.columns(4)

    with col_year:
        available_years = sorted(df_history["Year"].unique(), reverse=True)
        selected_year = st.selectbox("1. Year", available_years, key="select_year")

    year_df = df_history[df_history["Year"] == selected_year]

    with col_event:
        available_events = sorted(year_df["Event"].unique())
        selected_event = st.selectbox("2. Event", available_events, key="select_event")

    event_df = year_df[year_df["Event"] == selected_event]

    with col_driver:
        available_drivers = sorted(event_df["Driver"].unique())
        selected_driver = st.selectbox("3. Driver", available_drivers, key="select_driver")

    driver_df = event_df[event_df["Driver"] == selected_driver]

    with col_sample:
        driver_df_sorted = driver_df.sort_values("Start_Target_Lap")
        sample_options = driver_df_sorted["Target_Laps_Str"].tolist()
        selected_sample_str = st.selectbox("4. Forecast Horizon", sample_options, key="select_sample")

    plot_clicked = st.button("📊 Generate Plot", width='stretch', type="primary")

    # Initialize state or update upon button click
    if "current_sample" not in st.session_state and not driver_df.empty:
        st.session_state["current_sample"] = driver_df[
            driver_df["Target_Laps_Str"] == selected_sample_str
        ].iloc[0].to_dict()

    if plot_clicked:
        matching = driver_df[driver_df["Target_Laps_Str"] == selected_sample_str]
        if not matching.empty:
            st.session_state["current_sample"] = matching.iloc[0].to_dict()

    # RENDER HISTORICAL SAMPLE
    if "current_sample" in st.session_state:
        sample = st.session_state["current_sample"]
        
        # Split Badge Config
        split_colors = {"Train": "blue", "Validation": "orange", "Test": "red"}
        split_color = split_colors.get(sample["Data_Split"], "gray")

        st.markdown(
            f"**Data Split:** :{split_color}[{sample['Data_Split']} Set] | "
            f"**Sample ID:** `{sample['Sample_ID']}`"
        )

        # Plot Construction
        fig = go.Figure()

        # Historical Trace
        fig.add_trace(go.Scatter(
            x=list(sample["Hist_Laps"]),
            y=list(sample["Hist_Times"]),
            mode="lines+markers",
            name="History",
            line=dict(color="#1f77b4", width=3),
            marker=dict(size=8),
        ))

        # Actual Target Trace
        fig.add_trace(go.Scatter(
            x=list(sample["Target_Laps"]),
            y=list(sample["True_Times"]),
            mode="lines+markers",
            name="Actual",
            line=dict(color="#2ca02c", width=3),
            marker=dict(size=8),
        ))

        # Predicted Target Trace
        fig.add_trace(go.Scatter(
            x=list(sample["Target_Laps"]),
            y=list(sample["Pred_Times"]),
            mode="lines+markers",
            name="Predicted",
            line=dict(color="#d62728", width=3, dash="dash"),
            marker=dict(symbol="square", size=8),
        ))

        # Anchor Line Indicator
        fig.add_vline(
            x=sample["Anchor_Lap"],
            line_width=2,
            line_dash="dot",
            line_color="gray",
            annotation_text="Anchor Lap",
            annotation_position="top left",
        )

        fig.update_layout(
            title=dict(
                text=f"<b>{sample['Year']} {sample['Event']} | Driver: {sample['Driver']}</b><br><sup>(Sample ID: {sample['Sample_ID']})</sup>",
                x=0.5,
                xanchor="center",
            ),
            xaxis_title="Lap Number",
            yaxis_title="Lap Time (Seconds)",
            hovermode="x unified",
            template="plotly_white",
            height=520,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        st.plotly_chart(fig, width='stretch')

        # Render Metrics
        st.markdown(METRIC_CARD_CSS, unsafe_allow_html=True)

        true_horizon = np.array(sample["True_Times"][1:])
        pred_horizon = np.array(sample["Pred_Times"][1:])
        abs_errors = np.abs(true_horizon - pred_horizon)

        mae = np.mean(abs_errors)
        rmse = np.sqrt(np.mean((true_horizon - pred_horizon) ** 2))
        max_err = np.max(abs_errors)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">Forecast MAE</div>
                    <div class="metric-value">{mae:.3f}s</div>
                    <div class="metric-delta delta-good">↓ Avg Error</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">Forecast RMSE</div>
                    <div class="metric-value">{rmse:.3f}s</div>
                    <div class="metric-delta delta-good">↓ Penalized Error</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with m3:
            delta_class = "delta-bad" if max_err > 0.5 else "delta-good"
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">Max Single-Lap Error</div>
                    <div class="metric-value">{max_err:.3f}s</div>
                    <div class="metric-delta {delta_class}">Worst Off Lap</div>
                </div>""",
                unsafe_allow_html=True,
            )
            

# =========================================================
# MODE 2: CUSTOM INFERENCE (SYNTHETIC DATA)
# =========================================================
else:
    # -----------------------------------------------------
    # 1. RACE & ENVIRONMENTAL SETUP
    # -----------------------------------------------------
    st.markdown("### ⚙️ 1. Race & Environmental Setup")
    with st.expander("Metadata and Weather", expanded=True):
        col_year, col_event, col_driver, col_team = st.columns(4)

        with col_year:
            selected_year = st.number_input(
                "Year", min_value=2018, max_value=2030, value=2026, step=1, key="m2_year"
            )
        with col_event:
            selected_event = st.selectbox("Event", meta.get("events", []), key="m2_event")
        with col_driver:
            selected_driver = st.selectbox("Driver", meta.get("drivers", []), key="m2_driver")
        with col_team:
            selected_team = st.selectbox("Team", meta.get("team", []), key="m2_team")

        col_lap, col_ttemp, col_atemp, col_rain = st.columns(4)

        with col_lap:
            start_lap_num = st.number_input(
                "Starting Lap No.", value=30, step=1, key="m2_start_lap"
            )
        with col_ttemp:
            static_track_temp = st.number_input(
                "Track Temp (°C)", value=38.5, step=0.5, key="m2_track_temp"
            )
        with col_atemp:
            static_air_temp = st.number_input(
                "Air Temp (°C)", value=26.0, step=0.5, key="m2_air_temp"
            )
        with col_rain:
            # Vertical offset aligned using native Streamlit layout
            st.write("") 
            st.write("")
            static_rainfall = st.checkbox("Rainfall Active", value=False, key="m2_rainfall")

    # -----------------------------------------------------
    # 2. TRACK EVENT STATUS (Range-Based Setup)
    # -----------------------------------------------------
    st.markdown("### 🏁 2. Track Event Status")

    min_lap = int(start_lap_num)
    max_lap = min_lap + 12  # 13-lap sequence window

    col_event_type, col_event_range = st.columns([1, 2])

    with col_event_type:
        event_type = st.selectbox(
            "Select Event Condition",
            [
                "Clean Track (Normal Racing)",
                "Yellow Flag",
                "Safety Car (SC)",
                "Virtual Safety Car (VSC)",
                "Red Flag",
            ],
            key="m2_event_type",
        )

    event_start_lap, event_end_lap = min_lap, min_lap

    with col_event_range:
        if event_type != "Clean Track (Normal Racing)":
            event_start_lap, event_end_lap = st.slider(
                f"Select {event_type} Lap Range",
                min_value=min_lap,
                max_value=max_lap,
                value=(min_lap + 2, min_lap + 4),
                step=1,
                key="m2_event_range",
            )
            st.caption(f"🚨 **{event_type}** active from **Lap {event_start_lap}** to **Lap {event_end_lap}**")
        else:
            st.write("")
            st.success("🟢 Track Status: Clean Track for all 13 laps")
            
            
    # =========================================================
    # 3. TELEMETRY & STRATEGY INPUT
    # =========================================================
    st.markdown("### 📋 3. Telemetry & Strategy Input")

    tab1, tab2, tab3 = st.tabs([
        "⏱️ 3.1 Sector & Lap Times",
        "🏎️ 3.2 Tyre Strategy",
        "⚡ 3.3 Speeds & Gap",
    ])

    # ---------------------------------------------------------
    # TAB 1: SECTOR & LAP TIMES (10 Laps)
    # ---------------------------------------------------------
    with tab1:
        
        input_laps = [int(start_lap_num) + i for i in range(10)]

        # Initialize 10-lap input DataFrame in session state
        if "sector_input_df" not in st.session_state:
            st.session_state["sector_input_df"] = pd.DataFrame({
                "Lap": input_laps,
                "Sector1Time": [24.1, 24.2, 24.3, 24.4, 24.5, 32.5, 24.0, 23.9, 24.0, 24.1],
                "Sector2Time": [28.2, 28.3, 28.4, 28.5, 28.6, 42.1, 28.0, 27.9, 28.0, 28.1],
                "Sector3Time": [25.9, 26.0, 26.0, 26.1, 26.0, 27.5, 25.8, 25.8, 25.9, 26.0],
            })
        # Keep lap range updated if starting lap changes
        st.session_state["sector_input_df"]["Lap"] = input_laps

        
        col_input, col_display = st.columns([3, 2])

        # Editable Sector Table (Laps 1-10)
        with col_input:
            st.caption("✏️ **Input Sector Times (Laps 1-10)**")
            edited_sectors = st.data_editor(
                st.session_state["sector_input_df"],
                num_rows="fixed",
                key="editor_sectors",
                width='stretch',
                column_config={
                    "Lap": st.column_config.NumberColumn("Lap", disabled=True),
                    "Sector1Time": st.column_config.NumberColumn("Sector 1 (s)", format="%.3f"),
                    "Sector2Time": st.column_config.NumberColumn("Sector 2 (s)", format="%.3f"),
                    "Sector3Time": st.column_config.NumberColumn("Sector 3 (s)", format="%.3f"),
                },
            )
            st.session_state["sector_input_df"] = edited_sectors

        # Live Calculated Lap Times
        with col_display:
            st.caption("📊 **Live Calculated Lap Times & Deltas**")
            
            lap_times_df = edited_sectors.copy()
            lap_times_df["LapTimeSeconds"] = (
                lap_times_df["Sector1Time"] + lap_times_df["Sector2Time"] + lap_times_df["Sector3Time"]
            )
            lap_times_df["LapTimeDelta"] = lap_times_df["LapTimeSeconds"].diff().fillna(0.0)

            st.dataframe(
                lap_times_df[["Lap", "LapTimeSeconds", "LapTimeDelta"]],
                width='stretch',
                hide_index=True,
                column_config={
                    "Lap": st.column_config.NumberColumn("Lap"),
                    "LapTimeSeconds": st.column_config.NumberColumn("Lap Time (s)", format="%.3f"),
                    "LapTimeDelta": st.column_config.NumberColumn("Delta (s)", format="%+.3f"),
                },
            )
            
    # ---------------------------------------------------------
    # TAB 2: TYRE STRATEGY (13 Laps)
    # ---------------------------------------------------------
    with tab2:
        compound_options = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]
        full_laps = [int(start_lap_num) + i for i in range(13)]

        # Initialize sparse input table in session state
        if "tyre_strategy_df" not in st.session_state:
            st.session_state["tyre_strategy_df"] = pd.DataFrame({
                "Lap": full_laps,
                "Stint": [2] + [None] * 4 + [3] + [None] * 3 + [4] + [None] * 3,
                "Compound": ["MEDIUM"] + [None] * 4 + ["HARD"] + [None] * 3 + ["INTERMEDIATE"] + [None] * 3,
                "Pit Tyre Life": [15] + [None] * 4 + [4] + [None] * 3 + [None] + [None] * 3,
            })

        st.session_state["tyre_strategy_df"]["Lap"] = full_laps
        strat_input = st.session_state["tyre_strategy_df"]

        col_strat_input, col_strat_calc = st.columns([3, 2])

        # Editable Sparse Tyre Strategy Input
        with col_strat_input:
            st.caption("✏️ **Input Tyre Strategy (Laps 1-13)**")
            edited_strat = st.data_editor(
                strat_input,
                num_rows="fixed",
                key="editor_tyre_strat",
                width='stretch',
                height=490,
                column_config={
                    "Lap": st.column_config.NumberColumn("Lap", disabled=True),
                    "Stint": st.column_config.NumberColumn("Stint", min_value=1, max_value=6, step=1),
                    "Compound": st.column_config.SelectboxColumn("Compound", options=compound_options),
                    "Pit Tyre Life": st.column_config.NumberColumn(
                        "Pit Tyre Life",
                        min_value=1,
                        max_value=60,
                        step=1,
                        help="Enter initial tyre age for Row 0 or pit stop laps. Leave empty for fresh tyres.",
                    ),
                },
            )
            st.session_state["tyre_strategy_df"] = edited_strat

            st.caption(
                "ℹ️ **Forward-Fill Logic:** Values only need to be specified on **the first row** or at a **New Stint**. "
                "`Stint` and `Compound` forward-fill automatically. Empty `Pit Tyre Life` on new stints defaults to **1 (Fresh Set)**."
            )

        # Derived Strategy Metrics Output
        with col_strat_calc:
            st.caption("📊 **Stint and Tyre metrics**")
            df_filled = edited_strat.copy()
            df_filled["Stint"] = df_filled["Stint"].ffill()
            df_filled["Compound"] = df_filled["Compound"].ffill()

            stint_age_list, tyre_life_list, fresh_tyre_list = [], [], []
            prev_stint = None
            current_stint_age, current_tyre_life = 1, 1

            for idx, row in df_filled.iterrows():
                stint = row["Stint"]
                pit_life = row["Pit Tyre Life"]

                if idx == 0 or stint != prev_stint:
                    prev_stint = stint
                    current_stint_age = 1
                    current_tyre_life = int(pit_life) if pd.notna(pit_life) and pit_life > 0 else 1
                else:
                    current_stint_age += 1
                    current_tyre_life += 1

                stint_age_list.append(current_stint_age)
                tyre_life_list.append(current_tyre_life)
                fresh_tyre_list.append(current_tyre_life == 1)

            # Build derived matrix
            display_df = pd.DataFrame({
                "Lap": full_laps,
                "Stint": df_filled["Stint"],
                "Compound": df_filled["Compound"],
                "Stint_Age": stint_age_list,
                "TyreLife": tyre_life_list,
                "TyreLifeSquared": np.square(tyre_life_list),
                "FreshTyre": fresh_tyre_list,
            })

            # Save full forward-filled derived state for downstream assembly
            st.session_state["tyre_derived_df"] = display_df

            st.dataframe(
                display_df[["Lap", "Stint_Age", "TyreLife", "FreshTyre"]],
                width='stretch',
                height=490,
                hide_index=True,
                column_config={
                    "Lap": st.column_config.NumberColumn("Lap"),
                    "Stint_Age": st.column_config.NumberColumn("Stint Age"),
                    "TyreLife": st.column_config.NumberColumn("Total Tyre Life"),
                    "FreshTyre": st.column_config.CheckboxColumn("Fresh Set?"),
                },
            )
            
    # ---------------------------------------------------------
    # TAB 3: SPEEDS & GAP (10 Laps Input)
    # ---------------------------------------------------------
    with tab3:
        # Initialize 10-lap input DataFrame matching Tab 1 horizon
        if "speed_gap_df" not in st.session_state:
            st.session_state["speed_gap_df"] = pd.DataFrame({
                "Lap": input_laps,
                "SpeedI1": [305.0] * 10,
                "SpeedI2": [285.0] * 10,
                "SpeedFL": [290.0] * 10,
                "SpeedST": [315.0] * 10,
                "IsPersonalBest": [False] * 3 + [True] * 1 + [False] * 6,
                "Relative_Gap": [0.8] * 3 + [1.2] * 4 + [0.5] * 3,
            })

        st.session_state["speed_gap_df"]["Lap"] = input_laps

        col_speed_input, col_speed_calc = st.columns([3, 1])

        # Editable Speeds & Gap Input (10 Laps)
        with col_speed_input:
            st.caption("✏️ **Input Speed and Gap (Laps 1-10)**")
            edited_speeds = st.data_editor(
                st.session_state["speed_gap_df"],
                num_rows="fixed",
                key="editor_speeds_gap",
                width='stretch',
                height=400,
                column_config={
                    "Lap": st.column_config.NumberColumn("Lap", disabled=True),
                    "SpeedI1": st.column_config.NumberColumn("Speed I1 (km/h)", format="%.1f"),
                    "SpeedI2": st.column_config.NumberColumn("Speed I2 (km/h)", format="%.1f"),
                    "SpeedFL": st.column_config.NumberColumn("Speed FL (km/h)", format="%.1f"),
                    "SpeedST": st.column_config.NumberColumn("Speed ST (km/h)", format="%.1f"),
                    "IsPersonalBest": st.column_config.CheckboxColumn("Personal Best?"),
                    "Relative_Gap": st.column_config.NumberColumn("Gap Ahead (s)", format="%.3f"),
                },
            )
            st.session_state["speed_gap_df"] = edited_speeds

        # Derived DRS Status (10 Laps)
        with col_speed_calc:
            st.caption("📊 **Derived DRS Status**")
            display_speed_df = edited_speeds.copy()
            display_speed_df["DRS_Active"] = display_speed_df["Relative_Gap"] <= 1.0

            st.dataframe(
                display_speed_df[["Lap", "Relative_Gap", "DRS_Active"]],
                width='stretch',
                height=400,
                hide_index=True,
                column_config={
                    "Lap": st.column_config.NumberColumn("Lap"),
                    # "Relative_Gap": st.column_config.NumberColumn("Gap (s)", format="%.3f"),
                    "DRS_Active": st.column_config.CheckboxColumn("DRS Active (≤ 1.0s)"),
                },
            )

    # =========================================================
    # 4. FULL FEATURE MATRIX ASSEMBLY
    # =========================================================
    st.divider()
    st.subheader("📦 Full Feature Matrix Assembly")

    # Read Source Inputs
    df_tab1 = edited_sectors.copy()
    df_tab1["LapTimeSeconds"] = df_tab1["Sector1Time"] + df_tab1["Sector2Time"] + df_tab1["Sector3Time"]
    df_tab1["LapTimeDelta"] = df_tab1["LapTimeSeconds"].diff().fillna(0.0)

    df_tab2 = st.session_state.get("tyre_derived_df", display_df).copy()
    df_tab3 = edited_speeds.copy()
    df_tab3["DRS_Active"] = df_tab3["Relative_Gap"] <= 1.0

    # Merge into 13-Lap Sequence (Driven by Tab 2 Strategy Schedule)
    merged = df_tab2.merge(df_tab1, on="Lap", how="left")
    merged = merged.merge(df_tab3, on="Lap", how="left")
    merged.rename(columns={"Lap": "LapNumber"}, inplace=True)

    # Features to zero-mask for unrecorded future laps (Laps 11-13)
    unknown_fut_cols = [
        "LapTimeSeconds", "LapTimeDelta",
        "Sector1Time", "Sector2Time", "Sector3Time", 
        "IsPersonalBest", "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
        "Relative_Gap", "DRS_Active",
    ]

    for col in unknown_fut_cols:
        if col in ["IsPersonalBest", "DRS_Active"]:
            merged[col] = np.where(merged[col].isna(), False, merged[col]).astype(bool)
        else:
            merged[col] = merged[col].fillna(0.0)
            
            
    # Inject Global Race Metadata & Weather Constants
    merged["Year"] = selected_year
    merged["Event"] = selected_event
    merged["Driver"] = selected_driver
    merged["Team"] = selected_team
    merged["TrackTemp"] = static_track_temp
    merged["AirTemp"] = static_air_temp
    merged["Rainfall"] = static_rainfall

    # Apply Dynamic Track Flags per Lap
    merged["Lap"] = merged["LapNumber"]
    merged = apply_track_status_flags(
        df=merged,
        event_type=event_type,
        start_lap=event_start_lap,
        end_lap=event_end_lap,
    )
    merged.drop(columns=["Lap"], inplace=True)

    ordered_cols = [
        "Year", "Event", "Driver", "Team", "LapNumber",
        "LapTimeSeconds", "LapTimeDelta", 
        "Sector1Time", "Sector2Time", "Sector3Time",
        "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",  
        "IsPersonalBest", "Relative_Gap", "DRS_Active",
        "Stint", "Stint_Age", "Compound", "FreshTyre", "TyreLife", "TyreLifeSquared", 
        "IsYellowFlag", "IsRedFlag", "IsSafetyCar", "IsVSC",
        "TrackTemp", "AirTemp", "Rainfall"
    ]
    full_feature_matrix = merged[ordered_cols].copy()

    # Enforce Explicit Boolean Types
    bool_cols = ["IsPersonalBest", "DRS_Active", "FreshTyre", "IsYellowFlag", "IsRedFlag", "IsSafetyCar", "IsVSC"]
    for col in bool_cols:
        full_feature_matrix[col] = full_feature_matrix[col].astype(bool)

    # Store Final Matrix for Inference Pipeline
    st.session_state["full_feature_matrix"] = full_feature_matrix

    # Display 13-Row Matrix Output
    st.dataframe(full_feature_matrix, width='stretch', height=500,)
    st.caption(f"Full Feature Matrix Assembly: `{full_feature_matrix.shape[0]} rows × {full_feature_matrix.shape[1]} columns`")
    st.caption(" ⚠️ Note: Future Laps contain zero-masked values for features that are not yet observed or recorded.")


    # =========================================================
    # 5. MODEL INFERENCE EXECUTION
    # =========================================================
    st.divider()
    st.subheader("🚀 Run Model Inference")

    if st.button("🔮 Predict Remaining Laps", type="primary", width='stretch'):
        if preprocessor is None or model is None:
            st.error("Model or preprocessor artifacts are missing. Check 'models/' directory.")
        else:
            with st.spinner("Processing sequence and running neural network inference..."):
                # 1. Fetch full assembly matrix (13 rows)
                df_matrix = st.session_state["full_feature_matrix"]

                # 2. Transform continuous and categorical feature matrices
                x_cat_np  = preprocessor.transform_cat(df_matrix)   # Shape: (13, n_cat)
                x_cont_np = preprocessor.transform_cont(df_matrix)  # Shape: (13, n_cont)

                # 3. Slice Historical (Laps 1–10) vs. Future (Laps 11–13)
                # Historical Horizon: Rows 0 to 10
                x_cat_hist  = torch.tensor(x_cat_np[:10, :], dtype=torch.long).unsqueeze(0)      # [1, 10, n_cat]
                x_cont_hist = torch.tensor(x_cont_np[:10, :], dtype=torch.float32).unsqueeze(0)  # [1, 10, n_cont]

                # Future Horizon: Rows 10 to 13
                x_cat_fut = torch.tensor(x_cat_np[10:, :], dtype=torch.long).unsqueeze(0)        # [1, 3, n_cat]
                x_cont_fut = torch.tensor(x_cont_np[10:, :], dtype=torch.float32).unsqueeze(0)   # [1, 3, n_cont]

                # 4. Neural Network Inference Pass
                model.eval()
                with torch.no_grad():
                    # Forward signature: (x_cat, x_cont, x_cat_fut, x_cont_fut)
                    raw_predictions = model(
                        x_cat=x_cat_hist,
                        x_cont=x_cont_hist,
                        x_cat_fut=x_cat_fut,
                        x_cont_fut=x_cont_fut,
                        teacher_forcing_ratio=0.0  # Force model to rely purely on predicted outputs
                    )

                    # Output shape: [1, forecast_len, 1] -> squeeze to 1D array of 3 lap predictions
                    predictions_scaled = raw_predictions.squeeze().cpu().numpy()

                # 5. Inverse-Transform Predictions Back to Real Seconds
                # Retrieves target column index dynamically from preprocessor
                try:
                    lap_time_col = "LapTimeSeconds"
                    predictions_seconds = preprocessor.inverse_transform_cont(
                        scaled_values=predictions_scaled, 
                        col_name=lap_time_col
                    )
                except Exception:
                    # Fallback if inverse transform is handled outside
                    predictions_seconds = predictions_scaled

                # Store in session state
                st.session_state["inference_predictions"] = predictions_seconds
                st.success("Inference completed successfully!")

    # =========================================================
    # 6. INFERENCE RESULTS DISPLAY & VISUALIZATION
    # =========================================================
    if "inference_predictions" in st.session_state:
        preds = st.session_state["inference_predictions"]

        st.write("### 📈 Lap Time Forecast Analysis")

        # Read from full feature matrix to preserve actual dynamic lap indexing
        df_matrix = st.session_state["full_feature_matrix"]

        # Historical Laps (First 10 rows)
        hist_df = df_matrix.iloc[:10]
        hist_laps = hist_df["LapNumber"].tolist()
        hist_times = hist_df["LapTimeSeconds"].tolist()

        # Forecast Laps (Last 3 rows)
        fut_df = df_matrix.iloc[10:]
        fut_laps = fut_df["LapNumber"].tolist()

        # Anchor at the last historical lap (e.g., Lap 39)
        anchor_lap = hist_laps[-1]
        anchor_time = hist_times[-1]

        pred_laps_connected = [anchor_lap] + fut_laps
        pred_times_connected = [anchor_time] + list(preds)

        # Construct Plotly Figure
        fig = go.Figure()

        # History Line (Solid Blue with Circle Markers)
        fig.add_trace(
            go.Scatter(
                x=hist_laps,
                y=hist_times,
                mode="lines+markers",
                name="History",
                line=dict(color="#1f77b4", width=3),
                marker=dict(symbol="circle", size=8),
                hovertemplate="Lap %{x}: %{y:.3f}s<extra></extra>",
            )
        )

        # Predicted Line (Dashed Red with Square Markers)
        fig.add_trace(
            go.Scatter(
                x=pred_laps_connected,
                y=pred_times_connected,
                mode="lines+markers",
                name="Predicted",
                line=dict(color="#d62728", width=3, dash="dash"),
                marker=dict(symbol="square", size=8),
                hovertemplate="Lap %{x}: %{y:.3f}s<extra></extra>",
            )
        )

        # Vertical Anchor Reference Line
        fig.add_vline(
            x=anchor_lap,
            line_width=2,
            line_dash="dot",
            line_color="gray",
            annotation_text="Anchor Lap",
            annotation_position="top left",
        )

        # Layout Styling
        fig.update_layout(
            xaxis_title="Lap Number",
            yaxis_title="Lap Time (Seconds)",
            hovermode="x unified",
            margin=dict(l=40, r=40, t=40, b=40),
            height=450,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            xaxis=dict(dtick=1, gridcolor="#f0f0f0"),
            yaxis=dict(gridcolor="#f0f0f0"),
            template="plotly_white",
        )

        # Display Table & Plot
        col_table, col_chart = st.columns([1, 3])

        with col_table:
            st.caption("📋 **Predicted Horizon Output**")
            pred_df = pd.DataFrame(
                {"Lap": fut_laps, "Predicted Lap Time (s)": preds}
            )
            st.dataframe(
                pred_df,
                hide_index=True,
                width='stretch',
                column_config={
                    "Lap": st.column_config.NumberColumn("Lap"),
                    "Predicted Lap Time (s)": st.column_config.NumberColumn(
                        format="%.3f s"
                    ),
                },
            )

        with col_chart:
            st.plotly_chart(fig, width='stretch')