import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder, RobustScaler, MinMaxScaler

# Feature Groups / Categorization for Preprocessing

# Embedding features
embedding_features = ['Year', 'Event', 'Driver', 'Team']          

# One-hot encoding features
onehot_features = ['Compound']                   

# Robust scaling features
robust_features = [
    'LapTimeDelta', 'LapTimeSeconds', 
    'Sector1Time', 'Sector2Time', 'Sector3Time',
    'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST',
    'Relative_Gap',
    'TrackTemp', 'AirTemp'
]

# Min-max scaling features
minmax_features = [
    'LapNumber', 'Stint', 'Stint_Age', 
    'TyreLife', 'TyreLifeSquared'
]

# Boolean features
bool_features = [
    'IsPersonalBest', 'FreshTyre', 'DRS_Active',
    'IsYellowFlag', 'IsRedFlag', 'IsSafetyCar', 'IsVSC', 
    'Rainfall',
]


# Preprocessor Class for F1 Data
class F1Preprocessor:

    def __init__(
        self,
        embedding_cols=embedding_features,
        onehot_cols=onehot_features,
        robust_cols=robust_features,
        minmax_cols=minmax_features,
        bool_cols=bool_features,
    ):
        self.embedding_cols = embedding_cols or []
        self.onehot_cols = onehot_cols or []
        self.robust_cols = robust_cols or []
        self.minmax_cols = minmax_cols or []
        self.bool_cols = bool_cols or []

        self.label_encoders = {}
        self.onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.robust_scaler = RobustScaler()
        self.minmax_scaler = MinMaxScaler()

        self.valid_onehot = []
        self.valid_robust = []
        self.valid_minmax = []
        self.valid_bool = []

        self.ordered_cont_cols = ([])
        self.feature_names_all = []
        self.feature_names_cat = []
        self.feature_names_cont = []

    # ==========================================
    # 1. MODULAR FIT METHODS
    # ==========================================
    def fit_categoricals(self, df):
        for col in self.embedding_cols:
            if col in df.columns:
                unique = sorted(df[col].dropna().unique())
                self.label_encoders[col] = {
                    val: i + 1 for i, val in enumerate(unique)
                }
        return self

    def fit_continuous(self, df):
        # Lock column existence based on training set
        self.valid_onehot = [c for c in self.onehot_cols if c in df.columns]
        self.valid_robust = [c for c in self.robust_cols if c in df.columns]
        self.valid_minmax = [c for c in self.minmax_cols if c in df.columns]
        self.valid_bool = [c for c in self.bool_cols if c in df.columns]

        if self.valid_onehot:
            self.onehot_encoder.fit(df[self.valid_onehot])
        if self.valid_robust:
            self.robust_scaler.fit(df[self.valid_robust])
        if self.valid_minmax:
            self.minmax_scaler.fit(df[self.valid_minmax])

        # Save explicit continuous column order matching transform_cont output exactly
        self.ordered_cont_cols = (
            self.valid_onehot
            + self.valid_robust
            + self.valid_minmax
            + self.valid_bool
        )
        return self

    def fit(self, df_train, df_global=None):
        """Backward-compatible master fit."""
        global_data = df_global if df_global is not None else df_train
        self.fit_categoricals(global_data)
        self.fit_continuous(df_train)
        self._build_feature_names(df_train)
        return self

    # ==========================================
    # 2. MODULAR TRANSFORM METHODS
    # ==========================================
    def transform_cat(self, df):
        """Returns 2D NumPy array of 1-indexed categorical integer features."""
        cat_cols = []
        for col in self.embedding_cols:
            if col in df.columns:
                vals = (
                    df[col]
                    .map(self.label_encoders.get(col, {}))
                    .fillna(0)
                    .astype(np.int64)
                    .values.reshape(-1, 1)
                )
                cat_cols.append(vals)

        return np.hstack(cat_cols) if cat_cols else np.empty((len(df), 0))

    def transform_cont(self, df):
        cont_blocks = []

        # 1. One-Hot Features
        if self.valid_onehot:
            oh_array = self.onehot_encoder.transform(df[self.valid_onehot])
            cont_blocks.append(oh_array)

        # 2. Robust Features
        if self.valid_robust:
            rob_array = self.robust_scaler.transform(df[self.valid_robust])
            cont_blocks.append(rob_array)

        # 3. MinMax Features
        if self.valid_minmax:
            mm_array = self.minmax_scaler.transform(df[self.valid_minmax])
            cont_blocks.append(mm_array)

        # 4. Bool Features
        if self.valid_bool:
            bool_array = df[self.valid_bool].astype(int).to_numpy()
            if bool_array.ndim == 1:
                bool_array = bool_array.reshape(-1, 1)
            cont_blocks.append(bool_array)

        return (
            np.hstack(cont_blocks)
            if cont_blocks
            else np.empty((len(df), 0), dtype=np.float32)
        )
    
    def transform(self, df):
        """Combined transformation matrix (Categoricals + Continuous)."""
        x_cat = self.transform_cat(df)
        x_cont = self.transform_cont(df)

        if x_cat.size == 0:
            return x_cont
        if x_cont.size == 0:
            return x_cat
        return np.hstack([x_cat, x_cont])

    # ==========================================
    # 3. UTILITY & DECODING METHODS
    # ==========================================
    def _build_feature_names(self, df):
        self.feature_names_cat = []
        self.feature_names_cont = []

        # 1. Embedding / Categorical Columns
        for col in self.embedding_cols:
            if col in df.columns:
                self.feature_names_cat.append(col)

        # 2. Continuous Columns (Must strictly match transform_cont order!)
        if self.valid_onehot:
            oh_names = self.onehot_encoder.get_feature_names_out(self.valid_onehot)
            self.feature_names_cont.extend(list(oh_names))

        self.feature_names_cont.extend(self.valid_robust)
        self.feature_names_cont.extend(self.valid_minmax)
        self.feature_names_cont.extend(self.valid_bool)

        # 3. Master Global List
        self.feature_names_all = (self.feature_names_cat + self.feature_names_cont)

    def get_feature_names(self, feat_type="all"):
        if feat_type == "all":
            return self.feature_names_all.copy()
        elif feat_type == "cat":
            return self.feature_names_cat.copy()
        elif feat_type == "cont":
            return self.feature_names_cont.copy()
        raise ValueError(f"Invalid feat_type '{feat_type}'. Choose 'all', 'cat', or 'cont'.")

    def get_col_idx(self, col_name, rel="all"):
        target_list = getattr(self, f"feature_names_{rel}", None)

        if target_list is None:
            raise ValueError(f"Invalid mode '{rel}'. Must be 'all', 'cat', or 'cont'.")

        if col_name in target_list:
            return target_list.index(col_name)

        raise ValueError(f"Column '{col_name}' not found in feature_names_{rel}!")

    # ==========================================
    # 4. UTILITY & DECODING METHODS
    # ==========================================
    def inverse_transform_cat(self, col_name, encoded_val):
        if col_name not in self.label_encoders:
            return encoded_val

        rev_map = {v: k for k, v in self.label_encoders[col_name].items()}
        rev_map[0] = "<UNK>"

        if isinstance(encoded_val, (list, np.ndarray, torch.Tensor)):
            if isinstance(encoded_val, torch.Tensor):
                encoded_val = encoded_val.cpu().numpy()
            return [rev_map.get(int(v), "<UNK>") for v in np.ravel(encoded_val)]

        return rev_map.get(int(encoded_val), "<UNK>")

    def inverse_transform_cont(self, scaled_values, col_name):
        arr = np.array(scaled_values, dtype=np.float32)

        if col_name in self.valid_robust:
            col_idx = self.valid_robust.index(col_name)
            center = self.robust_scaler.center_[col_idx]
            scale = self.robust_scaler.scale_[col_idx]
            return arr * scale + center
        elif col_name in self.valid_minmax:
            col_idx = self.valid_minmax.index(col_name)
            min_val = self.minmax_scaler.data_min_[col_idx]
            max_val = self.minmax_scaler.data_max_[col_idx]
            return arr * (max_val - min_val) + min_val
        else:
            return arr
