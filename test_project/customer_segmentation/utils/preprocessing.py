import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from config.settings import Settings

def handle_missing_values(df):
    """Fills missing values with median (numeric) or mode (categorical)."""
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].dtype in ['int64', 'float64']:
            fill_val = df_clean[col].median() if Settings.IMPUTATION_STRATEGY == 'median' else df_clean[col].mean()
            df_clean[col] = df_clean[col].fillna(fill_val)
        else:
            if not df_clean[col].mode().empty:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    return df_clean

def remove_outliers(df, method="isolation_forest"):
    """
    Removes outliers using the user-selected method.
    Options: 'isolation_forest', 'iqr', or 'none'
    """
    numeric_df = df.select_dtypes(include=['number'])
    
    # Safety Check: If data is too small, skip outlier removal
    if numeric_df.empty or len(df) < 20:
        return df, pd.DataFrame()

    if method == "none":
        return df, pd.DataFrame()

    # --- METHOD 1: Isolation Forest (AI - Best for Skewed/Complex Data) ---
    if method == "isolation_forest":
        # contamination=0.05 means "Find the top 5% weirdest rows"
        iso = IsolationForest(contamination=0.05, random_state=42)
        yhat = iso.fit_predict(numeric_df)
        mask = yhat != -1
    
    # --- METHOD 2: IQR (Statistical - Best for Normal/Bell-Curve Data) ---
    elif method == "iqr":
        mask = pd.Series(True, index=df.index)
        for col in numeric_df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - (1.5 * iqr) # Standard strict 1.5 rule
            upper = q3 + (1.5 * iqr)
            col_mask = (df[col] >= lower) & (df[col] <= upper)
            mask = mask & col_mask
    
    else:
        return df, pd.DataFrame()

    df_clean = df[mask].copy()
    df_outliers = df[~mask].copy()
    
    return df_clean, df_outliers

def scale_data(df):
    """Normalizes data using RobustScaler."""
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(df)
    return pd.DataFrame(scaled_data, columns=df.columns), scaler