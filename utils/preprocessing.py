import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from config.settings import Settings

def handle_missing_values(df):
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

    numeric_df = df.select_dtypes(include=['number'])
    

    if numeric_df.empty or len(df) < 20:
        return df, pd.DataFrame()

    if method == "none":
        return df, pd.DataFrame()


    if method == "isolation_forest":
    
        iso = IsolationForest(contamination=0.05, random_state=42)
        yhat = iso.fit_predict(numeric_df)
        mask = yhat != -1
    
  
    elif method == "iqr":
        mask = pd.Series(True, index=df.index)
        for col in numeric_df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - (1.5 * iqr)
            upper = q3 + (1.5 * iqr)
            col_mask = (df[col] >= lower) & (df[col] <= upper)
            mask = mask & col_mask
    
    else:
        return df, pd.DataFrame()

    df_clean = df[mask].copy()
    df_outliers = df[~mask].copy()
    
    return df_clean, df_outliers

def scale_data(df):
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(df)
    return pd.DataFrame(scaled_data, columns=df.columns), scaler