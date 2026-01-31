import pandas as pd
import io

def load_data(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        return df
    except Exception as e:
        return None

def validate_data(df):
    if df is None:
        return False, "Could not read file."
    if len(df) < 10:
        return False, "Dataset must have at least 10 rows."
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) < 2:
        return False, "Dataset must have at least 2 numeric columns."
    return True, ""

def get_data_summary(df):
    summary = {
        'rows': df.shape[0],
        'cols': df.shape[1],
        'numeric_cols': list(df.select_dtypes(include=['number']).columns),
        'categorical_cols': list(df.select_dtypes(include=['object', 'category']).columns),
        'missing_values': df.isnull().sum().sum()
    }
    return summary
