import pandas as pd
import numpy as np
from config.settings import Settings

def auto_feature_selection(df):
    features = df.select_dtypes(include=['number']).copy()
    to_drop = []
    for col in features.columns:
        if features[col].nunique() / len(features) > Settings.ID_THRESHOLD:
            to_drop.append(col)
    if not features.empty:
        var = features.var()
        to_drop.extend(var[var < Settings.VARIANCE_THRESHOLD].index.tolist())
    features = features.drop(columns=list(set(to_drop)))
    if len(features.columns) > 1:
        corr_matrix = features.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop_corr = [column for column in upper.columns if any(upper[column] > Settings.CORRELATION_THRESHOLD)]
        features = features.drop(columns=to_drop_corr)
    return features

def get_priority_features(columns):
    priorities = []
    for col in columns:
        col_lower = col.lower()
        for category, keys in Settings.KEYWORDS.items():
            if any(k in col_lower for k in keys):
                priorities.append(col)
                break
    priorities = list(dict.fromkeys(priorities))
    return priorities if priorities else list(columns[:3])
