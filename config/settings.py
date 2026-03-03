class Settings:
    # Clustering parameters
    K_MIN = 2
    K_MAX = 10
    RANDOM_STATE = 42
    N_INIT = 20
    MAX_ITER = 300
    
    # Feature selection thresholds
    ID_THRESHOLD = 0.95        # Drop columns with >95% unique values
    CORRELATION_THRESHOLD = 0.95 
    VARIANCE_THRESHOLD = 0.01    
    
    # Preprocessing
    OUTLIER_METHOD = 'IQR'
    IQR_MULTIPLIER = 1.5
    IMPUTATION_STRATEGY = 'median'
    
    # Business Keywords for automatic detection
    KEYWORDS = {
        'monetary': ['amount', 'spend', 'income', 'revenue', 'sales', 'bill', 'price', 'total'],
        'frequency': ['count', 'freq', 'times', 'visit', 'order', 'transaction'],
        'temporal': ['recency', 'days', 'month', 'year', 'age', 'tenure', 'date', 'time'],
        'behavioral': ['score', 'rating', 'index', 'point', 'rank']
    }
