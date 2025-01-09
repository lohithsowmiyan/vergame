import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import LocalOutlierFactor


def analyze_dataset_noise(df):
    """
    Analyzes various aspects of noise in a dataset.
    
    Parameters:
    df (pandas.DataFrame): Input dataset
    
    Returns:
    dict: Dictionary containing various noise metrics
    """
    metrics = {}
    
    def missing_value_analysis():
        # Calculate percentage of missing values per column
        missing_percentages = (df.isnull().sum() / len(df)) * 100
        return {
            'total_missing_percentage': missing_percentages.mean(),
            'missing_by_column': missing_percentages.to_dict()
        }
    
    def outlier_analysis(numeric_cols):
        outliers = {}
        for col in numeric_cols:
            # Z-score method
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outliers[col] = {
                'z_score_outliers_percentage': (z_scores > 3).mean() * 100,
                'iqr_outliers_percentage': _iqr_outliers(df[col])
            }
        return outliers
    
    def _iqr_outliers(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        outlier_mask = ((series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR)))
        return (outlier_mask.sum() / len(series)) * 100
    
    def duplicate_analysis():
        # Calculate percentage of duplicate rows
        duplicate_percentage = (df.duplicated().sum() / len(df)) * 100
        return {
            'duplicate_rows_percentage': duplicate_percentage
        }
    
    def detect_anomalies(numeric_cols):
        if len(numeric_cols) == 0:
            return {}
        
        # Use Local Outlier Factor for anomaly detection
        lof = LocalOutlierFactor(contamination='auto')
        numeric_data = df[numeric_cols].fillna(df[numeric_cols].mean())
        anomaly_labels = lof.fit_predict(numeric_data)
        
        return {
            'anomaly_percentage': (anomaly_labels == -1).mean() * 100
        }
    
    def value_distribution_analysis(numeric_cols):
        distributions = {}
        for col in numeric_cols:
            # Calculate skewness and kurtosis
            distributions[col] = {
                'skewness': float(stats.skew(df[col].dropna())),
                'kurtosis': float(stats.kurtosis(df[col].dropna())),
                'variance': float(np.var(df[col].dropna()))
            }
        return distributions
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Compile all metrics
    metrics['missing_values'] = missing_value_analysis()
    metrics['outliers'] = outlier_analysis(numeric_cols)
    metrics['duplicates'] = duplicate_analysis()
    metrics['anomalies'] = detect_anomalies(numeric_cols)
    metrics['distributions'] = value_distribution_analysis(numeric_cols)
    
    return metrics

def print_noise_report(metrics):
    """
    Prints a formatted report of the noise metrics
    
    Parameters:
    metrics (dict): Output from analyze_dataset_noise function
    """
    print("=== Dataset Noise Analysis Report ===\n")
    
    print("1. Missing Values:")
    print(f"   Overall missing: {metrics['missing_values']['total_missing_percentage']:.2f}%")
    
    print("\n2. Duplicates:")
    print(f"   Duplicate rows: {metrics['duplicates']['duplicate_rows_percentage']:.2f}%")
    
    print("\n3. Outliers and Anomalies:")
    print(f"   Overall anomaly percentage: {metrics['anomalies'].get('anomaly_percentage', 'N/A')}%")
    
    print("\n4. Distribution Metrics by Column:")
    for col, stats in metrics['distributions'].items():
        print(f"\n   {col}:")
        print(f"   - Skewness: {stats['skewness']:.2f}")
        print(f"   - Kurtosis: {stats['kurtosis']:.2f}")
        print(f"   - Variance: {stats['variance']:.2f}")



df = pd.read_csv('data/config/SS-N.csv')

metrics = analyze_dataset_noise(df)

# Print formatted report
print_noise_report(metrics)