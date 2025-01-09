import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Sample data (replace with your dataset)
data = pd.DataFrame({
    'Feature1': np.random.normal(10, 2, 100),  # Normal data
    'Feature2': np.random.normal(50, 15, 100),  # Noisy data
    'Feature3': np.random.uniform(20, 30, 100),  # Uniform data
})

# Function to compute noise metrics
def quantify_noise(data):
    metrics = {}
    for column in data.columns:
        feature = data[column].dropna()  # Handle missing values
        metrics[column] = {
            'Variance': np.var(feature),
            'Skewness': skew(feature),
            'Kurtosis': kurtosis(feature),
            'Signal-to-Noise Ratio (SNR)': np.mean(feature) / (np.std(feature) + 1e-10)
        }
    return pd.DataFrame(metrics).T

# Function to reduce noise
def reduce_noise(data):
    # 1. Imputation for Missing Values
    imputer = SimpleImputer(strategy='mean')  # Mean, Median, or Mode
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    
    # 2. Scaling for Consistency
    scaler = StandardScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data_imputed), columns=data.columns)
    
    # 3. PCA for Dimensionality Reduction
    pca = PCA(n_components=0.95)  # Keep 95% of variance
    data_pca = pd.DataFrame(pca.fit_transform(data_scaled))
    
    return data_pca

# Quantify Noise
df = pd.read_csv("data/config/SS-N.csv")
noise_metrics = quantify_noise(df)
print("Noise Metrics:\n", noise_metrics)

# Reduce Noise
cleaned_data = reduce_noise(data)
print("\nCleaned Data (First 5 Rows):\n", cleaned_data.head())