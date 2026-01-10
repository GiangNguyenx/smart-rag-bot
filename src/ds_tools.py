# src/ds_tools.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from langchain_core.tools import tool


_df = None

def set_dataframe(df: pd.DataFrame):
    global _df
    _df = df

@tool
def analyze_clusters(n_clusters: int = 3) -> str:
    """
    Perform K-means clustering on numerical columns of the dataset.
    
    Args:
        n_clusters: Number of clusters to create (default: 3)
    
    Returns:
        Analysis summary with cluster information
    """
    if _df is None:
        return "Error: No dataframe loaded"
    
    try:
        # Select only numeric columns
        numeric_cols = _df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            return "No numeric columns found for clustering"
        
        # Prepare data
        X = _df[numeric_cols].fillna(_df[numeric_cols].mean())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to dataframe
        _df['cluster'] = clusters
        
        # Generate summary
        summary = f"✅ Clustering complete with {n_clusters} clusters\n\n"
        summary += f"📊 Cluster distribution:\n{pd.Series(clusters).value_counts().sort_index()}\n\n"
        summary += f"📈 Columns used: {', '.join(numeric_cols)}"
        
        return summary
    
    except Exception as e:
        return f"Error during clustering: {str(e)}"


@tool
def predict_trend(column: str, periods: int = 5) -> str:
    """
    Predict future values for a numeric column using simple linear regression.
    
    Args:
        column: Name of the column to predict
        periods: Number of future periods to predict (default: 5)
    
    Returns:
        Prediction results as formatted string
    """
    if _df is None:
        return "Error: No dataframe loaded"
    
    try:
        if column not in _df.columns:
            return f"Column '{column}' not found. Available columns: {', '.join(_df.columns)}"
        
        if not pd.api.types.is_numeric_dtype(_df[column]):
            return f"Column '{column}' is not numeric"
        
        # Simple linear trend
        from sklearn.linear_model import LinearRegression
        
        y = _df[column].dropna().values
        X = np.arange(len(y)).reshape(-1, 1)
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict future
        future_X = np.arange(len(y), len(y) + periods).reshape(-1, 1)
        predictions = model.predict(future_X)
        
        result = f"📈 Trend prediction for '{column}':\n\n"
        result += f"Current trend: {'📈 increasing' if model.coef_[0] > 0 else '📉 decreasing'}\n"
        result += f"Slope: {model.coef_[0]:.4f}\n\n"
        result += "Future predictions:\n"
        for i, pred in enumerate(predictions, 1):
            result += f"  Period +{i}: {pred:.2f}\n"
        
        return result
    
    except Exception as e:
        return f"Error during prediction: {str(e)}"