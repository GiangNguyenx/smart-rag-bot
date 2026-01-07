# src/ds_tools.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# 1. HÀM PHÂN CỤM (CLUSTERING)
def analyze_clusters(df: pd.DataFrame, features: list, n_clusters=3):
    """
    Thực hiện K-Means Clustering và vẽ biểu đồ Scatter.
    Args:
        df: DataFrame chứa dữ liệu
        features: List tên các cột dùng để phân cụm (VD: ['Age', 'Spending Score'])
        n_clusters: Số lượng cụm (mặc định 3)
    """
    # Xử lý dữ liệu: Loại bỏ dòng trống
    data = df[features].dropna()
    
    # Chạy thuật toán K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(data)
    
    # Vẽ biểu đồ
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=features[0], y=features[1], hue='Cluster', palette='viridis', s=100)
    plt.title(f'K-Means Clustering: {features[0]} vs {features[1]}')
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.grid(True)
    
    return f"Đã phân thành {n_clusters} cụm thành công. Cột 'Cluster' đã được thêm vào dữ liệu."

# 2. HÀM DỰ BÁO (FORECASTING)
def predict_trend(df: pd.DataFrame, target_col: str, months_ahead=3):
    """
    Dự báo xu hướng tương lai bằng Linear Regression đơn giản.
    Args:
        df: DataFrame
        target_col: Cột cần dự báo (VD: 'Sales')
        months_ahead: Số tháng muốn dự báo thêm
    """
    # Giả sử dữ liệu theo dòng thời gian, tạo biến X là index
    y = df[target_col].values
    X = np.array(range(len(y))).reshape(-1, 1)
    
    # Train Model
    model = LinearRegression()
    model.fit(X, y)
    
    # Dự báo tương lai
    last_index = len(y)
    future_X = np.array(range(last_index, last_index + months_ahead)).reshape(-1, 1)
    predictions = model.predict(future_X)
    
    # Vẽ biểu đồ nối dài
    plt.figure(figsize=(10, 6))
    # Vẽ dữ liệu cũ
    plt.plot(range(len(y)), y, label='Thực tế', marker='o')
    # Vẽ dự báo
    plt.plot(range(last_index, last_index + months_ahead), predictions, label='Dự báo AI', linestyle='--', color='red', marker='x')
    
    plt.title(f'Dự báo xu hướng {target_col} trong {months_ahead} kỳ tiếp theo')
    plt.legend()
    plt.grid(True)
    
    results_str = ", ".join([f"{p:.2f}" for p in predictions])
    return f"Dự báo cho {months_ahead} kỳ tới là: {results_str}"