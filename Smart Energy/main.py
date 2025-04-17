# main.py

from regression_energy_prediction import train_regression
from classification_appliance_status import train_classifier
from clustering_energy_patterns import cluster_days
from dimensionality_reduction_pca import run_pca
from anomaly_detection_energy import detect_anomalies

import pandas as pd

# Load and preprocess data (shared)
def load_common_data():
    df = pd.read_csv("household_power_consumption.txt", sep=';', low_memory=False, na_values='?', parse_dates=[[0,1]], infer_datetime_format=True)
    df.columns = df.columns.str.replace(' ', '_')
    df.rename(columns={'Date_Time': 'datetime'}, inplace=True)
    df = df.dropna()
    numeric_cols = df.columns.difference(['datetime'])
    df[numeric_cols] = df[numeric_cols].astype('float')
    df = df.set_index('datetime')
    df_hourly = df.resample('H').mean().dropna()
    df_hourly['hour'] = df_hourly.index.hour
    df_hourly['dayofweek'] = df_hourly.index.dayofweek
    return df_hourly

# Pivoted format for clustering/PCA
def get_daily_hour_matrix(df):
    df = df.copy()
    df['date'] = df.index.date
    df['hour'] = df.index.hour
    return df.pivot_table(values='Global_active_power', index='date', columns='hour').dropna()

if __name__ == '__main__':
    print("\n📊 Loading shared data...")
    df_hourly = load_common_data()

    print("\n🔢 Running regression...")
    train_regression(df_hourly)

    print("\n🔠 Running classification...")
    train_classifier(df_hourly)

    print("\n🔍 Running clustering...")
    df_cluster = get_daily_hour_matrix(df_hourly)
    cluster_days(df_cluster)

    print("\n📉 Running PCA...")
    run_pca(df_cluster)

    print("\n🚨 Running anomaly detection...")
    detect_anomalies(df_hourly)