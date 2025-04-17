def detect_anomalies(df):
    from sklearn.ensemble import IsolationForest
    import matplotlib.pyplot as plt

    df = df.copy()
    model = IsolationForest(contamination=0.01, random_state=42)
    df['anomaly'] = model.fit_predict(df[['Global_active_power']])

    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df['Global_active_power'], label='Power Consumption')
    plt.scatter(df[df['anomaly'] == -1].index,
                df[df['anomaly'] == -1]['Global_active_power'],
                color='red', label='Anomaly', s=10)
    plt.title("Anomaly Detection in Power Usage")
    plt.xlabel("Time")
    plt.ylabel("Global Active Power (kW)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
