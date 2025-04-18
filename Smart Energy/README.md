# Smart Energy ML Project

A machine learning project that analyzes household electricity consumption and applies core ML techniques like EDA, PCA, clustering, regression, classification, and anomaly detection using real-world time-series data.

---

## Dataset
**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)  
**Name:** Individual Household Electric Power Consumption  
**Format:** `.txt`, over 2 million records from 2006 to 2010.

> **Note**: Please download `household_power_consumption.txt` manually and place it in the root project directory.

---

##  Project Files
```
SmartEnergyML.ipynb                   # Jupyter notebook with EDA + ML workflows
main.py                               # Orchestrator script to run all ML modules
anomaly_detection_energy.py           # Detect unusual energy usage using Isolation Forest
classification_appliance_status.py    # Classify appliance ON/OFF state with Random Forest
clustering_energy_patterns.py         # Cluster daily energy usage patterns
dimensionality_reduction_pca.py       # Apply PCA for dimensionality reduction
regression_energy_prediction.py       # Predict power consumption using linear regression
household_power_consumption.txt       # Input dataset (manual download from UCI)
README.md                             # Project overview and instructions
requirements.txt                      # Python dependencies

```

**Note:** Each ML task has been modularized into its own Python script to keep the codebase clean, scalable, and easy to maintain. This decentralized structure allows for focused development, testing, and reuse of individual components such as regression, classification, clustering, and anomaly detection.

---

## How to Run
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Launch the notebook:
```bash
jupyter notebook SmartEnergyML.ipynb
```

3. Or run the main.py :
```bash
python main.py
```

---

## Machine Learning Techniques Used
| Section         | Task                         | ML Concepts                                          |
|-----------------|------------------------------|------------------------------------------------------|
| EDA             | Visualizing usage            | Time-series plots, feature engineering               |
| PCA             | Dimensionality reduction     | Variance analysis                                    |
| Clustering      | Pattern segmentation         | KMeans, PCA, behavioral grouping                     |
| Regression      | Power prediction             | Random Forest Regressor, RMSE, R², Cross-validation  |
| Classification  | ON/OFF appliance state       | Random Forest Classifier, GridSearchCV, F1 Score     |
| Anomalies       | Outlier detection            | Isolation Forest, rare event spotting                |
| Forecasting     | Next-day energy consumption  | Prophet model, trend & seasonality modeling          |


---

## Key Results

- **Random Forest Regression**: RMSE ≈ 0.76 kW (↓ from baseline 0.836), R² ≈ 0.267  
- **Classification (Appliance ON/OFF)**: Accuracy = 76%, F1 Score (ON) = 0.48  
- **PCA**: Top 3 components explain ~55% of variance in daily usage patterns  
- **Clustering**: 3 distinct daily consumption behavior groups identified  
- **Anomaly Detection**: 336 anomalies (~1%) flagged as outliers in power usage  
- **Forecasting (Prophet)**: 24-hour ahead forecast with daily seasonality captured


---
##  Author

**Name**: Sushmitha Bakthavatchalam

---
