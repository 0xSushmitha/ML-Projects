def train_regression(df):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error
    import matplotlib.pyplot as plt

    X = df[['hour', 'dayofweek']]
    y = df['Global_active_power']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("R²:", r2_score(y_test, y_pred))
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Linear Regression - Global Active Power")
    plt.grid(True)
    plt.show()
