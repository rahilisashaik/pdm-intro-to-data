from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor


def run_random_forest(df, features, split_size=0.2):
    X = df[features]
    y = df["Meme_Creativity_Score"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size, random_state=42)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    return r2, mae, mse

def run_linear_regression(df, features, split_size=0.2):
    X = df[features]
    y = df["Meme_Creativity_Score"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    return r2, mae, mse

