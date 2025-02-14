import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

calories = pd.read_csv('calories.csv')
exercise = pd.read_csv('exercise.csv')

df = exercise.merge(calories, on='User_ID')

df['Gender'] = df['Gender'].str.lower().map({'male': 1, 'female': 0})


X = df.drop(['User_ID', 'Calories'], axis=1)
y = df['Calories']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "LinearRegression": LinearRegression()
}

model_performance = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    model_performance[name] = {"MSE": mse, "R2 Score": r2}
    

    if name == "RandomForest":  
        joblib.dump(model, "best_model.pkl")


X_train.to_csv('X_train.csv', index=False)

performance_df = pd.DataFrame(model_performance).T
plt.figure(figsize=(8, 5))
sns.barplot(x=performance_df.index, y=performance_df["R2 Score"])
plt.title("Model Comparison (R2 Score)")
plt.ylabel("R2 Score")
plt.savefig("model_performance.png")

print("Model training completed and best model saved!")
print(performance_df)
