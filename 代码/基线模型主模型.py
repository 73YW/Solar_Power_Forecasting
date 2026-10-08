import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
df_gen = pd.read_csv('Plant_1_Generation_Data.csv')
df_weather = pd.read_csv('Plant_1_Weather_Sensor_Data.csv')
df_gen['DATE_TIME'] = pd.to_datetime(df_gen['DATE_TIME'])
df_weather['DATE_TIME'] = pd.to_datetime(df_weather['DATE_TIME'])
df_merged = pd.merge(df_gen, df_weather, on=['DATE_TIME'], how='inner').dropna()
df_daytime = df_merged[df_merged['IRRADIATION'] > 0].copy()
features = ['IRRADIATION', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE']
X = df_daytime[features]
y = df_daytime['AC_POWER']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print(f"训练集大小: {X_train.shape[0]} 条, 测试集大小: {X_test.shape[0]} 条")
print("\n--- 训练基线模型：线性回归 ---")
baseline_model = LinearRegression()
baseline_model.fit(X_train, y_train)
y_pred_base = baseline_model.predict(X_test)
mae_base = mean_absolute_error(y_test, y_pred_base)
rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_base))
print(f"基线模型(线性回归) -> MAE: {mae_base:.2f}, RMSE: {rmse_base:.2f}")
print("\n--- 训练主模型：随机森林 ---")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print(f"主模型(随机森林) -> MAE: {mae_rf:.2f}, RMSE: {rmse_rf:.2f}")
plt.figure(figsize=(12, 5))
plt.plot(y_test.values[:100], label='真实值 (True)', color='black', linewidth=2)
plt.plot(y_pred_base[:100], label='线性回归预测 (Baseline)', color='blue', linestyle='--')
plt.plot(y_pred_rf[:100], label='随机森林预测 (Main)', color='orange', linestyle='-.')
plt.title('Model Prediction Comparison (First 100 points)')
plt.xlabel('Sample Points (Time Steps)')
plt.ylabel('AC Power (kW)')
plt.legend()
plt.tight_layout()
plt.show()