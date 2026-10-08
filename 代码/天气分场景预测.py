import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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
daily_stats = df_daytime.groupby(df_daytime['DATE_TIME'].dt.date)['IRRADIATION'].agg(['std', 'mean'])
daily_stats['CV'] = daily_stats['std'] / daily_stats['mean']
threshold = daily_stats['CV'].median()
df_daytime['Weather_Type'] = df_daytime['DATE_TIME'].dt.date.map(
    lambda x: 'Sunny' if daily_stats.loc[x, 'CV'] < threshold else 'Cloudy'
)
base_features = ['IRRADIATION', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE']
results = {}
for weather in ['Sunny', 'Cloudy']:
    subset = df_daytime[df_daytime['Weather_Type'] == weather]
    X = subset[base_features]
    y = subset['AC_POWER']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results[weather] = {'MAE': mae, 'RMSE': rmse}
    print(f"【{weather} 场景】MAE = {mae:.2f}, RMSE = {rmse:.2f}")
plt.figure(figsize=(6, 4))
plt.bar(['Sunny', 'Cloudy'], [results['Sunny']['MAE'], results['Cloudy']['MAE']], color=['orange', 'gray'])
plt.title('MAE Comparison: Sunny vs Cloudy')
plt.ylabel('MAE (kW)')
plt.show()