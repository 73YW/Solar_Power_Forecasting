import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
df_gen = pd.read_csv('Plant_1_Generation_Data.csv')
df_weather = pd.read_csv('Plant_1_Weather_Sensor_Data.csv')
df_gen['DATE_TIME'] = pd.to_datetime(df_gen['DATE_TIME'])
df_weather['DATE_TIME'] = pd.to_datetime(df_weather['DATE_TIME'])
df_merged = pd.merge(df_gen, df_weather, on=['DATE_TIME'], how='inner').dropna()
df_daytime = df_merged[df_merged['IRRADIATION'] > 0].copy()
features = ['IRRADIATION', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'AC_POWER']
data = df_daytime[features].values
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
def create_dataset(dataset, time_step=10):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), :-1])
        Y.append(dataset[i + time_step, -1])
    return np.array(X), np.array(Y)
time_step = 10
X_lstm, y_lstm = create_dataset(data_scaled, time_step)
X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_lstm, test_size=0.2, shuffle=False)
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)
print("数据准备完毕！训练集大小:", X_train_tensor.shape)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out
model = LSTMModel(input_size=3, hidden_size=32, output_size=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print("开始训练 LSTM 模型...")
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 5 == 0:
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')
print("\n训练完成，正在计算误差指标...")
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).numpy()
dummy_pred = np.zeros((len(predictions), 4))
dummy_pred[:, 3] = predictions.flatten()
predictions_inv = scaler.inverse_transform(dummy_pred)[:, 3]
dummy_true = np.zeros((len(y_test), 4))
dummy_true[:, 3] = y_test.flatten()
true_inv = scaler.inverse_transform(dummy_true)[:, 3]
mae_lstm = mean_absolute_error(true_inv, predictions_inv)
rmse_lstm = np.sqrt(mean_squared_error(true_inv, predictions_inv))
print(f"\n【LSTM 模型】MAE = {mae_lstm:.2f}, RMSE = {rmse_lstm:.2f}")
