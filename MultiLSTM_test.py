import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
# import os


np.random.seed(42)

T = 1000
t = np.arange(T)

# 高频：周期20；低频：周期100；再加上随机噪声
series = (
    0.5 * np.sin(2 * np.pi * t / 20)
  + 1.0 * np.sin(2 * np.pi * t / 100)
  + 0.1 * np.random.randn(T)
)

scale_factors = [1, 5]
multi_scale_data = []
for k in scale_factors:
    multi_scale_data.append(series[::k])



class MultiScaleDataset(Dataset):
    def __init__(self, scales, window_size, horizon):
        # 以最短的尺度为参考（通常是高频数据）
        min_len = min(len(s) for s in scales)
        max_i = min_len - window_size - horizon + 1

        self.X, self.y = [], []
        for i in range(max_i):
            x_win = []
            for s in range(len(scales)):
                step = scale_factors[s]
                start = i * step
                end = start + window_size * step
                if end > len(scales[s]) * step:
                    continue
                x = scales[s][start//step:end//step]
                x_win.append(x)
            if len(x_win) == len(scales):  # 确保每个尺度都有数据
                self.X.append(np.stack(x_win, axis=0))
                self.y.append(scales[0][i+window_size:i+window_size+horizon])

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

window_size, horizon = 50, 10
dataset = MultiScaleDataset(multi_scale_data, window_size, horizon)

# 时间顺序划分
n = len(dataset)
train_end = int(n * 0.7)
val_end   = int(n * 0.85)
train_set = torch.utils.data.Subset(dataset, range(0, train_end))
val_set   = torch.utils.data.Subset(dataset, range(train_end, val_end))
test_set  = torch.utils.data.Subset(dataset, range(val_end, n))

train_loader = DataLoader(train_set, batch_size=32, shuffle=False)
val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False)
test_loader  = DataLoader(test_set,  batch_size=32, shuffle=False)



class MultiScaleLSTM(nn.Module):
    def __init__(self, scales, window_size, hidden_size, horizon):
        super().__init__()
        self.scales = scales
        self.lstm_cells = nn.ModuleList([
            nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
            for _ in scales
        ])
        self.fc = nn.Linear(hidden_size * len(scales), horizon)

    def forward(self, x):
        # x: [batch, scales, window_size]
        outs = []
        for i, lstm in enumerate(self.lstm_cells):
            xi = x[:, i, :].unsqueeze(-1)     # [batch, window, 1]
            out, _ = lstm(xi)                 # out: [batch, window, hidden]
            outs.append(out[:, -1, :])        # 取最后时刻隐藏层
        h_concat = torch.cat(outs, dim=1)     # [batch, hidden * scales]
        return self.fc(h_concat)             # [batch, horizon]


device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = MultiScaleLSTM(scale_factors, window_size, 64, horizon).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_losses, val_losses = [], []

for epoch in range(1, 51):
    # ---- 训练 ----
    model.train()
    running_loss = 0.0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(Xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))

    # ---- 验证 ----
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            val_loss += criterion(model(Xb), yb).item()
    val_losses.append(val_loss / len(val_loader))

plt.figure(figsize=(12,4))
plt.plot(t, series, color='blue')
plt.title('Original Time Series')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()

plt.figure(figsize=(12,4))
plt.plot(train_losses, color='orange', label='Train Loss')
plt.plot(val_losses,   color='purple', label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.show()

# 取测试中的第一个 batch 的第一个样本
X_test, y_test = next(iter(test_loader))
model.eval()
with torch.no_grad():
    pred_test = model(X_test.to(device)).cpu().numpy()

plt.figure(figsize=(12,4))
plt.plot(y_test[0], color='blue',   label='Actual')
plt.plot(pred_test[0], color='red', linestyle='--', label='Predicted')
plt.title('Forecast vs Actual (Test Sample)')
plt.xlabel('Horizon Step')
plt.ylabel('Value')
plt.legend()
plt.show()