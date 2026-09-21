import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
import matplotlib.pyplot as plt

# ==========================================
# 1. 数据自动下载与预处理
# ==========================================
DATA_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
ZIP_FILE = "jena_climate_2009_2016.csv.zip"
CSV_FILE = "jena_climate_2009_2016.csv"

#下载数据并解压
def download_and_extract():
    if not os.path.exists(CSV_FILE):
        print("正在下载Jena气候数据集 (约13MB)...")
        urllib.request.urlretrieve(DATA_URL, ZIP_FILE)
        print("下载完成，正在解压...")
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall()
        print("解压完成！")
    else:
        print("数据集已存在，跳过下载。")

download_and_extract()

# 读取数据
df = pd.read_csv(CSV_FILE)

# 每隔一小时取样一次
df = df[5::6]

# 截取最近的 10000 个小时的数据
df = df.tail(10000).reset_index(drop=True)

# 提取有用的气象特征列
# 我们使用以下列：p (mbar), T (degC), Tpot (K), Tdew (degC), rh (%), VPmax (mbar), VPact (mbar), VPdef (mbar), sh (g/kg), H2OC (mmol/mol), rho (g/m**3)
features = ['p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)', 'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)', 'H2OC (mmol/mol)', 'rho (g/m**3)']
df_features = df[features]

# ==========================================
# 2. 构建滑动窗口与数据集划分
# ==========================================

#选择要预测的气象特征
TARGET_COL = 'T (degC)' 
target_idx = features.index(TARGET_COL)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_features.values)

def create_sequences(data, seq_length, target_index):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length][target_index] # 预测未来的那一个值
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LENGTH = 48 # 使用过去48小时预测未来1小时
X, y = create_sequences(scaled_data, SEQ_LENGTH, target_idx)

# 划分训练集 (70%), 验证集 (10%), 测试集 (20%)
train_split = int(0.7 * len(X))
val_split = int(0.8 * len(X))

X_train, y_train = torch.FloatTensor(X[:train_split]), torch.FloatTensor(y[:train_split])
X_val, y_val = torch.FloatTensor(X[train_split:val_split]), torch.FloatTensor(y[train_split:val_split])
X_test, y_test = torch.FloatTensor(X[val_split:]), torch.FloatTensor(y[val_split:])

BATCH_SIZE = 64
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 3. Transformer 模型定义 (Positional Encoding + Encoder)
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class WeatherTransformer(nn.Module):
    def __init__(self, num_features, d_model, nhead, num_layers, dropout=0.2):
        super(WeatherTransformer, self).__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, src):
        src = self.input_projection(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        # 取最后一个时间步的隐状态进行预测
        last_step = output[:, -1, :]
        return self.decoder(last_step).squeeze()

# ==========================================
# 4. 超参数设置与训练框架
# ==========================================
NUM_FEATURES = len(features)
D_MODEL = 128
NHEAD = 8
NUM_LAYERS = 3
EPOCHS = 50
INIT_LR = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WeatherTransformer(NUM_FEATURES, D_MODEL, NHEAD, NUM_LAYERS).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=INIT_LR, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

patience = 7
best_val_loss = float('inf')
patience_counter = 0

print(f"\n--- 开始训练模型预测 {TARGET_COL} ---")
print(f"使用设备: {device} | 训练样本数: {len(X_train)}")

# 训练与验证循环
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item() * batch_x.size(0)
    
    train_loss /= len(X_train)
    
    # 验证集评估
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item() * batch_x.size(0)
    val_loss /= len(X_val)

    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step()
    
    print(f'Epoch [{epoch+1:02d}/{EPOCHS}] | LR: {current_lr:.6f} | Train Loss(MSE): {train_loss:.4f} | Val Loss(MSE): {val_loss:.4f}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_weather_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"--> 触发早停！验证集连续 {patience} 轮未下降，提前结束训练。")
            break

# ==========================================
# 5. 测试集评估与结果反归一化
# ==========================================
if os.path.exists('best_weather_model.pth'):
    model.load_state_dict(torch.load('best_weather_model.pth'))

model.eval()
test_loss = 0
predictions = []
actuals = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        test_loss += loss.item() * batch_x.size(0)
        
        predictions.extend(outputs.cpu().numpy())
        actuals.extend(batch_y.cpu().numpy())

test_loss /= len(X_test)
print(f"\n--- 测试集评估完成 ---")
print(f"Test Loss (归一化后的MSE): {test_loss:.4f}")

# 将预测结果反归一化回真实的物理温度值
dummy_matrix_pred = np.zeros((len(predictions), NUM_FEATURES))
dummy_matrix_pred[:, target_idx] = predictions
real_predictions = scaler.inverse_transform(dummy_matrix_pred)[:, target_idx]

dummy_matrix_actual = np.zeros((len(actuals), NUM_FEATURES))
dummy_matrix_actual[:, target_idx] = actuals
real_actuals = scaler.inverse_transform(dummy_matrix_actual)[:, target_idx]

# 计算真实物理单位下的平均绝对误差(MAE)
mae = np.mean(np.abs(real_predictions - real_actuals))
print(f"真实温度预测的平均绝对误差 (MAE): {mae:.2f} °C")

# ==========================================
# 6. 结果可视化 (真实值 vs 预测值)
# ==========================================
print("\n正在生成预测结果对比图...")

PLOT_LENGTH = 200

plt.figure(figsize=(14, 6)) # 设置画板大小，宽14，高6

# 绘制真实的温度曲线（蓝色实线）
plt.plot(real_actuals[:PLOT_LENGTH], label='Actual Temperature (Ground Truth)', color='#1f77b4', alpha=0.8, linewidth=2)

# 绘制模型预测的温度曲线（红色虚线）
plt.plot(real_predictions[:PLOT_LENGTH], label='Predicted Temperature (Transformer)', color='#d62728', alpha=0.8, linestyle='--', linewidth=2)

# 设置图表的标题和坐标轴标签
plt.title(f'Transformer Model: Actual vs Predicted {TARGET_COL} (First {PLOT_LENGTH} Hours of Test Set)', fontsize=15, fontweight='bold')
plt.xlabel('Time Steps (Hours)', fontsize=12)
plt.ylabel(f'{TARGET_COL}', fontsize=12)

# 添加图例和网格线
plt.legend(loc='upper right', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

# 自动调整布局防重叠，并展示图像
plt.tight_layout()
plt.show()