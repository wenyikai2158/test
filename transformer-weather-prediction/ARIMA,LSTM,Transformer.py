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
import warnings
import sys
warnings.filterwarnings("ignore")

# ==========================================
# 1. 数据自动下载与预处理
# ==========================================
DATA_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
ZIP_FILE = "jena_climate_2009_2016.csv.zip"
CSV_FILE = "jena_climate_2009_2016.csv"

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

df = pd.read_csv(CSV_FILE)
df = df[5::6] # 每隔一小时取样一次
df = df.tail(10000).reset_index(drop=True) # 截取10000条以保障本地运行效率

features = ['p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)', 'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)', 'H2OC (mmol/mol)', 'rho (g/m**3)']
df_features = df[features]

# ==========================================
# 2. 构建多步长滑动窗口（预测未来24小时）
# ==========================================
TARGET_COL = 'T (degC)' 
target_idx = features.index(TARGET_COL)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_features.values)

# 核心参数变更：
SEQ_LENGTH = 96    # 输入历史窗口：96小时
PRED_LENGTH = 24   # 预测未来窗口：24小时

def create_multistep_sequences(data, seq_length, pred_length, target_index):
    xs = []
    ys = []
    for i in range(len(data) - seq_length - pred_length + 1):
        x = data[i:(i + seq_length)]
        # y 变成了一个长度为 24 的序列，代表未来连续 24 小时的真实温度
        y = data[(i + seq_length):(i + seq_length + pred_length), target_index]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

X, y = create_multistep_sequences(scaled_data, SEQ_LENGTH, PRED_LENGTH, target_idx)

# 数据集划分
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
# 3. 模型定义
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
        return x + self.pe[:, :x.size(1), :]

class MultiStepWeatherTransformer(nn.Module):
    def __init__(self, num_features, d_model, nhead, num_layers, pred_length, dropout=0.2):
        super(MultiStepWeatherTransformer, self).__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, pred_length)
        )

    def forward(self, src):
        src = self.input_projection(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        last_step = output[:, -1, :]
        return self.decoder(last_step) # 形状: [Batch, 24]

# ==========================================
# 4. Transformer 训练框架
# ==========================================
NUM_FEATURES = len(features)
D_MODEL = 128
NHEAD = 8
NUM_LAYERS = 3
EPOCHS = 40
INIT_LR = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformer_model = MultiStepWeatherTransformer(NUM_FEATURES, D_MODEL, NHEAD, NUM_LAYERS, PRED_LENGTH).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(transformer_model.parameters(), lr=INIT_LR, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

patience = 6
best_val_loss = float('inf')
patience_counter = 0

print(f"\n--- [1/3] 开始训练 Transformer 预测未来 {PRED_LENGTH} 小时曲线 ---")
for epoch in range(EPOCHS):
    transformer_model.train()
    train_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = transformer_model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        nn.utils.clip_grad_norm_(transformer_model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item() * batch_x.size(0)
    
    train_loss /= len(X_train)
    
    transformer_model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = transformer_model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item() * batch_x.size(0)
    val_loss /= len(X_val)

    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step()
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f'Epoch [{epoch+1:02d}/{EPOCHS}] | LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(transformer_model.state_dict(), 'best_multistep_tf.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"--> Transformer 触发早停。")
            break

# 提取 Transformer 最佳预测结果并反归一化
transformer_model.load_state_dict(torch.load('best_multistep_tf.pth'))
transformer_model.eval()
tf_preds, actuals = [], []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        outputs = transformer_model(batch_x)
        tf_preds.extend(outputs.cpu().numpy())
        actuals.extend(batch_y.numpy())

# 针对多步矩阵做 2D 反归一化映射
def inverse_transform_multistep(preds_2d, scaler, num_features, target_idx):
    flat_preds = np.array(preds_2d).flatten()
    dummy = np.zeros((len(flat_preds), num_features))
    dummy[:, target_idx] = flat_preds
    real_flat = scaler.inverse_transform(dummy)[:, target_idx]
    return real_flat.reshape(np.array(preds_2d).shape)

real_tf_preds = inverse_transform_multistep(tf_preds, scaler, NUM_FEATURES, target_idx)
real_actuals = inverse_transform_multistep(actuals, scaler, NUM_FEATURES, target_idx)

tf_mae = np.mean(np.abs(real_tf_preds - real_actuals))
print(f">> Transformer 24小时多步预测完成。全量评估 MAE: {tf_mae:.2f} °C")

# ==========================================
# 5. 对照组一：LSTM 多步长预测
# ==========================================
print("\n--- [2/3] 开始训练基线模型 LSTM ---")

class MultiStepLSTM(nn.Module):
    def __init__(self, num_features, hidden_dim, pred_length, num_layers=2):
        super(MultiStepLSTM, self).__init__()
        self.lstm = nn.LSTM(num_features, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, pred_length) # 同样输出 24 步

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]
        return self.fc(last_time_step)

lstm_model = MultiStepLSTM(num_features=NUM_FEATURES, hidden_dim=64, pred_length=PRED_LENGTH).to(device)
lstm_optimizer = torch.optim.AdamW(lstm_model.parameters(), lr=0.002)

for le in range(50): 
    lstm_model.train()
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        lstm_optimizer.zero_grad()
        out = lstm_model(bx)
        loss = criterion(out, by)
        loss.backward()
        lstm_optimizer.step()

lstm_model.eval()
lstm_preds = []
with torch.no_grad():
    for bx, _ in test_loader:
        out = lstm_model(bx.to(device))
        lstm_preds.extend(out.cpu().numpy())

real_lstm_preds = inverse_transform_multistep(lstm_preds, scaler, NUM_FEATURES, target_idx)
lstm_mae = np.mean(np.abs(real_lstm_preds - real_actuals))
print(f">> LSTM 24小时多步预测完成。 MAE: {lstm_mae:.2f} °C")

# ==========================================
# 6. 对照组二：ARIMA
# ==========================================
print("\n--- [3/3] 开始执行 ARIMA ---")

try:
    from statsmodels.tsa.arima.model import ARIMA
    
    ARIMA_EVAL_WINDOWS = 60 
    arima_window_maes = []
    
    # 抽取测试集前 60 个时间窗口的真实历史温度（物理单位）
    for w in range(ARIMA_EVAL_WINDOWS):
        start_pt = val_split + w
        history = list(scaler.inverse_transform(X[start_pt])[:, target_idx])
        
        window_preds = []
        for step in range(PRED_LENGTH):
            arima_m = ARIMA(history, order=(2, 1, 0))
            arima_f = arima_m.fit()
            next_val = arima_f.forecast()[0]
            window_preds.append(next_val)
            history.append(next_val)
        
        actual_window = real_actuals[w]
        window_mae = np.mean(np.abs(np.array(window_preds) - actual_window))
        arima_window_maes.append(window_mae)
        
        if (w + 1) % 15 == 0:
            print(f"-> ARIMA 多步推演进度: {((w+1)/ARIMA_EVAL_WINDOWS)*100:.0f}% [{w+1}/{ARIMA_EVAL_WINDOWS}]")
            sys.stdout.flush()
            
    arima_mae = np.mean(arima_window_maes)
    print(f">> ARIMA 24步外推完成。平均 MAE: {arima_mae:.2f} °C")
except ImportError:
    print("未检测到 statsmodels，自动跳过。")
    arima_mae = None

# ==========================================
# 7. 生成多步预测误差柱状图
# ==========================================
print("\n正在绘制 24小时多步长预测 误差对比图...")

fair_tf_mae = np.mean(np.abs(real_tf_preds[:ARIMA_EVAL_WINDOWS] - real_actuals[:ARIMA_EVAL_WINDOWS]))
fair_lstm_mae = np.mean(np.abs(real_lstm_preds[:ARIMA_EVAL_WINDOWS] - real_actuals[:ARIMA_EVAL_WINDOWS]))

models_list = ['Transformer (Ours)', 'LSTM (Baseline)']
mae_list = [fair_tf_mae, fair_lstm_mae]

if arima_mae is not None:
    models_list.insert(0, 'ARIMA (Traditional)')
    mae_list.insert(0, arima_mae)

plt.figure(figsize=(9, 5))
colors_pool = ['#7f7f7f', '#2ca02c', '#d62728'] # 灰、绿、红
colors = colors_pool[-len(models_list):]

bars = plt.bar(models_list, mae_list, color=colors, width=0.4)
for bar in bars:
    y_height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, y_height + 0.05, f'{y_height:.2f} °C', ha='center', va='bottom', fontweight='bold')

plt.title(f'Ultimate Challenge: 24-Hour Multi-step Forecasting MAE (Fair Comparison)', fontsize=12, fontweight='bold')
plt.ylabel('Mean Absolute Error (MAE) in °C', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()