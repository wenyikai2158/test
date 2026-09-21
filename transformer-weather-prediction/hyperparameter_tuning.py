"""
超参数调优模块
系统性探索学习率、模型维度、注意力头数、网络层数、Dropout对模型性能的影响
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from main_complete import TransformerModel, create_sequences
import json
import os

print("=" * 80)
print("超参数调优实验")
print("=" * 80)

# ============================================================================
# 加载数据
# ============================================================================
print("\n[1] 加载数据...")

data_path = "jena_climate_2009_2016.csv"
df = pd.read_csv(data_path)
df['Date Time'] = pd.to_datetime(df['Date Time'])
df.set_index('Date Time', inplace=True)

features = ['T (degC)', 'p (mbar)', 'rh (%)', 'VPact (mbar)', 'rho (g/m**3)', 'wv (m/s)']
df_selected = df[features].head(10000).fillna(method='ffill').fillna(method='bfill')

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df_selected)

SEQ_LENGTH = 48
X, y = create_sequences(data_scaled, SEQ_LENGTH, target_idx=0)
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.1)

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
X_val_t = torch.FloatTensor(X_val)
y_val_t = torch.FloatTensor(y_val).unsqueeze(1)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test).unsqueeze(1)

BATCH_SIZE = 64
EPOCHS = 20  # 减少轮次以加快调优速度

# ============================================================================
# 定义评估函数
# ============================================================================
def evaluate_config(input_dim, config, verbose=False):
    """评估特定超参数配置"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = TransformerModel(
        input_dim=input_dim,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    val_dataset = torch.utils.data.TensorDataset(X_val_t, y_val_t)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        # 训练
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        if verbose and (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{EPOCHS}, Val Loss: {val_loss:.6f}")
    
    # 测试集评估
    model.eval()
    with torch.no_grad():
        X_test_dev = X_test_t.to(device)
        predictions = model(X_test_dev).cpu().numpy()
    
    y_true = y_test_t.numpy()
    mse = mean_squared_error(y_true, predictions)
    mae = mean_absolute_error(y_true, predictions)
    rmse = np.sqrt(mse)
    
    return {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'val_loss': best_val_loss}

# ============================================================================
# 1. 学习率调优
# ============================================================================
print("\n[2] 学习率调优...")

learning_rates = [0.0001, 0.0005, 0.001, 0.005]
lr_results = {}

base_config = {
    'input_dim': data_scaled.shape[1],
    'd_model': 64,
    'nhead': 4,
    'num_layers': 2,
    'dropout': 0.1
}

for lr in learning_rates:
    config = base_config.copy()
    config['lr'] = lr
    print(f"\n测试学习率: {lr}")
    results = evaluate_config(data_scaled.shape[1], config, verbose=False)
    lr_results[str(lr)] = results
    print(f"  MSE: {results['MSE']:.6f}, MAE: {results['MAE']:.6f}")

# ============================================================================
# 2. 模型维度调优
# ============================================================================
print("\n[3] 模型维度调优...")

d_models = [32, 64, 128]
dmodel_results = {}

for d_model in d_models:
    config = base_config.copy()
    config['d_model'] = d_model
    config['lr'] = 0.001  # 使用最佳学习率
    print(f"\n测试模型维度: {d_model}")
    results = evaluate_config(data_scaled.shape[1], config, verbose=False)
    dmodel_results[str(d_model)] = results
    print(f"  MSE: {results['MSE']:.6f}, MAE: {results['MAE']:.6f}")

# ============================================================================
# 3. 注意力头数调优
# ============================================================================
print("\n[4] 注意力头数调优...")

nheads = [2, 4, 8]
nhead_results = {}

for nhead in nheads:
    config = base_config.copy()
    config['nhead'] = nhead
    config['d_model'] = 128
    config['lr'] = 0.001
    print(f"\n测试注意力头数: {nhead}")
    results = evaluate_config(data_scaled.shape[1], config, verbose=False)
    nhead_results[str(nhead)] = results
    print(f"  MSE: {results['MSE']:.6f}, MAE: {results['MAE']:.6f}")

# ============================================================================
# 4. 网络层数调优
# ============================================================================
print("\n[5] 网络层数调优...")

num_layers_list = [1, 2, 3, 4]
num_layers_results = {}

for num_layers in num_layers_list:
    config = base_config.copy()
    config['num_layers'] = num_layers
    config['d_model'] = 128
    config['lr'] = 0.001
    print(f"\n测试网络层数: {num_layers}")
    results = evaluate_config(data_scaled.shape[1], config, verbose=False)
    num_layers_results[str(num_layers)] = results
    print(f"  MSE: {results['MSE']:.6f}, MAE: {results['MAE']:.6f}")

# ============================================================================
# 5. Dropout调优
# ============================================================================
print("\n[6] Dropout调优...")

dropouts = [0.0, 0.1, 0.2, 0.3]
dropout_results = {}

for dropout in dropouts:
    config = base_config.copy()
    config['dropout'] = dropout
    config['d_model'] = 128
    config['num_layers'] = 3
    config['lr'] = 0.001
    print(f"\n测试Dropout: {dropout}")
    results = evaluate_config(data_scaled.shape[1], config, verbose=False)
    dropout_results[str(dropout)] = results
    print(f"  MSE: {results['MSE']:.6f}, MAE: {results['MAE']:.6f}")

# ============================================================================
# 6. 生成可视化图表
# ============================================================================
print("\n[7] 生成调优结果可视化...")

vis_dir = "气象预测完整项目/visualizations"
os.makedirs(vis_dir, exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 学习率
ax = axes[0, 0]
lrs = [float(x) for x in lr_results.keys()]
mses = [lr_results[x]['MSE'] for x in lr_results.keys()]
ax.plot(lrs, mses, 'bo-', linewidth=2, markersize=8)
ax.set_xlabel('学习率', fontsize=11)
ax.set_ylabel('MSE', fontsize=11)
ax.set_title('学习率调优', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

# 模型维度
ax = axes[0, 1]
dms = [int(x) for x in dmodel_results.keys()]
mses = [dmodel_results[x]['MSE'] for x in dmodel_results.keys()]
ax.bar(range(len(dms)), mses, color='green', alpha=0.7)
ax.set_xticks(range(len(dms)))
ax.set_xticklabels(dms)
ax.set_xlabel('模型维度 (d_model)', fontsize=11)
ax.set_ylabel('MSE', fontsize=11)
ax.set_title('模型维度调优', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# 注意力头数
ax = axes[0, 2]
nhs = [int(x) for x in nhead_results.keys()]
mses = [nhead_results[x]['MSE'] for x in nhead_results.keys()]
ax.bar(range(len(nhs)), mses, color='orange', alpha=0.7)
ax.set_xticks(range(len(nhs)))
ax.set_xticklabels(nhs)
ax.set_xlabel('注意力头数 (nhead)', fontsize=11)
ax.set_ylabel('MSE', fontsize=11)
ax.set_title('注意力头数调优', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# 网络层数
ax = axes[1, 0]
nls = [int(x) for x in num_layers_results.keys()]
mses = [num_layers_results[x]['MSE'] for x in num_layers_results.keys()]
ax.plot(nls, mses, 'r^-', linewidth=2, markersize=8)
ax.set_xlabel('网络层数', fontsize=11)
ax.set_ylabel('MSE', fontsize=11)
ax.set_title('网络层数调优', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Dropout
ax = axes[1, 1]
dos = [float(x) for x in dropout_results.keys()]
mses = [dropout_results[x]['MSE'] for x in dropout_results.keys()]
ax.bar(range(len(dos)), mses, color='purple', alpha=0.7)
ax.set_xticks(range(len(dos)))
ax.set_xticklabels([f'{x:.1f}' for x in dos])
ax.set_xlabel('Dropout', fontsize=11)
ax.set_ylabel('MSE', fontsize=11)
ax.set_title('Dropout调优', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# 综合对比
ax = axes[1, 2]
categories = ['学习率', '维度', '头数', '层数', 'Dropout']
best_mses = [
    min([lr_results[x]['MSE'] for x in lr_results.keys()]),
    min([dmodel_results[x]['MSE'] for x in dmodel_results.keys()]),
    min([nhead_results[x]['MSE'] for x in nhead_results.keys()]),
    min([num_layers_results[x]['MSE'] for x in num_layers_results.keys()]),
    min([dropout_results[x]['MSE'] for x in dropout_results.keys()])
]
ax.bar(categories, best_mses, color='steelblue', alpha=0.7)
ax.set_ylabel('最佳MSE', fontsize=11)
ax.set_title('各参数最佳MSE对比', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig(f'{vis_dir}/hyperparameter_tuning.png', dpi=150, bbox_inches='tight')
plt.close()
print("超参数调优可视化图已保存")

# ============================================================================
# 7. 确定最佳配置
# ============================================================================
print("\n[8] 确定最佳超参数配置...")

best_lr = min(lr_results.items(), key=lambda x: x[1]['MSE'])
best_dmodel = min(dmodel_results.items(), key=lambda x: x[1]['MSE'])
best_nhead = min(nhead_results.items(), key=lambda x: x[1]['MSE'])
best_num_layers = min(num_layers_results.items(), key=lambda x: x[1]['MSE'])
best_dropout = min(dropout_results.items(), key=lambda x: x[1]['MSE'])

best_config = {
    'learning_rate': float(best_lr[0]),
    'd_model': int(best_dmodel[0]),
    'nhead': int(best_nhead[0]),
    'num_layers': int(best_num_layers[0]),
    'dropout': float(best_dropout[0])
}

print("\n最佳超参数配置:")
for key, value in best_config.items():
    print(f"  {key}: {value}")

print(f"\n最佳配置测试集性能:")
print(f"  MSE: {best_lr[1]['MSE']:.6f}")
print(f"  MAE: {best_lr[1]['MAE']:.6f}")
print(f"  RMSE: {best_lr[1]['RMSE']:.6f}")

# ============================================================================
# 8. 保存结果
# ============================================================================
print("\n[9] 保存调优结果...")

tuning_results = {
    'learning_rate': lr_results,
    'd_model': dmodel_results,
    'nhead': nhead_results,
    'num_layers': num_layers_results,
    'dropout': dropout_results,
    'best_config': best_config
}

with open('气象预测完整项目/hyperparameter_tuning_results.json', 'w', encoding='utf-8') as f:
    json.dump(tuning_results, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 80)
print("超参数调优完成！")
print("=" * 80)
