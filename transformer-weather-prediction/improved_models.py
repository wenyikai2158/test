"""
改进版Transformer模型
包含：稀疏注意力、局部注意力、CNN-Transformer、Graph Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

print("加载改进版Transformer模型...")

# ============================================================================
# 1. 稀疏注意力机制
# ============================================================================
class SparseAttention(nn.Module):
    """稀疏注意力机制"""
    def __init__(self, d_model, nhead, window_size=12, sparsity_pattern='local', dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.window_size = window_size
        self.sparsity_pattern = sparsity_pattern
        
        # 线性投影
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        # 线性投影
        q = self.q_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        
        # 创建稀疏注意力掩码
        mask = self._create_sparse_mask(seq_len, x.device)
        
        # 计算注意力分数
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # 注意力权重
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # 输出
        output = torch.matmul(attn_probs, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.out_proj(output)
        
        return output
    
    def _create_sparse_mask(self, seq_len, device):
        """创建稀疏注意力掩码"""
        if self.sparsity_pattern == 'local':
            # 局部窗口注意力
            mask = torch.zeros(seq_len, seq_len, device=device)
            for i in range(seq_len):
                start = max(0, i - self.window_size)
                end = min(seq_len, i + self.window_size + 1)
                mask[i, start:end] = 1
        elif self.sparsity_pattern == 'strided':
            # 跨步稀疏注意力
            mask = torch.zeros(seq_len, seq_len, device=device)
            for i in range(seq_len):
                mask[i, i::self.window_size] = 1
        else:  # 'random'
            mask = (torch.rand(seq_len, seq_len, device=device) > 0.7).float()
            mask = mask + mask.t()
            mask = (mask > 0).float()
            mask.fill_diagonal_(1)
        
        return mask

# ============================================================================
# 2. 局部+全局混合注意力
# ============================================================================
class LocalGlobalAttention(nn.Module):
    """局部+全局混合注意力"""
    def __init__(self, d_model, nhead, local_window=12, dropout=0.1):
        super().__init__()
        self.local_attention = SparseAttention(d_model, nhead, local_window, 'local', dropout)
        self.global_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # 可学习的融合权重
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x):
        # 局部注意力
        local_out = self.local_attention(x)
        
        # 全局注意力
        global_out, _ = self.global_attention(x, x, x)
        
        # 融合
        alpha = torch.sigmoid(self.fusion_weight)
        output = alpha * local_out + (1 - alpha) * global_out
        
        return output

# ============================================================================
# 3. CNN-Transformer混合模型
# ============================================================================
class CNNTransformerModel(nn.Module):
    """CNN-Transformer混合模型"""
    def __init__(self, num_features, d_model=128, nhead=4, num_layers=3, dropout=0.2):
        super().__init__()
        
        # CNN特征提取
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(num_features, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=7, padding=3),
            nn.BatchNorm1d(d_model),
            nn.GELU()
        )
        
        # 位置编码
        self.pos_encoder = nn.Parameter(torch.randn(1, 5000, d_model))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出解码器
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, src):
        batch_size, seq_len, features = src.size()
        
        # CNN提取局部特征 (batch, seq_len, features) -> (batch, features, seq_len)
        src_cnn = src.permute(0, 2, 1)
        cnn_out = self.cnn_layers(src_cnn)  # (batch, d_model, seq_len)
        cnn_out = cnn_out.permute(0, 2, 1)  # (batch, seq_len, d_model)
        
        # 添加位置编码
        src = cnn_out + self.pos_encoder[:, :seq_len, :]
        
        # Transformer编码
        output = self.transformer(src)
        
        # 取最后一个时间步
        last_step = output[:, -1, :]
        
        # 解码输出
        return self.decoder(last_step).squeeze()

# ============================================================================
# 4. Graph Transformer模型
# ============================================================================
class GraphAttentionLayer(nn.Module):
    """图注意力层"""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.W = nn.Linear(d_model, d_model)
        self.a = nn.Linear(2 * d_model, 1)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, h, adj_matrix):
        Wh = self.W(h)  # (batch, nodes, d_model)
        batch_size, num_nodes, d_model = Wh.size()
        
        # 计算注意力分数
        Wh_repeated = Wh.unsqueeze(2).repeat(1, 1, num_nodes, 1)
        Wh_repeated_transpose = Wh.unsqueeze(1).repeat(1, num_nodes, 1, 1)
        concat = torch.cat([Wh_repeated, Wh_repeated_transpose], dim=-1)
        
        e = self.leaky_relu(self.a(concat).squeeze(-1))
        
        # 应用邻接矩阵掩码
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj_matrix > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        
        # 聚合邻居信息
        h_prime = torch.matmul(attention, Wh)
        return h_prime

class GraphTransformerModel(nn.Module):
    """Graph Transformer模型"""
    def __init__(self, num_features, num_nodes=5, d_model=128, nhead=4, num_layers=3, dropout=0.2):
        super().__init__()
        self.num_nodes = num_nodes
        
        # 节点嵌入
        self.node_embedding = nn.Parameter(torch.randn(num_nodes, d_model))
        
        # 图注意力层
        self.graph_attention = GraphAttentionLayer(d_model, dropout)
        
        # 位置编码
        self.pos_encoder = nn.Parameter(torch.randn(1, 5000, d_model))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出解码器
        self.decoder = nn.Linear(d_model, 1)
        
    def forward(self, src, adj_matrix=None):
        batch_size, seq_len, features = src.size()
        
        # 初始化邻接矩阵（如果未提供）
        if adj_matrix is None:
            adj_matrix = torch.ones(self.num_nodes, self.num_nodes, device=src.device)
            adj_matrix = adj_matrix.fill_diagonal_(0)
        
        # 添加节点嵌入
        node_emb = self.node_embedding.unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, 1, 1)
        src_expanded = src.unsqueeze(2).repeat(1, 1, self.num_nodes, 1)
        src_with_node = src_expanded + node_emb
        
        # 图注意力
        graph_out = []
        for t in range(seq_len):
            step_out = self.graph_attention(src_with_node[:, t, :, :], adj_matrix)
            graph_out.append(step_out.mean(dim=1))  # 聚合所有节点
        graph_out = torch.stack(graph_out, dim=1)  # (batch, seq_len, d_model)
        
        # 添加位置编码
        src = graph_out + self.pos_encoder[:, :seq_len, :]
        
        # Transformer编码
        output = self.transformer(src)
        
        # 解码输出
        return self.decoder(output[:, -1, :]).squeeze()

# ============================================================================
# 5. 改进版Transformer（综合版）
# ============================================================================
class ImprovedTransformerModel(nn.Module):
    """改进版Transformer模型 - 集成多种改进"""
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=4, dropout=0.2):
        super().__init__()
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 相对位置编码
        self.relative_pos_encoding = RelativePositionalEncoding(d_model)
        
        # 标准Transformer + 稀疏注意力的混合
        self.sparse_attention = SparseAttention(d_model, nhead, window_size=12, sparsity_pattern='local', dropout=dropout)
        self.global_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(d_model)
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, src):
        # 输入投影
        src = self.input_proj(src)
        
        # 相对位置编码
        rel_pos_enc = self.relative_pos_encoding(src)
        
        # 混合注意力
        sparse_out = self.sparse_attention(src)
        global_out, _ = self.global_attention(src, src, src)
        
        # 融合注意力输出
        src = 0.5 * sparse_out + 0.5 * global_out + rel_pos_enc
        
        # Transformer编码
        output = self.transformer(src)
        
        # 层归一化
        output = self.layer_norm(output)
        
        # 取最后一个时间步
        output = output[:, -1, :]
        
        # 输出预测
        return self.fc(output)

class RelativePositionalEncoding(nn.Module):
    """相对位置编码"""
    def __init__(self, d_model, max_len=5000, num_buckets=32, max_distance=128):
        super().__init__()
        self.d_model = d_model
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_embeddings = nn.Parameter(torch.randn(num_buckets, d_model))
        
    def _relative_position_bucket(self, relative_position):
        """将相对位置映射到桶中"""
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        
        ret = 0
        n = -relative_position
        max_exact = num_buckets // 2
        is_small = n < max_exact
        
        val_if_large = max_exact + (torch.log(n.float() / max_exact) / 
                                   math.log(max_distance / max_exact) * 
                                   (num_buckets - max_exact)).long()
        val_if_large = torch.clamp(val_if_large, max_exact, num_buckets - 1)
        
        ret = torch.where(is_small, n, val_if_large)
        return ret
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        positions = torch.arange(seq_len, device=x.device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        buckets = self._relative_position_bucket(relative_positions)
        embeddings = self.relative_embeddings[buckets]
        return embeddings

# ============================================================================
# 模型工厂函数
# ============================================================================
def create_improved_model(model_type, input_dim, **kwargs):
    """创建改进版模型"""
    models = {
        'sparse': lambda: SparseAttention(kwargs.get('d_model', 128), kwargs.get('nhead', 4)),
        'local_global': lambda: LocalGlobalAttention(kwargs.get('d_model', 128), kwargs.get('nhead', 4)),
        'cnn_transformer': lambda: CNNTransformerModel(input_dim, **kwargs),
        'graph_transformer': lambda: GraphTransformerModel(input_dim, **kwargs),
        'improved_transformer': lambda: ImprovedTransformerModel(input_dim, **kwargs)
    }
    
    if model_type not in models:
        raise ValueError(f"未知的模型类型: {model_type}")
    
    return models[model_type]()

print("改进版Transformer模型加载完成！")
print("支持的模型类型:")
print("  - sparse: 稀疏注意力")
print("  - local_global: 局部+全局混合注意力")
print("  - cnn_transformer: CNN-Transformer混合模型")
print("  - graph_transformer: Graph Transformer模型")
print("  - improved_transformer: 综合改进版Transformer")
