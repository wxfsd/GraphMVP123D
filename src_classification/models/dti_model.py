import torch
from torch import nn
from torch_geometric.nn import global_mean_pool
import dgl
import networkx as nx
from torch_geometric.utils import to_networkx
import scipy.sparse as sp
import scipy.sparse.linalg as linalg
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class CrossModalAttentionFusion_mask(nn.Module):
    def __init__(self, mol_dim, prot_dim, hidden_dim=512, out_dim=300, num_heads=4, mask_ratio=0.2):
        super().__init__()
        self.query_proj = nn.Linear(mol_dim, hidden_dim)
        self.key_proj = nn.Linear(prot_dim, hidden_dim)
        self.value_proj = nn.Linear(prot_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.mask_ratio = mask_ratio

    def forward(self, mol_repr, prot_repr, mask_type=None, seed=None, samplewise_random=False):
        mol_repr = mol_repr.unsqueeze(1)
        prot_repr = prot_repr.unsqueeze(1)

        q = self.query_proj(mol_repr)
        k = self.key_proj(prot_repr)
        v = self.value_proj(prot_repr)

        fused, attn_weights = self.attn(q, k, v)  # fused: [B, 1, H]
        fused = fused.squeeze(1)# [B, H]
        #print(q.shape, k.shape, v.shape, attn_weights.shape)

        if mask_type == "random":
            fused = self.random_mask(fused, seed=seed, samplewise=samplewise_random)

        elif mask_type == "attention":
            fused = self.attention_mask(fused, attn_weights, seed=seed)

        return self.out_proj(fused)

    def random_mask(self, fused, seed=None, samplewise=False):
        B, H = fused.size()
        num_mask = int(H * self.mask_ratio)
        mask = torch.ones_like(fused, dtype=torch.bool)

        if seed is not None:
            torch.manual_seed(seed)

        for i in range(B):
            if samplewise and seed is not None:
                torch.manual_seed(seed + i)
            idx = torch.randperm(H)[:num_mask]
            mask[i, idx] = False

        return fused * mask

    def attention_mask(self, fused, attn_weights, seed=None):
        B, H = fused.size()
        num_mask = int(H * self.mask_ratio)
        mask = torch.ones_like(fused, dtype=torch.bool)
        for i in range(B):
            idx = torch.topk(fused[i], k=num_mask, largest=False).indices
            mask[i, idx] = False
        return fused * mask


class CrossModalAttentionFusion(nn.Module):
    def __init__(self, mol_dim, prot_dim, hidden_dim=512, out_dim=300, num_heads=4):
        super().__init__()
        self.query_proj = nn.Linear(mol_dim, hidden_dim)
        self.key_proj = nn.Linear(prot_dim, hidden_dim)
        self.value_proj = nn.Linear(prot_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, mol_repr, prot_repr):
        # [B, D] → [B, 1, D] for attention
        mol_repr = mol_repr.unsqueeze(1)
        prot_repr = prot_repr.unsqueeze(1)

        q = self.query_proj(mol_repr)
        k = self.key_proj(prot_repr)
        v = self.value_proj(prot_repr)

        fused, _ = self.attn(q, k, v)
        fused = fused.squeeze(1)
        return self.out_proj(fused)
    
def compute_laplacian_positional_encoding(g, pos_enc_dim):
    """Compute Laplacian Positional Encoding using scipy eigen decomposition."""
    G = dgl.to_networkx(g).to_undirected()
    A = nx.to_scipy_sparse_array(G, format='csr')
    N = A.shape[0]
    D = sp.diags(np.array(A.sum(1)).flatten())
    L = D - A
    try:
        k = min(pos_enc_dim + 1, N - 1)
        if k <= 0:
            raise ValueError(f"Too few nodes ({N}) for LapPE.")
        _, eigvec = linalg.eigsh(L, k=k, which='SM')
        if eigvec.shape[1] < pos_enc_dim + 1:
            pad = np.zeros((N, pos_enc_dim + 1 - eigvec.shape[1]))
            eigvec = np.concatenate([eigvec, pad], axis=1)
        pos_enc = torch.from_numpy(eigvec[:, 1:pos_enc_dim + 1]).float()
        if torch.isnan(pos_enc).any() or torch.all(pos_enc == 0):
            raise ValueError("Invalid PE: contains NaNs or all zeros.")
    except Exception as e:
        print(f"[LapPE Warning] Graph of {N} nodes failed: {e}")
        pos_enc = torch.randn((N, pos_enc_dim))
    return pos_enc

def pyg_to_dgl_batch_with_indices(batch, pos_enc_dim=None):
    """
    Convert a PyG batch to DGLGraph list with optional Laplacian PE,
    and return the indices of successfully converted graphs.
    """
    data_list = batch.to_data_list()
    dgl_graphs = []
    valid_indices = []

    for i, data in enumerate(data_list):
        try:
            if data.edge_index.size(1) == 0 or data.edge_attr.size(0) == 0:
                print(f"[⚠️ Skip] Graph {i} has no edges.")
                continue

            data.x = data.x[:, 0].view(-1).long()
            data.edge_attr = data.edge_attr[:, 0].view(-1).long()

            G_nx = to_networkx(data, node_attrs=['x'], edge_attrs=[])
            edge_attrs = data.edge_attr.cpu().tolist()
            for idx, (u, v) in enumerate(G_nx.edges()):
                G_nx[u][v]['edge_attr'] = edge_attrs[idx] if idx < len(edge_attrs) else 0

            g = dgl.from_networkx(G_nx, node_attrs=['x'], edge_attrs=['edge_attr'])
            g.ndata['feat'] = g.ndata.pop('x')
            g.edata['feat'] = g.edata.pop('edge_attr')

            if pos_enc_dim is not None:
                pe = compute_laplacian_positional_encoding(g, pos_enc_dim)
                if torch.all(pe == 0):
                    print(f"[Zero PE Warning] Graph {i} has all-zero Laplacian PE.")
                    pe = torch.randn(g.number_of_nodes(), pos_enc_dim)
                g.ndata['lap_pos_enc'] = pe

            dgl_graphs.append(g)
            valid_indices.append(i)

        except Exception as e:
            print(f"[❌ Error] Graph {i} conversion failed: {e}")
            continue

    return dgl_graphs, valid_indices

class ProteinModel(nn.Module):
    def __init__(self, emb_dim=128, num_features=25, output_dim=128, n_filters=32, kernel_size=8):
        super(ProteinModel, self).__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.intermediate_dim = emb_dim - kernel_size + 1

        self.embedding = nn.Embedding(num_features+1, emb_dim)
        self.n_filters = n_filters
        self.conv1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=kernel_size)
        self.fc = nn.Linear(n_filters*self.intermediate_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.conv1(x)
        x = x.view(-1, self.n_filters*self.intermediate_dim)
        x = self.fc(x)
        return x


class MoleculeProteinModel(nn.Module):
    def __init__(self, molecule_model, protein_model, molecule_emb_dim, protein_emb_dim, output_dim=1, dropout=0.2):
        super(MoleculeProteinModel, self).__init__()
        self.fc1 = nn.Linear(molecule_emb_dim+protein_emb_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, output_dim)
        self.molecule_model = molecule_model
        self.protein_model = protein_model
        self.pool = global_mean_pool
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, molecule, protein):
        molecule_node_representation = self.molecule_model(molecule)
        molecule_representation = self.pool(molecule_node_representation, molecule.batch)
        protein_representation = self.protein_model(protein)

        x = torch.cat([molecule_representation, protein_representation], dim=1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out(x)

        return x

class MoleculeProteinModel123D(nn.Module):
    def __init__(self, model_1d, model_2d, model_3d, fusion_module,
                 protein_model, protein_emb_dim=300, dropout=0.2):
        super(MoleculeProteinModel123D, self).__init__()
        self.model_1d = model_1d
        self.model_2d = model_2d
        self.model_3d = model_3d
        self.fusion = fusion_module
        self.protein_model = protein_model
        self.pool = global_mean_pool
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.ln_fused = nn.LayerNorm(300)

        # ✅ 使用 CrossModalAttention 做 multimodal 融合
        self.cross_modal_attention = CrossModalAttentionFusion(300, 300)

        self.mlp = nn.Sequential(
            nn.Linear(300, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, molecule, protein):
        # === 1D 特征 ===
        smiles_input = molecule.smiles
        ptr = molecule.ptr
        smiles_list = []
        for i in range(len(ptr) - 1):
            start, end = ptr[i].item(), ptr[i + 1].item()
            smiles_str = ''.join([chr(c.item()) for c in smiles_input[start:end]])
            content = [self.model_1d.vocab.stoi.get(token, self.model_1d.vocab.unk_index) for token in smiles_str]

            # 截断到最大 token 长度（不超过 Transformer PE 限制）
            max_len = 512
            X = [self.model_1d.vocab.sos_index] + content[:max_len - 2] + [self.model_1d.vocab.eos_index]
            if len(X) < max_len:
                X += [self.model_1d.vocab.pad_index] * (max_len - len(X))
            smiles_list.append(torch.tensor(X, device=smiles_input.device))

        smiles_batch = torch.stack(smiles_list).permute(1, 0).contiguous().to(smiles_input.device)
        with torch.no_grad():
            feat_1d = self.model_1d._encode(smiles_batch)
        feat_1d = torch.tensor(feat_1d, dtype=torch.float32, device=smiles_input.device)

        # === 3D 特征 ===
        feat_3d = self.model_3d(molecule.x[:, 0], molecule.positions, molecule.batch)

        # === 2D 特征 ===
        dgl_graphs, valid_indices = pyg_to_dgl_batch_with_indices(molecule, pos_enc_dim=self.model_2d.pos_enc_dim)
        
        if len(dgl_graphs) == 0:
            raise RuntimeError("❌ All molecules were skipped due to 2D graph failure.")

        # === 同步过滤 1D、3D、protein 输入 ===
        feat_1d = feat_1d[valid_indices]
        #print('1d', feat_1d.shape)
        feat_3d = feat_3d[valid_indices]
        #print('3d', feat_3d.shape)
        protein = [protein[i] for i in valid_indices] if isinstance(protein, list) else protein[valid_indices]

        # === 构建 DGLBatch 并跑 2D ===
        bg = dgl.batch(dgl_graphs).to(smiles_input.device)
        h = bg.ndata['feat'].long()
        e = bg.edata['feat'].long()
        h_lap = bg.ndata['lap_pos_enc'] if 'lap_pos_enc' in bg.ndata else None
        feat_2d = self.model_2d(bg, h, e, h_lap, h_wl_pos_enc=None, return_embedding=True)
        #print('2d', feat_2d.shape)
        # === 融合多模态特征 ===
        fused = self.fusion(feat_1d, feat_2d, feat_3d)
        fused = self.ln_fused(fused)

        # === 蛋白质表示 ===
        protein_rep = self.protein_model(protein)

        # # === 使用 Cross Attention 融合 molecule + protein 表示 ===
        # fused_output = self.cross_modal_attention(fused, protein_rep)  # ✅ 正确调用 forward

        # === 使用 Cross Attention 融合 molecule + protein 表示，并添加残差连接 ===
        # attn_output = self.cross_modal_attention(fused, protein_rep, mask_type="attention", seed=42, samplewise_random=True)
        attn_output = self.cross_modal_attention(fused, protein_rep)
        # fused_output = attn_output + fused  # ✅ 残差连接，保持原始 fused 信息流

        # # 添加残差连接 + normalization ===
        # attn_output = self.cross_modal_attention(fused, protein_rep)
        # fused_output = self.ln_fused_out(attn_output + fused)  # residual + normalization

        # # === 拼接 + MLP 预测 ===
        # x = torch.cat([fused, protein_rep], dim=1)
        # x = self.dropout(self.relu(self.fc1(x)))
        # fused_output = self.dropout(self.relu(self.fc2(x)))

        return self.mlp(attn_output)
