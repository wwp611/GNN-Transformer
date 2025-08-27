import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import random 
import pickle

from model import TemporalEGNN


# --------------------------------------------------------------------------
# 数据加载与划分
# --------------------------------------------------------------------------

def load_and_split_data(file_path: str, device: torch.device, train_ratio: float = 0.8, seed: int = 42):
    """加载数据并将其划分为训练集和验证集"""
    random.seed(seed)
    # weights_only=False 在较新的torch版本中可能需要设置为 allow_pickle=True
    # torch.load(file_path, map_location=device, pickle_module=pickle) with allow_pickle=True
    # 但通常直接加载即可
    loaded_sequences = torch.load(file_path, map_location='cpu', weights_only=False, pickle_module=pickle)

    processed_sequences = []
    for sequence in loaded_sequences:
        if isinstance(sequence, list) and len(sequence) > 1:
            # 确保序列中的每个元素都是Data对象
            cleaned_sequence = [item for item in sequence if isinstance(item, Data)]
            if len(cleaned_sequence) == len(sequence):
                 processed_sequences.append(cleaned_sequence)
            else:
                print("序列中包含非Data对象，已跳过。")
        else:
            print(f"跳过一个无效的序列（长度不足或类型错误）。")
    
    # 随机打乱序列
    random.shuffle(processed_sequences)
    
    # 根据比例划分
    split_index = int(len(processed_sequences) * train_ratio)
    train_sequences = processed_sequences[:split_index]
    val_sequences = processed_sequences[split_index:]
    
    print(f"成功加载 {len(processed_sequences)} 个总序列。")
    print(f"划分为 {len(train_sequences)} 个训练序列和 {len(val_sequences)} 个验证序列。")
    
    return train_sequences, val_sequences

# --------------------------------------------------------------------------
# 数据集类 (Dataset)
# --------------------------------------------------------------------------

class SequenceDataset(Dataset):
    """用于训练的 Dataset，返回完整的序列"""
    def __init__(self, sequences: List[List[Data]]):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[List[Data], torch.Tensor]:
        seq = self.sequences[idx]
        target_pos = seq[-1].pos # 目标是最后一个结构的位置
        return seq, target_pos


class ValidationDataset(Dataset):
    """
    【新增】用于验证的 Dataset。
    根据您的要求，它只返回初始结构 (S0) 和最终结构的原子位置。
    """
    def __init__(self, sequences: List[List[Data]]):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[Data, torch.Tensor]:
        seq = self.sequences[idx]
        initial_structure = seq[0]  # 只取第一个结构
        final_positions = seq[-1].pos # 目标还是最后一个结构的位置
        return initial_structure, final_positions

# --------------------------------------------------------------------------
# 自定义 collate_fn
# --------------------------------------------------------------------------

def collate_train_sequences(batch: List[Tuple[List[Data], torch.Tensor]]) -> Tuple[List[List[Data]], torch.Tensor]:
    """用于训练数据加载器的collate_fn，处理变长序列。"""
    sequences, targets_list = zip(*batch)
    # 将多个target tensor拼接成一个大的tensor
    targets_tensor = torch.cat(targets_list, dim=0)
    return list(sequences), targets_tensor

def collate_validation_batch(batch: List[Tuple[Data, torch.Tensor]]) -> Tuple[List[Data], torch.Tensor]:
    """
    【新增】用于验证数据加载器的collate_fn。
    它将一批 (初始结构, 最终位置) 数据整理好。
    """
    initial_structures, final_positions_list = zip(*batch)
    # 将多个target tensor拼接成一个大的tensor
    final_positions_tensor = torch.cat(final_positions_list, dim=0)
    return list(initial_structures), final_positions_tensor

# --------------------------------------------------------------------------
# 训练与评估函数
# --------------------------------------------------------------------------

def train_epoch(net, train_iter, loss_fn, optimizer, device, loss_scaling_factor):
    net.train()
    total_loss_per_epoch = 0.0
    total_atoms_in_epoch = 0

    for sequences, y_true_flat in train_iter:
        optimizer.zero_grad()

        # 将所有Data对象移动到指定设备
        sequences = [[g.to(device) for g in seq] for seq in sequences]
        y_true_flat = y_true_flat.to(device)

        y_hat_flat = net(sequences)
        
        loss = loss_fn(y_hat_flat, y_true_flat) * loss_scaling_factor
        loss.backward()
        optimizer.step()

        total_loss_per_epoch += loss.item() / loss_scaling_factor # 记录原始loss
        total_atoms_in_epoch += y_true_flat.shape[0]

    return total_loss_per_epoch / total_atoms_in_epoch


def evaluate_model(net, val_iter, loss_fn, device):
    """【修改】评估函数，以适应新的验证数据格式"""
    net.eval()
    total_loss = 0.0
    total_atoms = 0
    with torch.no_grad():
        for initial_structures_list, y_true_flat in val_iter:
            # `initial_structures_list` 是一个batch的初始结构Data对象列表
            y_true_flat = y_true_flat.to(device)

            # --- 关键修改 ---
            # 为每个初始结构构建模型期望的“伪序列”输入 [S0, S0]
            # 然后将这些伪序列组成一个batch
            pseudo_sequences_batch = []
            for s0 in initial_structures_list:
                s0_on_device = s0.to(device)
                pseudo_sequences_batch.append([s0_on_device, s0_on_device])
            
            # 使用模型进行“从初到末”的预测
            y_hat_flat = net(pseudo_sequences_batch)

            loss = loss_fn(y_hat_flat, y_true_flat)
            total_loss += loss.item()
            total_atoms += y_true_flat.shape[0]
            
    return total_loss / total_atoms


def train_model(train_iter, val_iter, device, hidden_channels, epochs, lr, loss_scaling_factor):
    net = TemporalEGNN(
        hidden_channels=hidden_channels,
        edge_channels=40,    # 这些参数最好能从数据中自动获取或作为参数传入
        n_atom_features=19,  # 以避免硬编码
        device=device
    ).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.L1Loss(reduction="sum") # 使用sum，然后在函数内外除以原子数

    for epoch in range(epochs):
        train_loss = train_epoch(net, train_iter, loss_fn, optimizer, device, loss_scaling_factor)
        val_loss = evaluate_model(net, val_iter, loss_fn, device)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}", flush=True)

    return net 


def save_predictions_vs_true(net, val_iter, device, output_file="pred_vs_true.log", max_samples=10):
    net.eval()
    with open(output_file, "w") as f:
        for batch_id, (initial_structures_list, y_true_flat) in enumerate(val_iter):
            y_true_flat = y_true_flat.to(device)

            # 构造伪序列输入
            pseudo_sequences_batch = []
            for s0 in initial_structures_list:
                s0_on_device = s0.to(device)
                pseudo_sequences_batch.append([s0_on_device, s0_on_device])

            with torch.no_grad():
                pred_pos = net(pseudo_sequences_batch)

            # 只打印前 max_samples 个原子
            num_atoms = min(pred_pos.size(0), max_samples)
            for i in range(num_atoms):
                error = torch.abs(pred_pos[i] - y_true_flat[i]).mean().item()
                f.write(
                    f"Atom {i:3d} | "
                    f"Pred: {pred_pos[i].cpu().numpy()} | "
                    f"True: {y_true_flat[i].cpu().numpy()} | "
                    f"MAE: {error:.6f}\n"
                )
            f.write("-"*80 + "\n")

# --------------------------------------------------------------------------
# 主函数
# --------------------------------------------------------------------------

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 1. 加载并划分数据 ---
    file_path = "all_sequences.pt"
    train_sequences, val_sequences = load_and_split_data(file_path, device, train_ratio=0.8)

    # --- 2. 创建不同的Dataset ---
    train_dataset = SequenceDataset(train_sequences)
    val_dataset = ValidationDataset(val_sequences) # 使用新的验证Dataset

    # --- 3. 定义超参数 ---
    BATCH_SIZE = 4
    LOSS_SCALE = 1
    LEARNING_RATE = 0.00005
    EPOCHS = 300
    HIDDEN_CHANNELS = 512

    # --- 4. 创建不同的DataLoader ---
    train_iter = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_train_sequences # 使用训练的collate
    )

    val_iter = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE, # 验证的batch size可以设置得大一些以加快速度
        shuffle=False,
        collate_fn=collate_validation_batch # 使用验证的collate
    )

    # --- 5. 开始训练 ---
    print("开始使用 model.py 中的 model 进行训练...")
 
    net = train_model(
        train_iter=train_iter,
        val_iter=val_iter,
        device=device,
        hidden_channels=HIDDEN_CHANNELS,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        loss_scaling_factor=LOSS_SCALE
    )

    save_predictions_vs_true(net, val_iter, device, output_file="pred_vs_true.log", max_samples=20)



if __name__ == "__main__":
    main()