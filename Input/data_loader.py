import torch
from torch.utils.data import Dataset

from util.ts2image import combine_matrices_with_ts_element


class PorkDataset(Dataset):
    def __init__(self, dates, prices, sample_size):
        # 读取CSV文件并处理数据
        self.dates, self.prices = dates, prices
        self.sample_size = sample_size

    def __len__(self):
        # 数据集大小
        return len(self.prices) - self.sample_size

    def __getitem__(self, idx):
        # 获取单个样本的价格数据
        all_x = self.prices[idx:idx + self.sample_size]
        y = self.prices[idx + self.sample_size]
        y = torch.tensor(y).float()
        # 将价格数据转换为图像数据
        all_x = combine_matrices_with_ts_element(all_x.reshape(1, -1)).float()
        # 切分Encoder输入和Decoder输入
        src_x = all_x[:self.sample_size, :]
        # 创建一个全零向量
        _, feature_size = src_x.shape
        # 创建一个形状为 [1, feature_size] 的全零向量
        zero_vector_dynamic = torch.zeros(1, feature_size)
        # 将全零向量添加到原始张量的末尾形成 tgt_x
        tgt_x = torch.cat((all_x[1:self.sample_size, :], zero_vector_dynamic), dim=0)
        # 获取对应的日期数据
        all_x_mark = self.dates[idx:idx + self.sample_size + 1]
        all_x_mark = torch.tensor(all_x_mark).float()
        # 日期数据切分Encoder输入和Decoder输入
        src_x_mark = all_x_mark[:self.sample_size, :]
        tgt_x_mark = all_x_mark[1:self.sample_size + 1, :]

        return src_x, src_x_mark, tgt_x, tgt_x_mark, y


# 数据集类型(train、val、test)，批次大小，数据文件路径，线程数
def load_data(dates, prices, sample_size, batch_size, num_workers):
    dataset = PorkDataset(dates, prices, sample_size)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    return dataloader

# 示例使用
# file_path = '../data/' + 'Input_data.csv'  # 替换为您的文件路径
# dataloader = load_data(csv_file_path=file_path, sample_size=16, batch_size=32, num_workers=0)
# for i, (src_x, src_x_mark, tgt_x, tgt_x_mark, y) in enumerate(dataloader):
#     print("批次", i)
#     print("src_x形状:", src_x.shape, src_x.type())
#     print("src_x_mark形状:", src_x_mark.shape, src_x_mark.type())
#     print("tgt_x形状:", tgt_x.shape, tgt_x.type())
#     print("tgt_x_mark形状:", tgt_x_mark.shape, tgt_x_mark.type())
#     print("y形状:", y.shape)
#     # 创建模型实例
#     input_dim = 49  # 输入维度
#     d_model = 512  # 嵌入维度
#     nhead = 8  # 多头注意力头数
#     num_encoder_layers = 6  # 编码器层数
#     num_decoder_layers = 6  # 解码器层数
#     dim_feedforward = 2048  # 前馈网络维度
#     max_seq_length = 100  # 最大序列长度
#
#     model = TransformerModel(input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward)
#
#     tgt_len = 16  # 目标序列长度
#     tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len) * float('-inf'), diagonal=1)
#     # tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len), diagonal=0) == 0
#     output = model(src_x, src_x_mark, tgt_x, tgt_x_mark, tgt_mask=tgt_mask)
#     print(output.shape)
#     break
