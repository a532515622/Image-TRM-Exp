import torch
from torch.utils.data import Dataset

from util.ts2image import ts2combine


class Pork_Dataset(Dataset):
    def __init__(self, data, sample_size, hw=5):
        super(Pork_Dataset, self).__init__()
        self.data = data
        self.sample_size = sample_size
        self.hw = hw

    def __len__(self):
        return len(self.data) - self.sample_size

    def __getitem__(self, index):
        x = self.data[index:index + self.sample_size]
        y = self.data[index + self.sample_size]
        # 存储转换后的数据序列
        transformed_sequences = []
        for i in range(self.sample_size - self.hw + 1):
            transformed_data = ts2combine(x[i:i + self.hw].reshape(1, -1))
            transformed_sequences.append(transformed_data)
        data = torch.stack(transformed_sequences, dim=0)
        T, C, H, W = data.shape
        # 暂时未设置多时间步
        # data = data.reshape(1, C, H, W).float()
        labels = torch.tensor(y).float()
        return data.float(), labels


# 数据集类型(train、val、test)，批次大小，数据文件路径，线程数
def load_data(data, sample_size, batch_size, num_workers):
    dataset = Pork_Dataset(data, sample_size)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    return dataloader
