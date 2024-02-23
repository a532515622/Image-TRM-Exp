import torch
from torch import nn, optim
from tqdm import tqdm

from Input.data_loader import PorkDataset, load_data
from models.transformerModel import TransformerModel
def main():

    # 模型初始化
    file_path = './data/' + 'Input_data.csv'  # 替换为您的文件路径
    sample_size = 16  # 样本长度
    batch_size = 32  # 批次大小
    num_workers = 6 # 线程数
    input_dim = 49  # 输入数据的特征维数
    d_model = 512  # 模型的维度
    nhead = 8  # 多头注意力机制的头数
    num_encoder_layers = 6  # 编码器层数
    num_decoder_layers = 6  # 解码器层数
    dim_feedforward = 2048  # 前馈网络的维度

    # 数据加载
    dataset = PorkDataset(file_path, sample_size=sample_size)
    dataloader = load_data(csv_file_path=file_path, sample_size=sample_size, batch_size=batch_size, num_workers=num_workers)

    model = TransformerModel(input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward)
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 例如使用均方误差损失函数，适用于回归问题
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        # 将tqdm包装器应用于dataloader
        with tqdm(enumerate(dataloader), total=len(dataloader)) as t:
            for batch_idx, (src_x, src_x_mark, tgt_x, tgt_x_mark, y) in t:
                src_x, src_x_mark, tgt_x, tgt_x_mark, y = src_x.cuda(), src_x_mark.cuda(), tgt_x.cuda(), tgt_x_mark.cuda(), y.cuda()
            # 生成目标序列的掩码
            tgt_len = sample_size  # 目标序列长度
            tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len) * float('-inf'), diagonal=1)
            # 前向传播
            outputs = model(src_x, src_x_mark, tgt_x, tgt_x_mark, tgt_mask=tgt_mask)
            # print(outputs[:, 0, :].size())
            loss = criterion(outputs[:, 0, :].squeeze(1), y)
            # print(loss)
            # 后向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 更新tqdm的描述信息
            t.set_description(f'Epoch {epoch + 1}/{num_epochs}')
            t.set_postfix(loss=loss.item())


    print("Training completed.")

if __name__ == '__main__':
    main()