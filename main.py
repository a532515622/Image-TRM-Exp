import argparse

from exp import Exp


def create_parser():
    parser = argparse.ArgumentParser()
    # 实验参数
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--ex_name', default='trm1-22', type=str)
    parser.add_argument('--seed', default=1, type=int)
    # 数据集参数
    parser.add_argument('--sample_size', default=32, type=int, help='样本大小')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--train_file_path', default='./data/' + 'all_data.csv', type=str, help='训练集csv文件路径')
    parser.add_argument('--val_file_path', default='./data/' + 'val_data.csv', type=str, help='验证集csv文件路径')
    parser.add_argument('--num_workers', default=6, type=int)
    # 模型参数
    parser.add_argument('--input_dim', default=97, type=int, help='输入数据的特征维数,这个数与sample_size有关,sample_size*3+1')
    parser.add_argument('--d_model', default=512, type=int, help='模型的维度')
    parser.add_argument('--nhead', default=8, type=int, help='多头注意力机制的头数')
    parser.add_argument('--num_encoder_layers', default=6, type=int, help='编码器层数')
    parser.add_argument('--num_decoder_layers', default=6, type=int, help='解码器层数')
    parser.add_argument('--dim_feedforward', default=2048, type=int, help='前馈网络的维度')
    # 训练参数
    parser.add_argument('--epochs', default=100, type=int, help='训练轮数')
    parser.add_argument('--log_step', default=100, type=int, help='日志步长')
    parser.add_argument('--lr', default=0.001, type=float, help='学习率')
    return parser


if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__

    exp = Exp(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>  start <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.train(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>> train end <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
