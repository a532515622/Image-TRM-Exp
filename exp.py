import json
import os.path as osp
import sys

from sklearn.metrics import mean_absolute_error
# from pyts.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from Input.data_loader import load_data
from Input.read_csv import load_and_process_csv
from models.transformerModel import TransformerModel
from utils import *


class Exp:
    def __init__(self, args):
        super(Exp, self).__init__()
        self.args = args
        self.config = self.args.__dict__
        self.device = self._acquire_device()

        self._preparation()
        print_log(output_namespace(self.args))

        self._get_data()
        self._select_optimizer()
        self._select_criterion()

    # 获取设备
    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:{}'.format(0))
            print_log('Use GPU: {}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print_log('Use CPU')
        return device

    # 准备工作
    def _preparation(self):
        # seed
        set_seed(self.args.seed)
        # log and checkpoint
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)

        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename=osp.join(self.path, 'log.log'),
                            filemode='a', format='%(asctime)s - %(message)s')
        # prepare data
        self._get_data()
        # build the model
        self._build_model()

    # 构建模型
    def _build_model(self):
        args = self.args
        self.model = TransformerModel(args.input_dim, args.d_model, args.nhead, args.num_encoder_layers,
                                      args.num_decoder_layers, args.dim_feedforward).to(self.device)

    # 设置优化器
    def _select_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.lr)
        return self.optimizer

    # 设置损失函数
    def _select_criterion(self):
        self.criterion = torch.nn.MSELoss()

    # 保存模型
    def _save(self, name=''):
        torch.save(self.model.state_dict(), os.path.join(
            self.checkpoints_path, name + '.pth'))

    # 准备数据
    def _get_data(self):
        args = self.args
        config = self.args.__dict__
        # 加载csv文件
        train_time, train_datas = load_and_process_csv(args.train_file_path)
        # vali_time, vali_datas = load_and_process_csv(args.val_file_path)

        # 创建 MinMaxScaler 对象
        self.scaler = MinMaxScaler()
        # 使用 fit 方法计算训练集的最小值和最大值
        self.scaler.fit(train_datas.reshape(-1, 1))
        # 使用 transform 方法对训练集和验证集进行归一化
        train_datas = self.scaler.transform(train_datas.reshape(-1, 1)).reshape(-1)
        # vali_datas = self.scaler.transform(vali_datas.reshape(-1, 1)).reshape(-1)
        self.train_loader = load_data(dates=train_time, prices=train_datas, sample_size=args.sample_size,
                                      batch_size=args.batch_size, num_workers=args.num_workers)
        # self.vali_loader = load_data(dates=vali_time, prices=vali_datas, sample_size=args.sample_size,
        #                              batch_size=args.batch_size, num_workers=args.num_workers)

    def train(self, args):
        config = args.__dict__

        train_data = []
        # val_data = []

        for src_x, src_x_mark, tgt_x, tgt_x_mark, y in tqdm(self.train_loader, desc="Loading Train Data"):
            train_data.append((src_x, src_x_mark, tgt_x, tgt_x_mark, y))
        # for src_x, src_x_mark, tgt_x, tgt_x_mark, y in tqdm(self.vali_loader, desc="Loading Val Data"):
        #     val_data.append((src_x, src_x_mark, tgt_x, tgt_x_mark, y))

        for epoch in range(config['epochs']):
            train_loss = []
            self.model.train()
            train_pbar = tqdm(train_data, file=sys.stdout)

            for src_x, src_x_mark, tgt_x, tgt_x_mark, y in train_pbar:
                # 将数据转移到GPU上
                src_x, src_x_mark, tgt_x, tgt_x_mark, y = src_x.to(self.device), src_x_mark.to(self.device), tgt_x.to(
                    self.device), tgt_x_mark.to(self.device), y.to(self.device)
                # 生成目标序列的掩码
                tgt_len = config['sample_size']
                tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len) * float('-inf'), diagonal=1)
                # 前向传播
                output = self.model(src_x, src_x_mark, tgt_x, tgt_x_mark, tgt_mask=tgt_mask)
                # 计算损失
                loss = self.criterion(output[:, 0, :].squeeze(1), y)
                train_loss.append(loss.item())
                train_pbar.set_description('Training Epoch {}, train loss: {:.4f}'.format(epoch, loss.item()))
                # 后向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            train_loss = np.average(train_loss)

            # vali_loss = self.vali(val_data, config['sample_size'])
            # print_log('Epoch: {}, train loss: {:.4f}, vali loss: {:.4f}'.format(epoch, train_loss, vali_loss))
            if epoch + 1 % args.log_step == 0:
                self._save(name='epoch_{}'.format(epoch))
            print_log('Epoch: {}, train loss: {:.4f}'.format(epoch, train_loss))
        return self.model

    def vali(self, val_data, tgt_len):
        self.model.eval()
        preds_list, trues_list, total_loss = [], [], []
        vali_pbar = tqdm(val_data, file=sys.stdout)
        for src_x, src_x_mark, tgt_x, tgt_x_mark, true_y in vali_pbar:
            # 将数据转移到GPU上
            src_x, src_x_mark, tgt_x, tgt_x_mark, true_y = src_x.to(self.device), src_x_mark.to(self.device), tgt_x.to(
                self.device), tgt_x_mark.to(self.device), true_y.to(self.device)
            # 生成目标序列的掩码
            tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len) * float('-inf'), diagonal=1)
            # 预测
            pred_y = self.model(src_x, src_x_mark, tgt_x, tgt_x_mark)[:, 0, :].squeeze(1)
            pred_y = pred_y.to(self.device)
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                pred_y, true_y], [preds_list, trues_list]))
            loss = self.criterion(pred_y, true_y)
            vali_pbar.set_description('vali loss: {:.4f}'.format(loss.mean().item()))
            total_loss.append(loss.mean().item())

        total_loss = np.average(total_loss)
        preds = np.concatenate(preds_list, axis=0)
        trues = np.concatenate(trues_list, axis=0)
        preds = self.scaler.inverse_transform(preds.reshape(-1, 1)).reshape(-1)
        trues = self.scaler.inverse_transform(trues.reshape(-1, 1)).reshape(-1)
        print(preds.shape, trues.shape)
        # 测试,公式不一定对
        mae = mean_absolute_error(trues, preds)
        maep = mean_absolute_percentage_error(preds, trues)
        print_log('vali MSE:{:.4f}, MAE:{:.4f},MAEP:{:.4f}'.format(total_loss, mae, maep))
        self.model.train()
        return total_loss
