import torch
from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot


# wind_data = np.loadtxt('data/wind_dataset.csv', delimiter=',', usecols=3, skiprows=1)
# print(wind_data.shape)

# 自行在method中选择生成的gaf
def ts2GAF(time_series, catalog='train', label=None, method='summation'):
    if method == 'summation':
        gaf = GramianAngularField(image_size=time_series.size, method='summation')
        type = 'GASF'
    else:
        gaf = GramianAngularField(image_size=time_series.size, method='difference')
        type = 'GADF'
    ts_gaf = gaf.fit_transform(time_series)
    GAF_tensor = torch.from_numpy(ts_gaf)
    return GAF_tensor


# print(ts2GAF(wind_data[0:32].reshape(1, -1), label='test01', method='difference').size())
# print(ts2GAF(wind_data[0:32].reshape(1, -1), label='test01', method='summation').size())


# 马尔科夫变迁场
def ts2MTF(time_series, catalog='train', label=None, strategy='normal'):
    mtf = MarkovTransitionField(image_size=time_series.size, n_bins=5, strategy=strategy)
    ts_MTF = mtf.fit_transform(time_series)
    MTF_tensor = torch.from_numpy(ts_MTF)
    return MTF_tensor


# ts2MTF(wind_data[0, 0:16].reshape(1, -1), label='test01')


# 递归图
def ts2RP(time_series, catalog='train', label=None, threshold=None):
    ts_Rp = RecurrencePlot(threshold=threshold).fit_transform(time_series)
    RP_tensor = torch.from_numpy(ts_Rp)
    return RP_tensor


# print(ts2RP(wind_data[0:16].reshape(1, -1), label='test01').size())


def ts2combine(time_series, catalog='train', label="test01", GAF_method="summation", MTF_strategy="normal",
               RP_threshold=None):
    GASF_tensor = ts2GAF(time_series, catalog, label, 'summation')
    GADF_tensor = ts2GAF(time_series, catalog, label, 'difference')
    # MTF_tensor = ts2MTF(time_series, catalog, label, MTF_strategy)
    # RP_tensor = ts2RP(time_series, catalog, label, RP_threshold)
    # Combine_Tensor = torch.cat([GASF_tensor, GASF_tensor])
    Combine_Tensor = torch.cat([GASF_tensor, GADF_tensor])
    # Combine_Tensor =torch.stack([GASF_tensor, MTF_tensor,RP_tensor])
    # print(Combine_Tensor.shape)
    return Combine_Tensor


def combine_matrices_with_ts_element(time_series, catalog='train', label="test01", GAF_method="summation",
                                     MTF_strategy="normal", RP_threshold=None):
    """
    将GAF、MTF、RP矩阵按顺序水平拼接，并在每行开头添加时间序列中对应的元素。

    参数:
    time_series (numpy.ndarray): 时间序列数据。
    catalog (str): 数据类别，如'train'或'test'。
    label (str): 数据标签。
    GAF_method (str): GAF转换方法，'summation'或'difference'。
    MTF_strategy (str): MTF转换策略。
    RP_threshold (float): RP转换的阈值。

    返回:
    torch.Tensor: 拼接后的矩阵。
    """
    # 生成GAF、MTF和RP矩阵
    GAF_tensor = ts2GAF(time_series, catalog, label, GAF_method)
    MTF_tensor = ts2MTF(time_series, catalog, label, MTF_strategy)
    RP_tensor = ts2RP(time_series, catalog, label, RP_threshold)
    # 水平拼接三个矩阵
    combined_matrix = torch.cat([GAF_tensor, MTF_tensor, RP_tensor], dim=2)
    # 在每行开头添加时间序列中对应的元素
    ts_elements = torch.from_numpy(time_series).reshape(1, -1, 1)  # 调整ts_elements的形状为[1, n, 1]
    final_matrix = torch.cat([ts_elements, combined_matrix], dim=2)

    return final_matrix.squeeze(0)

# 示例调用
# combined_matrix = combine_matrices_with_ts_element(wind_data[0:32].reshape(1, -1))
# print(combined_matrix.size())

# 示例使用
# file_path = '../data/' + 'Input_data.csv'  # 替换为您的文件路径
# dates_array, prices_array = load_and_process_csv(file_path)
# print("价格数组:", prices_array[:16])
# print(combine_matrices_with_ts_element(prices_array[0:16].reshape(1, -1)))
