from datetime import datetime

import numpy as np
import pandas as pd


def load_and_process_csv(file_path):
    """
    读取CSV文件并处理，以提取日期和价格信息。

    参数:
    file_path (str): CSV文件的路径。

    返回:
    tuple: 包含两个numpy数组的元组，一个用于日期，一个用于价格。
    """

    # 读取CSV文件
    data = pd.read_csv(file_path, encoding='GBK')  # 如有必要，调整编码方式

    # 提取日期和价格列
    dates = data.iloc[:, 0]  # 假设日期在第一列
    prices = data.iloc[:, 1]  # 假设价格在第二列

    # 将日期字符串转换为datetime对象，然后转换为[月, 日]
    dates_converted = np.array([convert_date_to_month_day(date) for date in dates])

    # 将价格转换为numpy数组
    prices_converted = prices.to_numpy()

    return dates_converted, prices_converted


def convert_date_to_month_day(date_str):
    """
    将日期字符串转换为包含月份和日的列表。

    参数:
    date_str (str): 日期字符串。

    返回:
    list: 包含月份和日作为整数的列表。
    """
    date_obj = datetime.strptime(date_str, '%m/%d/%Y')
    return [date_obj.month, date_obj.day]


# # 示例使用
# file_path = '../data/' + 'Input_data.csv'  # 替换为您的文件路径
# dates_array, prices_array = load_and_process_csv(file_path)
#
# # 打印前几个元素以供验证
# print("日期数组:", dates_array[:5])
# print("价格数组:", prices_array[:5])
