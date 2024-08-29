import pandas as pd

# 读取CSV文件
df = pd.read_csv('savefile_filtered.csv', header=None)
# 选取目标点和125到244帧的数据
columns_to_keep = [0] + list(range(125, 245))

# 取前两行数据并转换为浮点数类型的NumPy数组
df = df.iloc[:, columns_to_keep]
# 处理后的数据包括target和最后120帧的gaze,共121列
df.to_csv('modified_file.csv', index=False, header =False)
