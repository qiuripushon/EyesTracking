import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('savefile.csv', header=None)

# 计算每行的长度
df['length'] = df.apply(lambda row: len(row.dropna()), axis=1)

# 过滤掉长度不足 246 的行
filtered_df = df[df['length'] >= 246]

# 删除长度列
filtered_df = filtered_df.drop(columns=['length'])

# 保存到新的 CSV 文件
filtered_df.to_csv('savefile_filtered.csv', index=False, header=False)
