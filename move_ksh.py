import pandas as pd
import matplotlib.pyplot as plt
import math
# 读取CSV文件
df = pd.read_csv('savefile.csv', header=None)

# 取前两行数据并转换为浮点数类型的NumPy数组
data = df.values.astype(float)

# 设置中文字体
plt.rcParams['font.family'] = ['SimHei', 'sans-serif']

# 创建图形窗口
fig, ax = plt.subplots(figsize=(8, 6))

# 实时更新动态折线图
group = 1
num_rows = data.shape[0]

for i in range(0, num_rows, 2):
    print(f"开始绘制第 {group} 组数据")
    group += 1

    if i + 1 < num_rows:  # 确保有成对的行数据
        x_coords = data[i]
        y_coords = data[i + 1]
        print(f"目标点坐标:{x_coords[0]}, {y_coords[0]}")
        for j in range(1, len(x_coords)):
            ax.set_title('动态绘制前两行数据的折线图')
            ax.set_xlabel('X 坐标')
            ax.set_ylabel('Y 坐标')
            ax.grid(True)
            ax.plot(x_coords[0], y_coords[0], marker='o', color='r')
            ax.plot(x_coords[1:j + 1], y_coords[1:j + 1], marker='o', color='b', linestyle='-')
            print(f"第{j}帧坐标:{x_coords[j]}, {y_coords[j]}")
            print(f"与target之间的距离:{math.sqrt((x_coords[j]-x_coords[0])**2 + (y_coords[j]-y_coords[0])**2)}")
            plt.draw()  # 绘制更新后的图形
            plt.pause(0.03)  # 暂停一段时间以显示动态效果
            ax.cla()
    print(f"第 {group - 1} 组数据绘制完成")
