import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.family'] = ['SimHei', 'sans-serif']


# 数据加载模块
def load_data(file_path):
    data = pd.read_csv(file_path, header=None)
    gaze_points = []
    target_points = []

    for i in range(0, len(data), 2):
        # 读取gaze点
        x_coords = data.iloc[i, 1:245].values.astype(float)
        y_coords = data.iloc[i + 1, 1:245].values.astype(float)
        t_x_coords = data.iloc[i, 0:1].values.astype(float)
        t_y_coords = data.iloc[i + 1, 0:1].values.astype(float)
        gaze_points.append(list(zip(x_coords, y_coords)))
        target_points.append(list(zip(t_x_coords, t_y_coords)))
    return gaze_points, target_points


# 计算速度和状态
def analyze_gaze(gaze_points, speed_threshold=0.01):
    results = []

    for gaze in gaze_points:
        states = []
        for i in range(1, len(gaze)):
            # 计算速度
            dx = gaze[i][0] - gaze[i - 1][0]
            dy = gaze[i][1] - gaze[i - 1][1]
            speed = np.sqrt(dx ** 2 + dy ** 2)

            # 判断状态
            if speed < speed_threshold:
                states.append('注视')
            else:
                states.append('扫视')

        results.append(states)

    return results


# 主代码
file_path = 'savefile.csv'  # 修改为你的CSV文件路径
gaze_points, target_points = load_data(file_path)
gaze_states = analyze_gaze(gaze_points)


# 可视化结果
def visualize_gaze(gaze_points, gaze_states, target_points):
    fig, ax = plt.subplots(figsize=(8, 6))

    # 绘制线条
    for i in range(len(gaze_points)):
        x_coords, y_coords = zip(*gaze_points[i])
        ax.plot(x_coords, y_coords, linestyle='-', alpha=0.5, label=f'路径 {i + 1}')
        # 根据状态绘制点并设置颜色
        for j in range(1, len(gaze_points[i])):
            state = gaze_states[i][j - 1]
            color = 'green' if state == '注视' else 'red'
            ax.plot(x_coords[j], y_coords[j], marker='o', color=color, markersize=6 )

        # 绘制目标点
        t_x, t_y = target_points[i][0]
        ax.plot(t_x, t_y, marker='o', color='black', markersize=8)

    # 添加图例
    ax.plot([], [], marker='o', color='green', markersize=8, linestyle='None', label='注视')
    ax.plot([], [], marker='o', color='red', markersize=8, linestyle='None', label='扫视')
    ax.plot([], [], marker='o', color='black', markersize=8, linestyle='None', label='目标点')
    ax.legend()

    # 设置图形标题和标签
    ax.set_title('眼动追踪状态图')
    ax.set_xlabel('X 坐标')
    ax.set_ylabel('Y 坐标')
    ax.grid(True)

    # 显示图形
    plt.show()


# 可视化第1组数据
visualize_gaze([gaze_points[0]], [gaze_states[0]], [target_points[0]])
