import pandas as pd
import numpy as np


# 数据加载模块
def load_data(file_path):
    data = pd.read_csv(file_path, header=None)
    gaze_points = []

    for i in range(0, len(data), 2):
        # 读取gaze点
        x_coords = data.iloc[i, 1:245].values.astype(float)
        y_coords = data.iloc[i + 1, 1:245].values.astype(float)
        gaze_points.append(list(zip(x_coords, y_coords)))
    return gaze_points


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
gaze_points = load_data(file_path)
gaze_states = analyze_gaze(gaze_points)

# 输出结果
for i, states in enumerate(gaze_states):
    print(f"第 {i + 1} 组眼动状态：")
    for j, state in enumerate(states):
        print(f"  帧 {j + 1}: {state}")
