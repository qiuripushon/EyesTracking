import numpy as np
import matplotlib.pyplot as plt

# 加载预测点和真实点的数据
file_path = 'predicted_and_true_points.csv'  # 请确保文件路径正确
data = np.loadtxt(file_path, delimiter=',', skiprows=1)

# 提取预测点和真实点
predicted_points = data[:, :2]
true_points = data[:, 2:]

# 随机选择50个样本的索引
num_samples = 50
random_indices = np.random.choice(len(predicted_points), size=num_samples, replace=False)

# 根据随机索引提取样本
random_predicted_points = predicted_points[random_indices]
random_true_points = true_points[random_indices]

# 计算每个预测点和真实点之间的距离
distances = np.linalg.norm(random_predicted_points - random_true_points, axis=1)

plt.figure(figsize=(12, 6))

plt.bar(range(num_samples), distances, color='g', alpha=0.6)
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.title('Distances between Predicted and True Points (50 Random Samples)')

plt.tight_layout()
plt.show()
