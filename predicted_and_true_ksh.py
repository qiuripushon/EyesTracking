import numpy as np
import matplotlib.pyplot as plt

# 加载预测点和真实点的数据
file_path = 'predicted_and_true_points.csv'  # 请确保文件路径正确
data = np.loadtxt(file_path, delimiter=',', skiprows=1)

# 提取预测点和真实点
predicted_points = data[:, :2]
true_points = data[:, 2:]

# 随机选择10个样本的索引
num_samples = 10
random_indices = np.random.choice(len(predicted_points), size=num_samples, replace=True)

# 根据随机索引提取样本
random_predicted_points = predicted_points[random_indices]
random_true_points = true_points[random_indices]

# 画出预测点（用三角形表示）和真实点（用圆形表示）
plt.scatter(random_true_points[:, 0], random_true_points[:, 1], color='blue', marker='o', label='True Points')
plt.scatter(random_predicted_points[:, 0], random_predicted_points[:, 1], color='red', marker='^', label='Predicted Points')



# 添加标签和标题
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Predicted Points vs True Points')
plt.legend()

# 显示图表
plt.grid(True)
plt.show()
