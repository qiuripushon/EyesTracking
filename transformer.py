import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np

# 自定义数据集类
class GazeDataset(Dataset):
    def __init__(self, gaze_points, target_points):
        self.gaze_points = gaze_points
        self.target_points = target_points

    def __len__(self):
        return len(self.gaze_points)

    def __getitem__(self, idx):
        gaze = self.gaze_points[idx]
        target = self.target_points[idx]
        return torch.tensor(gaze, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

# Transformer模型类定义
class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, batch_first=True), num_layers=num_layers)  # 添加 batch_first=True
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 将输入转换为嵌入
        x = self.embedding(x)
        # transformer_encoder 期望的输入形状为 (batch_size, seq_len, feature_size)
        x = self.transformer_encoder(x)
        # 取最后一个时间步的输出
        x = x[:, -1, :]  # 使用最后的输出
        x = self.fc(x)
        return x




# 数据加载模块
def load_data(file_path):
    data = pd.read_csv(file_path, header=None)
    gaze_points = []
    target_points = []

    for i in range(0, len(data), 2):
        # 读取target点
        target_x = data.iloc[i, 0]
        target_y = data.iloc[i + 1, 0]

        # 读取gaze点
        x_coords = data.iloc[i, 1:121].values.astype(float)
        y_coords = data.iloc[i + 1, 1:121].values.astype(float)

        gaze_points.append(list(zip(x_coords, y_coords)))
        target_points.append([target_x, target_y])

    return gaze_points, target_points

# 共6840组数据，5130组作为训练集，1710组作为测试集
def split_train_test_data(gaze_points, target_points, test_size=0.25, random_state=42):
    # 使用 train_test_split 函数进行数据集划分
    X_train, X_test, y_train, y_test = train_test_split(gaze_points, target_points, test_size=test_size,
                                                        random_state=random_state)
    return X_train, X_test, y_train, y_test

# 主代码
file_path = 'modified_file.csv'  # 修改为你的CSV文件路径
gaze_points, target_points = load_data(file_path)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = split_train_test_data(gaze_points, target_points, test_size=0.25, random_state=42)

# 参数设置
input_size = 2
hidden_size = 128
output_size = 2
num_layers = 3
batch_size = 64
num_epochs = 200
learning_rate = 0.0005

# 创建训练集和测试集的数据集和数据加载器
train_dataset = GazeDataset(X_train, y_train)
test_dataset = GazeDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 创建Transformer模型实例
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerModel(input_size, hidden_size, output_size, num_layers).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 模型训练
for epoch in range(num_epochs):
    model.train()
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 2 == 0:
        print(f'Train Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 模型测试
model.eval()
test_loss = 0.0
predicted_points = []
true_points = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)

        # 计算损失并累加
        loss = criterion(outputs, targets)
        test_loss += loss.item()

        # 将预测点和真实点保存起来
        predicted_points.append(outputs.cpu().numpy())
        true_points.append(targets.cpu().numpy())

# 计算平均测试损失
test_loss /= len(test_loader)
print(f'Test Loss: {test_loss:.4f}')


# 将预测点和真实点保存为numpy数组
predicted_points = np.concatenate(predicted_points, axis=0)
true_points = np.concatenate(true_points, axis=0)

# 合并预测点和真实点为一个numpy数组，每行两列分别是预测点和真实点
combined_data = np.concatenate((predicted_points, true_points), axis=1)

# 保存预测点和真实点到同一个文件
np.savetxt('predicted_and_true_points.csv', combined_data, delimiter=',',
           header='Predicted_X,Predicted_Y,True_X,True_Y', comments='')

# 输出预测点和真实点
print('预测点数和真实点数保存到predicted_and_true_points.csv')
print(combined_data)
