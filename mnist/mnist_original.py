import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 0. 定义超参数和设备
batch_size = 64
learning_rate = 0.001
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. 加载和预处理 MNIST 数据集 (保持不变)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# 2. 构建神经网络模型 (保持不变)
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.layers(x)


# 实例化模型并移动到指定设备
model = MLP().to(device)
print(model)

# 3. 定义损失函数和优化器 (保持不变)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# 4. 训练模型 (保持不变)
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 100 == 0:
            print(
                f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


# 5. 评估模型 (保持不变)
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(
        f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)\n')


# 6. 执行训练和评估 (保持不变)
for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

# ==================================================================
# 7. 【新增】保存训练好的模型
# ==================================================================
MODEL_PATH = "mnist_mlp_model.pth"
torch.save(model.state_dict(), MODEL_PATH)
print(f"模型已成功保存到: {MODEL_PATH}")


# ==================================================================
# 8. 【新增】加载模型并验证 (这是一个演示)
# ==================================================================
# 创建一个新的、未经训练的模型实例
loaded_model = MLP().to(device)

# 加载我们刚刚保存的参数
loaded_model.load_state_dict(torch.load(MODEL_PATH))

# 将加载的模型设置为评估模式
loaded_model.eval()

print("\n--- 使用加载的模型进行最终验证 ---")
# 使用 test 函数来验证加载后的模型是否和训练结束时的模型性能一致
test(loaded_model, device, test_loader)