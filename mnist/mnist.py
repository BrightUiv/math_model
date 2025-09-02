import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image # 从 Pillow 库导入 Image 模块

# ==================================================================
# 1. 定义模型结构 (必须和训练时的一模一样)
# ==================================================================
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

# ==================================================================
# 2. 定义一个函数来处理图片并进行预测
# ==================================================================
def predict_digit(image_path, model, device):
    # --- a. 图像预处理 ---
    # 打开图片
    img = Image.open(image_path)

    # 定义和训练时完全相同的图像转换流程
    # 注意：MNIST是灰度图，所以我们需要先将图片转为灰度
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), # 转换为单通道灰度图
        transforms.Resize((28, 28)),                # 调整大小为 28x28
        transforms.ToTensor(),                       # 转换为张量
        transforms.Normalize((0.1307,), (0.3081,))   # 标准化
    ])

    # 应用转换
    img_tensor = transform(img)

    # === 处理背景反转问题 (非常重要！) ===
    # MNIST 是白字黑底，但我们画图通常是黑字白底。
    # ToTensor() 会把像素值从 [0, 255] 转到 [0, 1]。
    # 如果是黑字白底，那么背景(白色)会变成1，字体(黑色)会变成0。
    # 这和 MNIST 的数据正好相反，所以需要反转一下。
    # 我们通过检查图像张量的平均值来判断是否需要反转。
    # 如果平均值大于0.5，说明大部分是白色背景，需要反转。
    if img_tensor.mean() > 0.5:
        img_tensor = 1.0 - img_tensor

    # 添加一个批次维度 (batch dimension)
    # 模型的输入需要是 [batch_size, channels, height, width]
    # 我们只有一张图片，所以 batch_size 是 1
    img_tensor = img_tensor.unsqueeze(0).to(device)


    # --- b. 进行预测 ---
    # 将模型设置为评估模式
    model.eval()

    # 关闭梯度计算
    with torch.no_grad():
        output = model(img_tensor)
        # 获取预测结果
        pred = output.argmax(dim=1, keepdim=True)
        # 获取概率分布
        probabilities = torch.softmax(output, dim=1)

    return pred.item(), probabilities

# ==================================================================
# 3. 主程序：加载模型并调用预测函数
# ==================================================================
if __name__ == '__main__':
    # --- 设置 ---
    MODEL_PATH = "mnist_mlp_model.pth"  # 模型文件路径
    IMAGE_PATH = "figure.png"         # 【请在这里修改为您自己的图片文件名】

    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 加载模型 ---
    # 实例化模型结构
    model = MLP().to(device)
    # 加载训练好的参数
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    # `map_location=device` 确保即使你是在没有GPU的电脑上加载模型，也能正常工作

    # --- 进行预测 ---
    try:
        predicted_digit, probabilities = predict_digit(IMAGE_PATH, model, device)
        print(f"模型预测的数字是: {predicted_digit}")
        # 打印每个类别的概率
        print("\n详细概率分布:")
        for i, p in enumerate(probabilities.squeeze()):
            print(f"数字 {i}: {p.item():.4f}  ({'=' * int(p.item() * 50)})")
    except FileNotFoundError:
        print(f"错误：找不到图片文件 '{IMAGE_PATH}'。请确保文件存在于同一目录下，并且文件名正确。")
    except Exception as e:
        print(f"发生了一个错误: {e}")