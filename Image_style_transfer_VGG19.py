import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import vgg19
from PIL import Image
import copy
import os
# 指定模型缓存目录
os.environ['TORCH_HOME'] = './model_directory'  # 这里填写你希望存储模型的目录

# 加载图像
def load_image(image_path, max_size=400):
    image = Image.open(image_path).convert('RGB')

    # 如果图片过大则调整大小
    size = min(max_size, max(image.size))

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = transform(image).unsqueeze(0)  # 添加批次维度
    return image


# 保存图像
def save_image(tensor, path):
    image = tensor.clone().detach()
    image = image.squeeze(0)  # 去掉批次维度
    image = transforms.ToPILImage()(image)
    image.save(path)


# 定义VGG19模型，只提取特定层的特征
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = vgg19(pretrained=True).features[:21].eval()  # 只使用前21层

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in {0, 5, 10, 19, 21}:  # 选择特定层的输出
                features.append(x)
        return features


# 内容损失函数
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        return nn.functional.mse_loss(input, self.target)


# 风格损失函数
class StyleLoss(nn.Module):
    def __init__(self, target):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target).detach()

    def gram_matrix(self, input):
        batch_size, channels, height, width = input.size()
        features = input.view(batch_size * channels, height * width)
        G = torch.mm(features, features.t())
        return G.div(batch_size * channels * height * width)

    def forward(self, input):
        G = self.gram_matrix(input)
        return nn.functional.mse_loss(G, self.target)


# 图像风格迁移
def style_transfer(content_img, style_img, num_steps=1000, style_weight=1e9, content_weight=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content_img = content_img.to(device)
    style_img = style_img.to(device)

    model = VGG().to(device)

    # 提取风格和内容特征
    style_features = model(style_img)
    content_features = model(content_img)

    # 初始化输入图像（使用内容图像作为初始图像）
    input_img = content_img.clone().requires_grad_(True).to(device)

    # 定义优化器
    optimizer = optim.LBFGS([input_img])

    style_losses = []
    content_losses = []

    # 创建损失模块
    for sf, cf in zip(style_features, content_features):
        content_losses.append(ContentLoss(cf))
        style_losses.append(StyleLoss(sf))

    run = [0]
    while run[0] <= num_steps:

        def closure():
            optimizer.zero_grad()

            input_features = model(input_img)
            content_loss = 0
            style_loss = 0

            for cl, input_f in zip(content_losses, input_features):
                content_loss += content_weight * cl(input_f)

            for sl, input_f in zip(style_losses, input_features):
                style_loss += style_weight * sl(input_f)

            loss = content_loss + style_loss
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f'Step {run[0]}, Content Loss: {content_loss.item():4f}, Style Loss: {style_loss.item():4f}')

            return loss

        optimizer.step(closure)

    # 取消归一化并返回结果
    unnormalize = transforms.Normalize(
        mean=[-2.118, -2.036, -1.804],
        std=[4.367, 4.464, 4.444]
    )
    result = unnormalize(input_img)
    return result


if __name__ == '__main__':
    content_image_path = 'content_image.png'
    style_image_path = 'style_image.png'
    output_image_path = 'output_image.jpg'

    content_img = load_image(content_image_path)
    style_img = load_image(style_image_path)

    result = style_transfer(content_img, style_img)

    save_image(result, output_image_path)
    print(f"风格迁移完成，图像已保存为 {output_image_path}")
