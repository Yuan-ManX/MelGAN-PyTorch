import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


def weights_init(m):
    """
    初始化模型权重

    参数:
        m (nn.Module): 要初始化的模型或层。

    说明:
        该函数遍历模型的所有层，根据层的类型应用不同的初始化方法。
        - 如果层名称包含 "Conv"，则使用均值为0.0，标准差为0.02的正态分布初始化权重。
        - 如果层名称包含 "BatchNorm2d"，则使用均值为1.0，标准差为0.02的正态分布初始化权重，并将偏置初始化为0。
    """
    classname = m.__class__.__name__

    # 如果类名包含 "Conv"
    if classname.find("Conv") != -1:
        # 使用正态分布初始化权重
        """
        正态分布初始化:
            mean: 0.0
            std: 0.02
        例如:
            weight ~ N(0.0, 0.02^2)
        """
        m.weight.data.normal_(0.0, 0.02)

    # 如果类名包含 "BatchNorm2d"
    elif classname.find("BatchNorm2d") != -1:
        # 使用正态分布初始化权重
        """
        正态分布初始化:
            mean: 1.0
            std: 0.02
        例如:
            weight ~ N(1.0, 0.02^2)
        """
        m.weight.data.normal_(1.0, 0.02)
        # 将偏置初始化为0
        m.bias.data.fill_(0)


def WNConv1d(*args, **kwargs):
    """
    应用权重归一化的1D卷积层。
    使用 torch.nn.utils.weight_norm 对 nn.Conv1d 进行权重归一化。
    权重归一化可以加速训练过程并提高模型的泛化性能。

    参数:
        *args: 传递给 nn.Conv1d 的位置参数。
        **kwargs: 传递给 nn.Conv1d 的关键字参数。

    返回:
        nn.Module: 应用权重归一化后的1D卷积层。
    """
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    """
    应用权重归一化的转置1D卷积层。
    使用 torch.nn.utils.weight_norm 对 nn.ConvTranspose1d 进行权重归一化。
    权重归一化可以加速训练过程并提高模型的泛化性能。

    参数:
        *args: 传递给 nn.ConvTranspose1d 的位置参数。
        **kwargs: 传递给 nn.ConvTranspose1d 的关键字参数。

    返回:
        nn.Module: 应用权重归一化后的转置1D卷积层。
    """
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class ResnetBlock(nn.Module):
    """
    ResnetBlock 类实现了一个残差块（Residual Block），该残差块包含两个卷积层和一个跳跃连接。
    该模块通过堆叠两个卷积层和激活函数，逐步增加感受野，同时通过跳跃连接保持信息的流动。

    参数说明:
        dim (int): 输入和输出的通道数。
        dilation (int, 可选): 卷积层的膨胀因子，默认为1。
    """
    def __init__(self, dim, dilation=1):
        super().__init__()
        # 定义残差块中的卷积层序列
        self.block = nn.Sequential(
            # 应用 LeakyReLU 激活函数
            nn.LeakyReLU(0.2),
            # 应用反射填充
            nn.ReflectionPad1d(dilation),
            # 应用带权重归一化的1D卷积层
            WNConv1d(dim, dim, kernel_size=3, dilation=dilation),
            # 再次应用 LeakyReLU 激活函数
            nn.LeakyReLU(0.2),
            # 应用带权重归一化的1D卷积层
            WNConv1d(dim, dim, kernel_size=1),
        )
        # 定义跳跃连接卷积层
        self.shortcut = WNConv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        """
        前向传播方法，执行残差块的前向计算。

        参数:
            x (Tensor): 输入张量。

        返回:
            Tensor: 输出张量。
        """
        # 跳跃连接 + 卷积块输出
        return self.shortcut(x) + self.block(x)


class MelGAN(nn.Module):
    """
    MelGAN 类实现了一个基于生成对抗网络（GAN）的音频生成模型。
    该模型通过多个转置卷积层和残差块，逐步将低分辨率的梅尔频谱转换为高分辨率的音频信号。
    MelGAN 结合了高效的计算资源利用和高质量的音频生成能力，广泛应用于语音合成和音频生成任务。

    参数说明:
        cfg: 配置参数对象，包含以下字段:
            preprocess:
                n_mel (int): 梅尔频谱的维度数。
            model:
                melgan:
                    ratios (List[int]): 上采样率列表。
                    ngf (int): 生成器的基础通道数。
                    n_residual_layers (int): 残差层的数量。
    """
    def __init__(self, cfg):
        super().__init__()
        # 配置参数
        self.cfg = cfg

        # 计算跳步长度
        self.hop_length = np.prod(self.cfg.model.melgan.ratios)
        # 计算初始通道数的倍数
        mult = int(2 ** len(self.cfg.model.melgan.ratios))

        # 构建模型的第一部分：初始卷积层
        model = [
            nn.ReflectionPad1d(3),  # 反射填充
            WNConv1d(  # 应用权重归一化的1D卷积层
                self.cfg.preprocess.n_mel,  # 输入通道数（梅尔频谱维度）
                mult * self.cfg.model.melgan.ngf,  # 输出通道数
                kernel_size=7,  # 卷积核大小
                padding=0,  # 填充大小
            ),
        ]

        # 对梅尔频谱进行上采样，使其达到原始音频的尺度
        for i, r in enumerate(self.cfg.model.melgan.ratios):
            # 添加 LeakyReLU 激活函数
            model += [
                nn.LeakyReLU(0.2),
                WNConvTranspose1d(  # 应用权重归一化的转置1D卷积层
                    mult * self.cfg.model.melgan.ngf,  # 输入通道数
                    mult * self.cfg.model.melgan.ngf // 2,  # 输出通道数
                    kernel_size=r * 2,  # 卷积核大小
                    stride=r,  # 上采样率
                    padding=r // 2 + r % 2,  # 填充大小
                    output_padding=r % 2,  # 输出填充
                ),
            ]

            # 添加残差块
            for j in range(self.cfg.model.melgan.n_residual_layers):
                model += [
                    ResnetBlock(mult * self.cfg.model.melgan.ngf // 2, dilation=3**j) # 应用残差块
                ]

            # 减少通道数
            mult //= 2

        # 构建模型的最后部分：最后的卷积层和激活函数
        model += [
            nn.LeakyReLU(0.2),  # LeakyReLU 激活函数
            nn.ReflectionPad1d(3),  # 反射填充
            WNConv1d(self.cfg.model.melgan.ngf, 1, kernel_size=7, padding=0), # 应用权重归一化的1D卷积层
            nn.Tanh(),
        ]

        # 将所有层组合成一个 Sequential 模型
        self.model = nn.Sequential(*model)
        # 应用权重初始化
        self.apply(weights_init)

    def forward(self, x):
        """
        前向传播方法，执行 MelGAN 的前向计算。

        参数:
            x (Tensor): 输入梅尔频谱，形状为 (B, n_mel, T)。

        返回:
            Tensor: 生成的高分辨率音频信号，形状为 (B, 1, T * hop_length)。
        """
        return self.model(x)
