import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from librosa.filters import mel as librosa_mel_fn


def weights_init(m):
    """
    初始化模型权重。

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


class Audio2Mel(nn.Module):
    """
    Audio2Mel 类实现了一个音频到梅尔频谱的转换模块。
    该模块将原始音频信号转换为梅尔频谱表示，常用于语音合成和音频生成任务。

    参数说明:
        n_fft (int, 可选): FFT 窗口大小，默认为1024。
        hop_length (int, 可选): 帧移长度，默认为256。
        win_length (int, 可选): 窗长度，默认为1024。
        sampling_rate (int, 可选): 采样率，默认为22050。
        n_mel_channels (int, 可选): 梅尔频谱的通道数，默认为80。
        mel_fmin (float, 可选): 梅尔频率的最小值，默认为0.0。
        mel_fmax (float, 可选): 梅尔频率的最大值，默认为None。
    """
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        sampling_rate=22050,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=None,
    ):
        super().__init__()
        
        # FFT 参数设置     
        # 生成窗口                   
        window = torch.hann_window(win_length).float()
        # 生成梅尔滤波器组
        mel_basis = librosa_mel_fn(
            sampling_rate, n_fft, n_mel_channels, mel_fmin, mel_fmax
        )

        # 转换为 PyTorch 张量
        mel_basis = torch.from_numpy(mel_basis).float()
        # 注册梅尔滤波器组为缓冲区
        self.register_buffer("mel_basis", mel_basis)
        # 注册窗为缓冲区
        self.register_buffer("window", window)
        # FFT 窗口大小
        self.n_fft = n_fft
        # 帧移长度
        self.hop_length = hop_length
        # 窗长度
        self.win_length = win_length
        # 采样率
        self.sampling_rate = sampling_rate
        # 梅尔频谱通道数
        self.n_mel_channels = n_mel_channels

    def forward(self, audio):
        """
        前向传播方法，执行音频到梅尔频谱的转换。

        参数:
            audio (Tensor): 输入音频信号，形状为 (B, 1, T)。

        返回:
            Tensor: 梅尔频谱，形状为 (B, n_mel_channels, T').

        步骤:
            1. 对输入音频进行反射填充。
            2. 计算短时傅里叶变换（STFT）。
            3. 计算频谱幅度。
            4. 将幅度转换为梅尔频谱。
            5. 对梅尔频谱取对数。
        """
        # 计算填充大小
        p = (self.n_fft - self.hop_length) // 2
        # 对音频进行反射填充并去除通道维度
        audio = F.pad(audio, (p, p), "reflect").squeeze(1)
        # 计算短时傅里叶变换
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
        )

        # 分离实部和虚部
        real_part, imag_part = fft.unbind(-1)
        # 计算幅度
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        # 将幅度转换为梅尔频谱
        mel_output = torch.matmul(self.mel_basis, magnitude)
        # 对梅尔频谱取对数
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
        # 返回梅尔频谱
        return log_mel_spec


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
            nn.LeakyReLU(0.2), # 应用 LeakyReLU 激活函数
            nn.ReflectionPad1d(dilation), # 应用反射填充
            WNConv1d(dim, dim, kernel_size=3, dilation=dilation), # 应用带权重归一化的1D卷积层
            nn.LeakyReLU(0.2), # 再次应用 LeakyReLU 激活函数
            WNConv1d(dim, dim, kernel_size=1), # 应用带权重归一化的1D卷积层
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


class Generator(nn.Module):
    """
    Generator 类实现了一个生成器模型，用于将低分辨率的特征图转换为高分辨率的音频信号。
    该模型通过多个转置卷积层和残差块，逐步增加特征图的分辨率和感受野。

    参数说明:
        input_size (int): 输入特征图的通道数。
        ngf (int): 生成器的基础通道数。
        n_residual_layers (int): 残差层的数量。
    """
    def __init__(self, input_size, ngf, n_residual_layers):
        super().__init__()
        # 上采样率列表
        ratios = [8, 8, 2, 2]
        # 计算跳步长度
        self.hop_length = np.prod(ratios)
        # 计算初始通道数的倍数
        mult = int(2 ** len(ratios))

        # 构建模型的第一部分：初始卷积层
        model = [
            nn.ReflectionPad1d(3), # 反射填充
            WNConv1d(input_size, mult * ngf, kernel_size=7, padding=0), # 应用带权重归一化的1D卷积层
        ]

        # 对特征图进行上采样，使其达到原始音频的尺度
        for i, r in enumerate(ratios):
            model += [
                nn.LeakyReLU(0.2),  # 应用 LeakyReLU 激活函数
                WNConvTranspose1d(  # 应用带权重归一化的转置1D卷积层
                    mult * ngf,
                    mult * ngf // 2,
                    kernel_size=r * 2,
                    stride=r,
                    padding=r // 2 + r % 2,
                    output_padding=r % 2,
                ),
            ]

            # 添加残差块
            for j in range(n_residual_layers):
                model += [ResnetBlock(mult * ngf // 2, dilation=3 ** j)]

            # 减少通道数
            mult //= 2

        # 构建模型的最后部分：最后的卷积层和激活函数
        model += [
            nn.LeakyReLU(0.2),  # 应用 LeakyReLU 激活函数
            nn.ReflectionPad1d(3),  # 反射填充
            WNConv1d(ngf, 1, kernel_size=7, padding=0),  # 应用带权重归一化的1D卷积层
            nn.Tanh(),  # 应用 Tanh 激活函数
        ]

        # 将所有层组合成一个 Sequential 模型
        self.model = nn.Sequential(*model)
        # 应用权重初始化
        self.apply(weights_init)

    def forward(self, x):
        """
        前向传播方法，执行生成器的前向计算。

        参数:
            x (Tensor): 输入特征图，形状为 (B, input_size, T)。

        返回:
            Tensor: 生成的高分辨率音频信号，形状为 (B, 1, T * hop_length)。
        """
        return self.model(x)


class NLayerDiscriminator(nn.Module):
    """
    NLayerDiscriminator 类实现了一个多层的判别器模型，用于区分真实音频和生成音频。
    该模型通过多个卷积层和激活函数，逐步提取音频特征，并最终输出判别结果。

    参数说明:
        ndf (int): 判别器的基础通道数。
        n_layers (int): 判别器的层数。
        downsampling_factor (int): 下采样因子，用于控制卷积层的步长。
    """
    def __init__(self, ndf, n_layers, downsampling_factor):
        super().__init__()
        # 使用 ModuleDict 来存储模型中的各个层
        model = nn.ModuleDict()

        # 第一层：包含反射填充、1D卷积和LeakyReLU激活函数
        model["layer_0"] = nn.Sequential(
            nn.ReflectionPad1d(7), # 反射填充
            WNConv1d(1, ndf, kernel_size=15), # 应用权重归一化的1D卷积层
            nn.LeakyReLU(0.2, True), # 应用 LeakyReLU 激活函数
        )

        # 当前通道数初始化为 ndf
        nf = ndf
        # 下采样因子
        stride = downsampling_factor
        for n in range(1, n_layers + 1):
            # 前一层的通道数
            nf_prev = nf
            # 计算当前层的通道数，最大不超过1024
            nf = min(nf * stride, 1024)

            # 添加卷积层和LeakyReLU激活函数
            model["layer_%d" % n] = nn.Sequential(
                WNConv1d(
                    nf_prev,  # 输入通道数
                    nf,  # 输出通道数
                    kernel_size=stride * 10 + 1,  # 卷积核大小
                    stride=stride,  # 步长
                    padding=stride * 5,  # 填充大小
                    groups=nf_prev // 4,  # 分组卷积
                ),
                # 应用 LeakyReLU 激活函数
                nn.LeakyReLU(0.2, True),
            )

        # 倒数第二层：通道数翻倍
        nf = min(nf * 2, 1024)
        model["layer_%d" % (n_layers + 1)] = nn.Sequential(
            # 应用权重归一化的1D卷积层
            WNConv1d(nf_prev, nf, kernel_size=5, stride=1, padding=2),
            # 应用 LeakyReLU 激活函数
            nn.LeakyReLU(0.2, True),
        )

        # 最后一层：输出通道数为1
        model["layer_%d" % (n_layers + 2)] = WNConv1d(
            nf, 1, kernel_size=3, stride=1, padding=1 # 应用权重归一化的1D卷积层
        )

        # 将模型存储在 ModuleDict 中
        self.model = model

    def forward(self, x):
        """
        前向传播方法，执行判别器的前向计算。

        参数:
            x (Tensor): 输入音频信号，形状为 (B, 1, T)。

        返回:
            List[Tensor]: 每一层的输出结果列表。
        """
        # 初始化结果列表
        results = []
        for key, layer in self.model.items():
            # 通过当前层
            x = layer(x)
            # 添加到结果列表中
            results.append(x)
        return results


class Discriminator(nn.Module):
    """
    Discriminator 类实现了一个多判别器模型，由多个 NLayerDiscriminator 组成。
    该模型通过多个判别器并行处理输入音频信号，并最终输出判别结果。

    参数说明:
        num_D (int): 判别器的数量。
        ndf (int): 判别器的基础通道数。
        n_layers (int): 判别器的层数。
        downsampling_factor (int): 下采样因子，用于控制卷积层的步长。
    """
    def __init__(self, num_D, ndf, n_layers, downsampling_factor):
        super().__init__()
        # 使用 ModuleDict 来存储多个判别器
        self.model = nn.ModuleDict()
        for i in range(num_D):
            self.model[f"disc_{i}"] = NLayerDiscriminator(
                ndf, n_layers, downsampling_factor # 创建 NLayerDiscriminator 实例
            )

        # 定义平均池化层用于下采样
        self.downsample = nn.AvgPool1d(4, stride=2, padding=1, count_include_pad=False)
        # 应用权重初始化
        self.apply(weights_init)

    def forward(self, x):
        """
        前向传播方法，执行判别器的前向计算。

        参数:
            x (Tensor): 输入音频信号，形状为 (B, 1, T)。

        返回:
            List[List[Tensor]]:: 每个判别器的输出结果列表。
        """
        # 初始化结果列表
        results = []
        for key, disc in self.model.items():
            # 通过当前判别器
            results.append(disc(x))
            # 对输入进行下采样
            x = self.downsample(x)
        return results
