import torch
from torch import nn
import torch.nn.functional as F


class SosNet(nn.Module):
    def __init__(self, n_inputs=11, n_outputs=1, wide=False):
        super(SosNet, self).__init__()
        scale = 2 if wide else 1
        self.net = nn.Sequential(
            # Encoder
            DownBlock(n_inputs, 32 * scale, 32 * scale, 3, stride=[1, 2], pool=None,   push=False, layers=3),
            DownBlock(32 * scale,  32 * scale,  32 * scale,     3, stride=[1, 2], pool=None,   push=False, layers=3),
            DownBlock(32 * scale,  32 * scale,  32 * scale,     3, stride=[1, 2], pool=None,   push=False, layers=3),
            UpsampleBlock(),
            DownBlock(32 * scale,  32 * scale,  32 * scale,     3, stride=[1, 2], pool=None,   push=True,  layers=3),
            DownBlock(32 * scale,  32 * scale,  64 * scale,     3, stride=1,      pool=[2, 2], push=True,  layers=3),
            DownBlock(64 * scale,  64 * scale,  128 * scale,    3, stride=1,      pool=[2, 2], push=True,  layers=3),
            DownBlock(128 * scale, 128 * scale, 512 * scale,    3, stride=1,      pool=[2, 2], push=False, layers=3),
            # Decoder
            UpBlock(512 * scale, 128 * scale, 3, scale_factor=2, pop=False, layers=3),
            UpBlock(256 * scale, 64 * scale, 3,  scale_factor=2, pop=True,  layers=3),
            UpBlock(128 * scale, 32 * scale, 3,  scale_factor=2, pop=True,  layers=3),
            UpBlock(64 * scale, 32 * scale, 3,   scale_factor=2, pop=True,  layers=3),
            UpStep(32 * scale, 32 * scale, 3,    scale_factor=1),
            Compress(32 * scale, n_outputs))

    def forward(self, x):
        y = self.net((x, []))
        return y


class UpsampleBlock(nn.Module):

    def __init__(self):
        super(UpsampleBlock, self).__init__()
    
    def forward(self, x):
        i, s = x 
        _, _, h, w = i.shape
        i = torch.nn.functional.interpolate(i, (h, h * 4), mode='bilinear')
        return (i, s)


class DownStep(nn.Module):
    """
    Down scaling step in the encoder decoder network
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, pool: tuple = None) -> None:
        """Constructor

        Arguments:
            in_channels {int} -- Number of input channels for 2D convolution
            out_channels {int} -- Number of output channels for 2D convolution
            kernel_size {tuple} -- Convolution kernel size

        Keyword Arguments:
            stride {int} -- Stride of convolution, set to 1 to disable (default: {1})
            pool {tuple} -- max pulling size, set to None to disable (default: {None})
        """
        super(DownStep, self).__init__()

        self.c = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2)
        self.n = nn.BatchNorm2d(out_channels)
        self.pool = pool

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Run the forward step

        Arguments:
            x {torch.tensor} -- input tensor

        Returns:
            torch.tensor -- output tensor
        """
        x = self.c(x)
        x = F.relu(x)
        if self.pool is not None:
            x = F.max_pool2d(x, self.pool)
        x = self.n(x)

        return x


class UpStep(nn.Module):
    """
    Up scaling step in the encoder decoder network
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, scale_factor: int = 2) -> None:
        """Constructor

        Arguments:
            in_channels {int} -- Number of input channels for 2D convolution
            out_channels {int} -- Number of output channels for 2D convolution
            kernel_size {int} -- Convolution kernel size

        Keyword Arguments:
            scale_factor {int} -- Upsampling scaling factor (default: {2})
        """
        super(UpStep, self).__init__()

        self.c = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.n = nn.BatchNorm2d(out_channels)
        self.scale_factor = scale_factor

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Run the forward step

        Arguments:
            x {torch.tensor} -- input tensor

        Returns:
            torch.tensor -- output tensor
        """
        if isinstance(x, tuple):
            x = x[0]

        if self.scale_factor != 1:
            x = F.interpolate(x, scale_factor=self.scale_factor)

        x = self.c(x)
        x = F.relu(x)
        x = self.n(x)

        return x


class Compress(nn.Module):
    """
    Up scaling step in the encoder decoder network
    """
    def __init__(self, in_channels: int, out_channels: int = 1, kernel_size: int = 1, scale_factor: int = 1) -> None:
        """Constructor

        Arguments:
            in_channels {int} -- [description]

        Keyword Arguments:
            out_channels {int} -- [description] (default: {1})
            kernel_size {int} -- [description] (default: {1})
        """
        super(Compress, self).__init__()

        self.scale_factor = scale_factor

        self.c = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Run the forward step

        Arguments:
            x {torch.tensor} -- input tensor

        Returns:
            torch.tensor -- output tensor
        """
        if isinstance(x, tuple) or isinstance(x, list):
            x = x[0]

        x = self.c(x)

        if self.scale_factor != 1:
            x = F.interpolate(x, scale_factor=self.scale_factor)

        return x


class DownBlock(nn.Module):
    def __init__(
            self,
            in_chan: int, inter_chan: int, out_chan: int,
            kernel_size: int = 3, stride: int = 1, pool: tuple = None,
            push: bool = False,
            layers: int = 3):
        super().__init__()

        self.s = []
        for i in range(layers):
            self.s.append(DownStep(
                in_chan if i == 0 else inter_chan,
                inter_chan if i < layers - 1 else out_chan,
                kernel_size,
                1 if i < layers - 1 else stride,
                None if i < layers - 1 else pool))
        self.s = nn.Sequential(*self.s)

        self.push = push

    def forward(self, x: torch.tensor) -> torch.tensor:
        i, s = x

        i = self.s(i)

        if self.push:
            s.append(i)

        return i, s


class UpBlock(nn.Module):
    def __init__(
            self,
            in_chan: int, out_chan: int,
            kernel_size: int, scale_factor: int = 2,
            pop: bool = False,
            layers: int = 3):
        super().__init__()

        self.s = []
        for i in range(layers):
            self.s.append(UpStep(
                in_chan if i == 0 else out_chan,
                out_chan,
                kernel_size,
                1 if i < layers - 1 else scale_factor))
        self.s = nn.Sequential(*self.s)

        self.pop = pop

    def forward(self, x: torch.tensor) -> torch.tensor:
        i, s = x

        if self.pop:
            i = torch.cat((i, s.pop()), dim=1)

        i = self.s(i)

        return i, s
