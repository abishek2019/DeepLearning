import torch
import torch.nn as nn
from typing import Optional, Tuple

# Note: All settings are non-causal setting
class ConvBlock(nn.Module):
    """1D Convolutional block.
    Args: io_channels (int): The number of input/output channels, <B, Sc>
          hidden_channels (int): The number of channels in the internal layers, <H>.
          kernel_size (int): The convolution kernel size of the middle layer, <P>.
          padding (int): Padding value of the convolution in the middle layer.
          dilation (int, optional): Dilation value of the convolution in the middle layer.
          no_residual (bool, optional): Disable residual block/output.
    """
    def __init__(self, io_channels: int, hidden_channels: int, kernel_size: int, padding: int, dilation: int = 1, no_residual: bool = False):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=io_channels, out_channels=hidden_channels, kernel_size=1),
            nn.PReLU(),
            nn.GroupNorm(num_groups=1, num_channels=hidden_channels, eps=1e-08),
            nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=hidden_channels),
            nn.PReLU(),
            nn.GroupNorm(num_groups=1, num_channels=hidden_channels, eps=1e-08))
        self.res_out = (None if no_residual
            else nn.Conv1d(in_channels=hidden_channels, out_channels=io_channels, kernel_size=1))
        self.skip_out = nn.Conv1d(in_channels=hidden_channels, out_channels=io_channels, kernel_size=1)

    def forward(self, input: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        feature = self.conv_layers(input)
        if self.res_out is None:
            residual = None
        else:
            residual = self.res_out(feature)
        skip_out = self.skip_out(feature)
        return residual, skip_out

class MaskGenerator(torch.nn.Module):
    """TCN Separation Module. Generates masks for separation.
    Args: input_dim (int): Input feature dimension, <N>.
          num_sources (int): The number of sources to separate.
          kernel_size (int): The convolution kernel size of conv blocks, <P>.
          num_featrs (int): Input/output feature dimension of conv blocks, <B, Sc>.
          num_hidden (int): Intermediate feature dimension of conv blocks, <H>
          num_layers (int): The number of conv blocks in one stack, <X>.
          num_stacks (int): The number of conv block stacks, <R>.
          msk_activate (str): The activation function of the mask output.
    """
    def __init__(self, input_dim: int, num_sources: int, kernel_size: int, num_feats: int, num_hidden: int, num_layers: int, num_stacks: int, msk_activate: str):
        super().__init__()
        self.input_dim = input_dim
        self.num_sources = num_sources
        self.input_norm = torch.nn.GroupNorm(num_groups=1, num_channels=input_dim, eps=1e-8)
        self.input_conv = torch.nn.Conv1d(in_channels=input_dim, out_channels=num_feats, kernel_size=1)
        self.receptive_field = 0
        self.conv_layers = torch.nn.ModuleList([])
        for s in range(num_stacks):
            for l in range(num_layers):
                multi = 2**l
                self.conv_layers.append(
                    ConvBlock(io_channels=num_feats, hidden_channels=num_hidden, kernel_size=kernel_size, dilation=multi,
                              padding=multi,
                              # The last ConvBlock does not need residual
                              no_residual=(l == (num_layers - 1) and s == (num_stacks - 1))))
                self.receptive_field += kernel_size if s == 0 and l == 0 else (kernel_size - 1) * multi
        self.output_prelu = torch.nn.PReLU()
        self.output_conv = torch.nn.Conv1d(in_channels=num_feats, out_channels=input_dim * num_sources,kernel_size=1)
        if msk_activate == "sigmoid":
            self.mask_activate = torch.nn.Sigmoid()
        elif msk_activate == "relu":
            self.mask_activate = torch.nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation {msk_activate}")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """ Args: input (torch.Tensor): 3D Tensor with shape [batch, features, frames]
            Returns: Tensor: shape [batch, num_sources, features, frames]"""
        batch_size = input.shape[0]
        feats = self.input_norm(input)
        feats = self.input_conv(feats)
        output = 0.0
        for layer in self.conv_layers:
            residual, skip = layer(feats)
            if residual is not None:  # the last conv layer does not produce residual
                feats = feats + residual
            output = output + skip
        output = self.output_prelu(output)
        output = self.output_conv(output)
        output = self.mask_activate(output)
        return output.view(batch_size, self.num_sources, self.input_dim, -1)

class ConvTasNet(torch.nn.Module):
    """Conv-TasNet architecture
    Args: num_sources (int, optional): The number of sources to split.
          enc_kernel_size (int, optional): The convolution kernel size of the encoder/decoder, <L>.
          enc_num_feats (int, optional): The feature dimensions passed to mask generator, <N>.
          msk_kernel_size (int, optional): The convolution kernel size of the mask generator, <P>.
          msk_num_feats (int, optional): The input/output feature dimension of conv block in the mask generator, <B, Sc>.
          msk_num_hidden_feats (int, optional): The internal feature dimension of conv block of the mask generator, <H>.
          msk_num_layers (int, optional): The number of layers in one conv block of the mask generator, <X>.
          msk_num_stacks (int, optional): The numbr of conv blocks of the mask generator, <R>.
          msk_activate (str, optional): The activation function of the mask output (Default: ``sigmoid``)."""
    def __init__(
        self,
        num_sources: int = 2,
        # encoder/decoder parameters
        enc_kernel_size: int = 16,
        enc_num_feats: int = 512,
        # mask generator parameters
        msk_kernel_size: int = 3,
        msk_num_feats: int = 128,
        msk_num_hidden_feats: int = 512,
        msk_num_layers: int = 8,
        msk_num_stacks: int = 3,
        msk_activate: str = "sigmoid"):

        super().__init__()
        self.num_sources = num_sources
        self.enc_num_feats = enc_num_feats
        self.enc_kernel_size = enc_kernel_size
        self.enc_stride = enc_kernel_size // 2
        self.encoder = torch.nn.Conv1d(in_channels=1, out_channels=enc_num_feats, kernel_size=enc_kernel_size, stride=self.enc_stride, padding=self.enc_stride, bias=False)
        self.mask_generator = MaskGenerator(input_dim=enc_num_feats, num_sources=num_sources, kernel_size=msk_kernel_size, num_feats=msk_num_feats, num_hidden=msk_num_hidden_feats, num_layers=msk_num_layers, num_stacks=msk_num_stacks, msk_activate=msk_activate)
        self.decoder = torch.nn.ConvTranspose1d( in_channels=enc_num_feats, out_channels=1, kernel_size=enc_kernel_size, stride=self.enc_stride, padding=self.enc_stride, bias=False)

    def _align_num_frames_with_strides(self, input: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Pad input Tensor so that the end of the input tensor corresponds with
        1. (if kernel size is odd) the center of the last convolution kernel
        or 2. (if kernel size is even) the end of the first half of the last convolution kernel
        Assumption:
            The resulting Tensor will be padded with the size of stride (== kernel_width // 2)
            on the both ends in Conv1D

        |<--- k_1 --->|
        |      |            |<-- k_n-1 -->|
        |      |                  |  |<--- k_n --->|
        |      |                  |         |      |
        |      |                  |         |      |
        |      v                  v         v      |
        |<---->|<--- input signal --->|<--->|<---->|
         stride                         PAD  stride

        Args:
            input (torch.Tensor): 3D Tensor with shape (batch_size, channels==1, frames)
        Returns:
            Tensor: Padded Tensor
            int: Number of paddings performed"""

        batch_size, num_channels, num_frames = input.shape
        is_odd = self.enc_kernel_size % 2
        num_strides = (num_frames - is_odd) // self.enc_stride
        num_remainings = num_frames - (is_odd + num_strides * self.enc_stride)
        if num_remainings == 0:
            return input, 0
        num_paddings = self.enc_stride - num_remainings
        pad = torch.zeros(batch_size, num_channels, num_paddings, dtype=input.dtype, device=input.device)
        return torch.cat([input, pad], 2), num_paddings

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        """Separates Sources
        Args: input (torch.Tensor): 3D Tensor with shape [batch, channel==1, frames]
        Returns: Tensor: 3D Tensor with shape [batch, channel==num_sources, frames]"""
        if input.ndim != 3 or input.shape[1] != 1:
            raise ValueError(f"Expected 3D tensor (batch, channel==1, frames). Found: {input.shape}")
        # B: batch size
        # L: input frame length
        # L': padded input frame length
        # F: feature dimension
        # M: feature frame length
        # S: number of sources
        padded, num_pads = self._align_num_frames_with_strides(input)  # B, 1, L'
        batch_size, num_padded_frames = padded.shape[0], padded.shape[2]
        feats = self.encoder(padded)  # B, F, M
        masked = self.mask_generator(feats) * feats.unsqueeze(1)  # B, S, F, M
        masked = masked.view(batch_size * self.num_sources, self.enc_num_feats, -1)  # B*S, F, M
        decoded = self.decoder(masked)  # B*S, 1, L'
        output = decoded.view(batch_size, self.num_sources, num_padded_frames)  # B, S, L'
        if num_pads > 0:
            output = output[..., :-num_pads]  # B, S, L
        return output

def conv_tasnet_base(num_sources: int = 2) -> ConvTasNet:
    """Builds non-causal version of :class:`~torchaudio.models.ConvTasNet`.
    The parameter settings follow the ones with the highest Si-SNR metirc score in the paper,
    except the mask activation function is changed from "sigmoid" to "relu" for performance improvement.
    Args: num_sources (int, optional): Number of sources in the output. (Default: 2)
    Returns: ConvTasNet: ConvTasNet model."""
    return ConvTasNet( num_sources=num_sources, enc_kernel_size=16, enc_num_feats=512, msk_kernel_size=3, msk_num_feats=128, msk_num_hidden_feats=512, msk_num_layers=8, msk_num_stacks=3, msk_activate="relu")


class Discriminator(nn.Module):
    def __init__(self, kernel_size =3):
        super(Discriminator, self).__init__()
# """ 1---------------------------------------------GAN1-------------------------------------------------------------------------"""
# One Conv1d, BiLSTM and Conv1d (no Relu or Pool)

        # self.conv1d = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=kernel_size) # 1D Convolutional Layer
        # self.bilstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True, bidirectional=True) # Bidirectional LSTM
        # self.conv_output = nn.Conv1d(in_channels=2 * 64, out_channels=1, kernel_size=kernel_size)# Convolutional Layer
# """ 1---------------------------------------------------------------------------------------------------------------------------"""

# """ 2.1---------------------------------------------GAN2 original-----------------------------------------------------------------"""
# Three Conv1d, BiLSTM and Conv1d (with ReLU and Pool)

        # self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        # self.lstm = nn.LSTM(input_size=256, hidden_size=128, batch_first=True, bidirectional=True)
        # self.final_conv = nn.Conv1d(in_channels=256, out_channels=1, kernel_size=3, padding=1)
        # self.relu = nn.ReLU()
        # self.downsample = nn.MaxPool1d(kernel_size=2, stride=2)
# """ 2.1---------------------------------------------------------------------------------------------------------------------------"""

# """ F2.1.1---------------------------------------------GAN2 original extended------------------------------------------------------"""
# Three Conv1d, BiLSTM and Conv1d (with ReLU and Pool)
# WE can try 2.1.1noFC as well from this model
        # 1-D Convolution
        self.conv1d = nn.Conv1d(2, 16, kernel_size=3, stride=1, padding=1)
        self.bilstm = nn.LSTM(16, 16, bidirectional=True, batch_first=True)

        # CNN module with spectral normalization
        self.cnn_blocks = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.utils.spectral_norm(nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.utils.spectral_norm(nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.utils.spectral_norm(nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.utils.spectral_norm(nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.utils.spectral_norm(nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            nn.PReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Conv1d(64, 1, kernel_size=3, stride=1, padding=1)

        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        #
        # # Linear layers
        # self.linear1 = nn.Linear(64, 16)
        # self.linear2 = nn.Linear(16, 1)

# """ F2.1.1-------------------------------------------------------------------------------------------------------------------------"""

# """ 2.2---------------------------------------------GAN2 original Upsampled-------------------------------------------------------"""
# Three Conv1d, BiLSTM and Conv1d (with ReLU and Upsample)

        # self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        # # Upsample more using transposed convolution with stride=4 or 2
        # self.upsample = nn.ConvTranspose1d(in_channels=256, out_channels=256, kernel_size=2, stride=2)
        # self.lstm = nn.LSTM(input_size=256, hidden_size=128, batch_first=True, bidirectional=True)
        # self.final_conv = nn.Conv1d(in_channels=256, out_channels=1, kernel_size=3, padding=1)
        # self.relu = nn.ReLU()
# """ 2.2---------------------------------------------------------------------------------------------------------------------------"""

# """ 2.3----------------------------------------GAN2 with regularization + optional FCL--------------------------------------------"""
# Three Conv1d, BiLSTM and Conv1d (with regularization, ReLU and Pool)

        # self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        # self.batch_norm1 = nn.BatchNorm1d(64)  # Add BatchNorm1d after conv1
        # self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # self.batch_norm2 = nn.BatchNorm1d(128)  # Add BatchNorm1d after conv2
        # self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        # self.batch_norm3 = nn.BatchNorm1d(256)  # Add BatchNorm1d after conv3
        # self.lstm = nn.LSTM(input_size=256, hidden_size=128, batch_first=True, bidirectional=True)
        # self.final_conv = nn.Conv1d(in_channels=256, out_channels=1, kernel_size=3, padding=1)
        # self.relu = nn.ReLU()
        # self.downsample = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.fc = nn.Linear(24*4000*1, 1)
# """ 2.3----------------------------------------------------------------------------------------------------------------------------"""

# """ 2.4-----------------------------------------GAN2 with Regularization Upsampled ------------------------------------------------"""
# Three Conv1d, BiLSTM and Conv1d (with regularization, ReLU and Upsample)
# went out of memory for batch size 24

        # self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        # self.batch_norm1 = nn.BatchNorm1d(64)
        # self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # self.batch_norm2 = nn.BatchNorm1d(128)
        # self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        # self.batch_norm3 = nn.BatchNorm1d(256)
        # # Upsample more using transposed convolution with stride=4 or 2
        # self.upsample = nn.ConvTranspose1d(in_channels=256, out_channels=256, kernel_size=2, stride=2)
        # self.lstm = nn.LSTM(input_size=256, hidden_size=128, batch_first=True, bidirectional=True)
        # self.final_conv = nn.Conv1d(in_channels=256, out_channels=1, kernel_size=3, padding=1)
        # self.relu = nn.ReLU()
# """ 2.4----------------------------------------------------------------------------------------------------------------------------"""

# """ 3--------------------------------------------------GAN3------------------------------------------------------------------------"""
# Five Conv1d, BiLSTM and Conv1d (with ReLU and Pool)

        # self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        # self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        # self.conv5 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        # self.bilstm = nn.LSTM(input_size=1024, hidden_size=512, batch_first=True, bidirectional=True)
        # self.conv6 = nn.Conv1d(in_channels=1024, out_channels=1, kernel_size=3, padding=1)
        # self.relu = nn.ReLU()
        # self.downsample = nn.MaxPool1d(kernel_size=2, stride=2)
# """ 3-------------------------------------------------------------------------------------------------------------------------------"""

# """ 4-----------------------------------------------GAN4----------------------------------------------------------------------------"""
# One Conv1d, BiLSTM and three Conv1d (with ReLU and Pool)

        # self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        # self.lstm = nn.LSTM(input_size=64, hidden_size=128, batch_first=True, bidirectional=True)
        # self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        # self.conv4 = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=1)  # Adjust kernel size here
        # self.relu = nn.ReLU()
        # self.downsample = nn.MaxPool1d(kernel_size=4, stride=4)  # Adjust pooling parameters
# """ 4-------------------------------------------------------------------------------------------------------------------------------"""

    def forward(self, x):
# """ F1---------------------------------------------GAN1------------------------------------------------------------------------"""
# One Conv1d, BiLSTM and Conv1d (no Relu or Pool)

        # x = self.conv1d(x) # x has shape [1, 2, x]
        # x = x.transpose(1, 2) # Transpose to [batch_size, sequence_length, hidden_size]
        # x, _ = self.bilstm(x) # Apply Bidirectional LSTM
        # x = x.transpose(1, 2) # Transpose back to [batch_size, hidden_size, sequence_length]
        # x = self.conv_output(x) # Apply another 1D convolution
        # x = x.squeeze(0) # Transpose to [batch_size, sequence_length]
# """ F1---------------------------------------------------------------------------------------------------------------------------"""

# """ F2.1---------------------------------------------GAN2 original-----------------------------------------------------------------"""
# Three Conv1d, BiLSTM and Conv1d (with ReLU and Pool)

        # x = self.relu(self.conv1(x))
        # x = self.downsample(x)
        # x = self.relu(self.conv2(x))
        # x = self.downsample(x)
        # x = self.relu(self.conv3(x))
        # x = self.downsample(x)
        # x = x.permute(0, 2, 1)  # Reshape for LSTM: (batch_size, sequence_length, channels)
        # x, _ = self.lstm(x)
        # x = x.permute(0, 2, 1)  # Reshape back to (batch_size, channels, sequence_length) for convolution
        # x = self.final_conv(x)
        # return x
# """ F2.1---------------------------------------------------------------------------------------------------------------------------"""

# """ F2.1.1---------------------------------------------GAN2 original Extended------------------------------------------------------"""

        x = self.conv1d(x)
        # Transpose for LSTM
        x = x.transpose(1, 2)
        x, _ = self.bilstm(x)
        x = x.transpose(1, 2)
        # Expand dimensions for 2D convolution
        x = x.unsqueeze(1)
        x = self.cnn_blocks(x)
        x = x.squeeze(2)
        # print(f'squeeze out: {x.shape}')
        x = self.conv2(x)
        # x = self.avg_pool(x)
        # x = x.view(x.size(0), -1)
        # x = self.linear1(x)
        # x = self.linear2(x)

        return x
# """ F2.1.1-------------------------------------------------------------------------------------------------------------------------"""

# """ F2.2---------------------------------------------GAN2 original Upsampled-------------------------------------------------------"""
# Three Conv1d, BiLSTM and Conv1d (with ReLU and Pool)

        # x = self.relu(self.conv1(x))
        # x = self.relu(self.conv2(x))
        # x = self.relu(self.conv3(x))
        # # Upsample more using transposed convolution with stride=4 or 2
        # # print(f'B4 UPsample: {x.shape}')
        # x = self.upsample(x)
        # # print(f'After UPsample: {x.shape}')
        # x = x.permute(0, 2, 1)  # Reshape for LSTM: (batch_size, sequence_length, channels)
        # x, _ = self.lstm(x)
        # x = x.permute(0, 2, 1)  # Reshape back to (batch_size, channels, sequence_length) for convolution
        # x = self.final_conv(x)
        # return x
# """ F2.2---------------------------------------------------------------------------------------------------------------------------"""


# """ F2.3----------------------------------------GAN2 with regularization + optional FCL--------------------------------------------"""
# Three Conv1d, BiLSTM and Conv1d (with regularization, ReLU and Pool)

        # x = self.batch_norm1(self.conv1(x))
        # x = self.relu(x)  # Apply BatchNorm after conv1
        # x = self.downsample(x)
        # x = self.batch_norm2(self.conv2(x))
        # x = self.relu(x)  # Apply BatchNorm after conv2
        # x = self.downsample(x)
        # x = self.batch_norm3(self.conv3(x))
        # x = self.relu(x)  # Apply BatchNorm after conv3
        # x = self.downsample(x)
        # x = x.permute(0, 2, 1)
        # x, _ = self.lstm(x)
        # x = x.permute(0, 2, 1)
        # x = self.final_conv(x)
        # # x = x.view(x.size(0), -1)
        # # x = x.flatten()
        # # x = self.fc(x)
        # return x
# """ F2.3----------------------------------------------------------------------------------------------------------------------------"""

# """ F2.4-------------------------------------------GAN2 with regularization Upsampled-----------------------------------------------"""
# Three Conv1d, BiLSTM and Conv1d (with regularization, ReLU and Upsample)

        # x = self.batch_norm1(self.conv1(x))
        # x = self.relu(x)
        # x = self.batch_norm2(self.conv2(x))
        # x = self.relu(x)
        # x = self.batch_norm3(self.conv3(x))
        # x = self.relu(x)
        # # Upsample more using transposed convolution with stride=4 or 2
        # x = self.upsample(x)
        # x = x.permute(0, 2, 1)
        # x, _ = self.lstm(x)
        # x = x.permute(0, 2, 1)
        # x = self.final_conv(x)
        # print(f'Discr op size: {x.size()}')
        # return x
# """ F2.4----------------------------------------------------------------------------------------------------------------------------"""

# """ F3--------------------------------------------------GAN3------------------------------------------------------------------------"""
# Five Conv1d, BiLSTM and Conv1d (with ReLU and Pool)

        # x = self.relu(self.conv1(x))
        # x = self.downsample(x)
        # x = self.relu(self.conv2(x))
        # x = self.downsample(x)
        # x = self.relu(self.conv3(x))
        # x = self.downsample(x)
        # x = self.relu(self.conv4(x))
        # x = self.downsample(x)
        # x = self.relu(self.conv5(x))
        # x = x.permute(0, 2, 1)  # Reshape for BiLSTM: (batch_size, sequence_length, channels)
        # x, _ = self.bilstm(x)
        # x = x.permute(0, 2, 1)  # Reshape back to (batch_size, channels, sequence_length) for convolution
        # x = self.conv6(x)
        # return x
# """ F3-------------------------------------------------------------------------------------------------------------------------------"""

# """ F4-----------------------------------------------GAN4----------------------------------------------------------------------------"""
# One Conv1d, BiLSTM and three Conv1d (with ReLU and Pool)

        # x = self.conv1(x)
        # x = self.relu(x)
        # # Apply LSTM
        # x, _ = self.lstm(x.permute(0, 2, 1))  # Permute to (batch_size, sequence_length, input_size)
        # x = x.permute(0, 2, 1)  # Revert permutation
        # # Additional convolutional layers
        # x = self.conv2(x)
        # x = self.relu(x)
        # x = self.conv3(x)
        # x = self.relu(x)
        # # Adjusting output shape
        # x = self.conv4(x)
        # x = self.downsample(x)
        # return x
# """ F4-------------------------------------------------------------------------------------------------------------------------------"""

#-------------------------PESQ IMPLEMENTATION WITH DISCR AS TCN---------------------------------------------------------------------------
# class GlobalLayerNorm(nn.Module):
#     def __init__(self, num_features):
#         super(GlobalLayerNorm, self).__init__()
#         self.num_features = num_features
#         self.gamma = nn.Parameter(torch.ones(num_features))
#         self.beta = nn.Parameter(torch.zeros(num_features))
#
#     def forward(self, x):
#         mean = x.mean(dim=-1, keepdim=True)
#         std = x.std(dim=-1, keepdim=True)
#         x = self.gamma * (x - mean) / (std + 1e-8) + self.beta
#         return x




# class GlobalLayerNorm(nn.Module):
#     def __init__(self, num_features):
#         super(GlobalLayerNorm, self).__init__()
#         self.num_features = num_features
#         self.gamma = nn.Parameter(torch.ones(1, num_features, 1))
#         self.beta = nn.Parameter(torch.zeros(1, num_features, 1))
#
#     def forward(self, x):
#         mean = x.mean(dim=(1, 2), keepdim=True)
#         std = x.std(dim=(1, 2), keepdim=True)
#         x = self.gamma * (x - mean) / (std + 1e-8) + self.beta
#         return x
#
# #
# class ConvBlockDiscr(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride):
#         super(ConvBlockDiscr, self).__init__()
#         self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
#         self.norm = GlobalLayerNorm(out_channels)
#         self.activation = nn.LeakyReLU(0.2)
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.norm(x)
#         x = self.activation(x)
#         return x
#
#
class TCNBlockDiscr(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, depthwise_channels, B=8, R=2):
        super(TCNBlockDiscr, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, depthwise_channels, kernel_size=1)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(depthwise_channels, depthwise_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
                # GlobalLayerNorm(depthwise_channels),
                nn.LeakyReLU(0.2)
            )
            for _ in range(B)
        ])
        self.conv2 = nn.Conv1d(depthwise_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        for layer in self.layers:
            x = x + layer(x)
        x = self.conv2(x)
        return x
#
# # Without LeakyReLU and Regularization
# # class Discriminator(nn.Module):
# #     def __init__(self):
# #         super(Discriminator, self).__init__()
# #         self.encoder = nn.Sequential(
# #             ConvBlockDiscr(2, 256, kernel_size=16, stride=8),
# #         )
# #         self.tcn = TCNBlockDiscr(256, 256, kernel_size=3, stride=1, depthwise_channels=256, B=8, R=2)
# #         self.conv3 = ConvBlockDiscr(256, 8, kernel_size=15, stride=1)
# #         self.conv4 = ConvBlockDiscr(8, 1, kernel_size=1, stride=1)
# #         self.avgpool = nn.AdaptiveAvgPool1d(1)
# #
# #
# #     def forward(self, x):
# #         x = self.encoder(x)
# #         x = self.tcn(x)
# #         x = self.conv3(x)
# #         x = self.conv4(x)
# #         x = self.avgpool(x)
# #         print(f'Pool o/p: {x.shape}')
# #         x = x.view(x.size(0), -1)  # Flatten
# #         print(f'Flattened o/p: {x.shape}')
# #         x = x.view(-1)  # Flatten
# #         # print(f'Flattened o/p: {x.shape}')
# #         # x = torch.mean(x)
# #         x = self.fc(x)
# #         print(f'Final FC o/p: {x.shape}')
# #         return x
#
#
# With LeakyReLU and Regularization
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(4, 256, kernel_size=16, stride=8),
            nn.LeakyReLU(0.2),
            # nn.GroupNorm(256, 256)  # Global Layer Normalization
        )
        self.tcn = TCNBlockDiscr(256, 256, kernel_size=3, stride=1, depthwise_channels=256, B=8, R=2)

        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 1, kernel_size=15, stride=1),
            nn.LeakyReLU(0.2),
            # nn.GroupNorm(8, 8)  # Global Layer Normalization
        )
        # self.conv4 = nn.Conv1d(8, 1, kernel_size=1, stride=1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(1, 1)  # Fully connected layer with 1 linear node

    def forward(self, x):
        x = self.encoder(x)
        x = self.tcn(x)
        x = self.conv3(x)
        # x = self.conv4(x)
        x = self.avg_pool(x)  # Global average pooling along the temporal dimension
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x
