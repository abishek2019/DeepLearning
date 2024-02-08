import torch
import torch.nn as nn
import torch.nn.init as init

# ---------------------------------------------GAN2 original--------------------------------------------------------------------------
# class Discriminator(nn.Module):
#     def __init__(self, kernel_size=3):
#         super(Discriminator, self).__init__()
#         # self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
#         # self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
#         # self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
#         # self.lstm = nn.LSTM(input_size=256, hidden_size=128, batch_first=True, bidirectional=True)
#         # self.final_conv = nn.Conv1d(in_channels=256, out_channels=1, kernel_size=3, padding=1)
#         # self.relu = nn.ReLU()
#         # self.downsample = nn.MaxPool1d(kernel_size=2, stride=2)
#
#     def forward(self, x):
#         # x = self.relu(self.conv1(x))
#         # x = self.downsample(x)
#         # x = self.relu(self.conv2(x))
#         # x = self.downsample(x)
#         # x = self.relu(self.conv3(x))
#         # x = self.downsample(x)
#         # x = x.permute(0, 2, 1)  # Reshape for LSTM: (batch_size, sequence_length, channels)
#         # x, _ = self.lstm(x)
#         # x = x.permute(0, 2, 1)  # Reshape back to (batch_size, channels, sequence_length) for convolution
#         # x = self.final_conv(x)
#         # return x
# -------------------------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------GAN2 extended (used)------------------------------------------------------------------
# Three Conv1d, BiLSTM and Conv1d (with ReLU and Pool)
# WE can try 2.1.1noFC as well from this model

# class Discriminator(nn.Module):
#     def __init__(self, kernel_size =3):
#         super(Discriminator, self).__init__()
#         self.conv1d = nn.Conv1d(2, 8, kernel_size=3, stride=1, padding=1)
#         self.bilstm = nn.LSTM(8, 8, bidirectional=True, batch_first=True)
#
#         # CNN module with spectral normalization
#         self.cnn_blocks = nn.Sequential(
#             nn.utils.spectral_norm(nn.Conv1d(16, 16, kernel_size=3, stride=2, padding=1)),
#             nn.LeakyReLU(),
#             # nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.utils.spectral_norm(nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1)),
#             nn.LeakyReLU(),
#             nn.utils.spectral_norm(nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1)),
#             nn.LeakyReLU(),
#             nn.utils.spectral_norm(nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)),
#             nn.LeakyReLU(),
#             nn.utils.spectral_norm(nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1)),
#             nn.LeakyReLU(),
#             # nn.utils.spectral_norm(nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))),
#             # nn.LeakyReLU(),
#         )
#         # self.conv2 = nn.Conv1d(256, 8, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv1d(64, 1, kernel_size=3, stride=1, padding=1)
#
#         # # Linear layer
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.linear1 = nn.Linear(1, 1)
#
#     def forward(self, x):
#         x = self.conv1d(x)
#         x = x.transpose(1, 2)     # Transpose for LSTM
#         x, _ = self.bilstm(x)
#         x = x.transpose(1, 2)
#         # x = x.unsqueeze(1)        # Expand dimensions for 2D convolution
#         x = self.cnn_blocks(x)
#         # Conv1d
#         # x = x.squeeze(2)
#         x = self.conv2(x)
#         # x = self.conv3(x)
#
#         # Linear layer
#         x = self.avg_pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.linear1(x)
#         # x = torch.sigmoid(x)
#         return x
# -----------------------------------------------------------------------------------------------------------------------------------
import torch.nn.init as init

class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()

        # self.conv1 = nn.Conv1d(2, 8, kernel_size=3, stride=2, padding=1)
        def conv_layer(in_ch, out_ch):
            layer = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
            init.xavier_uniform_(layer.weight)
            return layer

        def bn(num_features):
            return nn.BatchNorm1d(num_features, 0.8)

        self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6 = (conv_layer(in_channels, 32), conv_layer(32, 64),
               conv_layer(64, 128),conv_layer(128, 256), conv_layer(256, 512), conv_layer(512, 1))

        self.bn2, self.bn3, self.bn4, self.bn5 = bn(64), bn(128), bn(256), bn(512)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # The height and width of downsampled audio
        # ds_size = 32000 // 2 ** 5
        self.fcn = nn.Linear(1, 1)
        init.xavier_uniform_(self.fcn.weight)
        # self.bilstm = nn.LSTM(16, 16, bidirectional=True, batch_first=True)



    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        # x = x.transpose(1, 2)
        # x, _ = self.bilstm(x)
        # x = x.transpose(1, 2)
        x = self.lrelu(self.bn2(self.conv2(x)))
        x = self.lrelu(self.bn3(self.conv3(x)))
        x = self.lrelu(self.bn4(self.conv4(x)))
        x = self.lrelu(self.bn5(self.conv5(x)))
        x = self.lrelu(self.conv6(x))
        x = self.avg_pool(x)
        # x = torch.sigmoid(x)
        x = x.view(x.shape[0], -1)
        x = self.fcn(x)
        return x



# # class Discriminator(nn.Module):
# #     def __init__(self, kernel_size =3):
# #         super(Discriminator, self).__init__()
# #         self.conv1 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=1, padding=1)
# #         self.conv_layers = nn.Sequential(
# #             nn.Conv2d(in_channels=1, out_channels=8, kernel_size=1, stride=2, padding=1),
# #             nn.LeakyReLU(0.1, inplace=True),
# #             nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
# #             nn.LeakyReLU(0.1, inplace=True),
# #             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=1),
# #             nn.LeakyReLU(0.1, inplace=True),
# #             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=2, padding=1),
# #             nn.LeakyReLU(0.1, inplace=True),
# #             nn.AdaptiveAvgPool2d((1,1)),
# #         )
# #         self.fc_layers = nn.Sequential(
# #             nn.Linear(64, 64),
# #             nn.LeakyReLU(0.1, inplace=True),
# #             nn.Linear(64, 10),
# #             nn.LeakyReLU(0.1, inplace=True),
# #             nn.Linear(10, 1)
# #         )
# #
# #     def forward(self, x):
# #         x = self.conv1(x)
# #         x = x.unsqueeze(1)
# #         x = self.conv_layers(x)
# #         x = x.view(x.size(0), -1)  # Flatten the output for fully connected layers
# #         x = self.fc_layers(x)
# #         return x
#
#
# class Discriminator(nn.Module):
#     def __init__(self, in_ch):
#         super(Discriminator, self).__init__()
#
#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Conv1d(in_ch, 256, kernel_size=32, stride=16, bias=False),
#             nn.LeakyReLU(0.2, True),
#         )
#
#         # self.cnnModule =  nn.Sequential(
#         #                 nn.Conv1d(in_channels=in_ch, out_channels=8, kernel_size=3, stride=1, padding=1),
#         #                 nn.GroupNorm(1, 8),
#         #                 nn.LeakyReLU(0.2, inplace=True),
#         #                 nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
#         #                 nn.GroupNorm(1, 16),
#         #                 nn.LeakyReLU(0.2, inplace=True),
#         #                 nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
#         #                 nn.GroupNorm(1, 32),
#         #                 nn.LeakyReLU(0.2, inplace=True),
#         #                 nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
#         #                 nn.GroupNorm(1, 64),
#         #                 nn.LeakyReLU(0.2, inplace=True),
#         #                 # nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
#         #                 # nn.GroupNorm(1, 256),
#         #                 # nn.LeakyReLU(0.2, inplace=True),
#         #     # nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
#         #     # nn.GroupNorm(1, 512),
#         #     # nn.LeakyReLU(0.2, inplace=True),
#         #                 )
#
#         # TCN
#         self.tcn = self._build_tcn()
#
#         # self.receptive = nn.Sequential(
#         #     nn.Conv1d(in_channels=64, out_channels=16, kernel_size=3, stride=4, padding=1),
#         #     nn.GroupNorm(1, 16),        #SpectralNorm????
#         #     nn.LeakyReLU(0.2, inplace=True),
#         #     nn.Conv1d(in_channels=16, out_channels=4, kernel_size=3, stride=4, padding=1),
#         #     nn.GroupNorm(1, 4),
#         #     nn.LeakyReLU(0.2, inplace=True),
#         #     nn.Conv1d(in_channels=4, out_channels=1, kernel_size=3, stride=4, padding=1),
#         #     nn.LeakyReLU(0.2, inplace=True),
#         #     # nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
#         #     # nn.GroupNorm(1, 128),
#         #     # nn.LeakyReLU(0.2, inplace=True),
#         #     # nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
#         #     # nn.GroupNorm(1, 256),
#         #     # nn.LeakyReLU(0.2, inplace=True),
#         # )
#
#         # Decoder
#         self.decoder = nn.Sequential(
#             nn.Conv1d(256, 1, kernel_size=32, stride=16, bias=False),
#             nn.LeakyReLU(0.2, True),
#         )
#
#
#         # 1-D convolutional layers
#         # self.conv1 = nn.Conv1d(256, 8, kernel_size=32,stride=16)
#         # self.conv2 = nn.Conv1d(8, 1, kernel_size=1)
#
#         # Fully connected layer
#         self.fc = nn.Linear(in_features=121, out_features=1)  # Adjust in_features based on the output shape
#
#         # LeakyReLU and Global Layer Normalization
#         # self.leaky_relu = nn.LeakyReLU(0.2, True)
#         # self.global_norm = nn.GroupNorm(1, 8)  # Adjust num_groups based on the number of features
#         # self.gap = nn.AdaptiveAvgPool1d(1)
#         self.apply(self.init_weights)
#
#     def init_weights(self, m):
#         if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
#             init.xavier_uniform_(m.weight.data)
#             if m.bias is not None:
#                 init.constant_(m.bias.data, 0)
#
#     def _build_tcn(self):
#         tcn_blocks = []
#         for s in range(8):  # B=8
#             for i in range(2):  # R=3
#                 tcn_blocks.append(self._build_tcn_block())
#         return nn.Sequential(*tcn_blocks)
#
#     def _build_tcn_block(self):
#         return nn.Sequential(
#             # nn.Conv1d(64, 64, kernel_size=3, stride=1),
#             # nn.GroupNorm(1, 64),
#             # nn.LeakyReLU(0.2, True),
#             nn.Conv1d(256, 256, kernel_size=3, stride=1),
#             nn.GroupNorm(1, 256),
#             nn.LeakyReLU(0.2, True),
#         )
#
#     def forward(self, x):
#         # x = self.tanh(x)
#         # print(f'Before Encoder: {x.shape}')
#         x = self.encoder(x)
#         # x = self.cnnModule(x)
#         # print(f'After Encoder: {x.shape}')
#         x = self.tcn(x)
#         # print(f'After tcn: {x.shape}')
#         # x = self.receptive(x)
#         # print(f'After receptive: {x.shape}')
#         x = self.decoder(x)
#         # x = self.conv1(x)
#         # x = self.leaky_relu(x)
#         # x = self.global_norm(x)
#         # x = self.conv2(x)
#         # print(f'After decoder: {x.shape}')
#         x = x.view(x.size(0), -1)  # Flatten for fully connected layer
#         # print(f'Before FCN: {x.shape}')
#         # x = self.gap(x)
#         x = self.fc(x)
#
#         return x


