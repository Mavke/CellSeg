import torch
from torch import nn

class PatchExpanding(nn.Module):
    def __init__(self, input_channels, kernel=3, stride=2, padding=1, ):
        super(PatchExpanding, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels=input_channels, out_channels=input_channels, kernel_size=kernel, stride=stride, padding=padding)
    
    def forward(self, x, output_size):
        x = self.upsample(x, output_size)
        return x

class DecoderConvolution(nn.Module):
    def __init__(self, input_channels, output_channels, padding=1, kernel=3):
        super(DecoderConvolution, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, padding=padding, kernel_size=kernel)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        
        self.convLayer1 = nn.Sequential(self.conv1, self.bn1, self.relu)

    def forward(self, x):
        x = self.convLayer1(x.float())
        return x

class SkipConnection(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(SkipConnection, self).__init__()

        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1,bias=False)
        self.norm = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        
        return x

class PredictionHead(nn.Module):
    
    def __init__(self,  input_channels, K):
        super(PredictionHead, self).__init__()

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=K, kernel_size=1)
        self.predictor = nn.Sequential(self.upsample, self.conv1)
    def forward(self,x):
        out = self.predictor(x)
        return out

class MultiScaleFeatures(nn.Module):
    """_summary_
    This module is used to extract multi scale features.
    """
    def __init__(self, kernel_size: list, ratios: list, input_channel_size: int, out_put_channel_size: int, drop_out_rate: float = 0):
        """_summary_

        Args:
            kernel_size (list): kernel sizes
            ratios (list): The ratio for each kernel size for the output feature
            input_channel_size (int): Number of channels
        """
        super(MultiScaleFeatures,self).__init__()
        self.convBlock = nn.ModuleList()
        self.drop_out = nn.Dropout2d(drop_out_rate)
        for kernel, ratio in zip(kernel_size, ratios):
            padding = (kernel - 1) // 2
            output_size = int(out_put_channel_size // ratio)
            conv = nn.Conv2d(input_channel_size, output_size, kernel, stride = 2, padding = padding, bias=False)

            seq = nn.Sequential(conv)
            self.convBlock.append(seq)
        
    def forward(self, x):
        multiScales = []
        
        for conv in self.convBlock:
           tmp = conv(x)
           tmp = self.drop_out(tmp)
           multiScales.append(tmp)

        scales_x = torch.cat(multiScales, dim=1)
        return scales_x


class PatchMerg(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(PatchMerg, self).__init__()
        
        self.conv = nn.Conv2d(kernel_size= 3, in_channels= input_channels, out_channels=output_channels, stride = 2, padding = 1)

   
    def forward(self, x):
        x = self.conv(x)
        
        return x

class ChannelAttention(nn.Module):
    def __init__(self, k = 3, name = None):
        super(ChannelAttention, self).__init__()
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1,1, kernel_size=k, padding = int(k/2), bias=False)
        self.sigmoid = nn.Sigmoid()
        self.name = name
   
    def forward(self, x):

        y = self.avg_pooling(x.float())
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)

        y = self.sigmoid(y)

        res = x * y.expand_as(x)
        return res

class PatchEmbedding(nn.Module):
    def __init__(self, input_chan, embedding_dim = 64, patch_size = 4):
        super(PatchEmbedding, self).__init__()

        self.embedding_dim = embedding_dim

        self.conv1 = nn.Conv2d(input_chan, self.embedding_dim, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(self.embedding_dim)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=patch_size, stride=patch_size, padding=0)
    

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.max_pool(x)
            
        return x