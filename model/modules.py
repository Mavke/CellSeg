import torch
from torch import nn

class DeConvBlock(nn.Module):
    
    def __init__(self, input_channels, output_channels, kernel=2, stride=2, padding=0, drop_out_rate=0.0):
        super(DeConvBlock, self).__init__()
        
        upsample = nn.ConvTranspose2d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel, stride=stride, padding=padding)

        conv = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, stride=1, padding= 1, bias=False)


        self.block = nn.Sequential(
            upsample,
            conv,
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Dropout2d(drop_out_rate)
        )
    
    def forward(self, x):
        x = self.block(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, padding=1, kernel=3, drop_out_rate=0.0):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, padding=padding, kernel_size=kernel)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()

        self.drop_out = nn.Dropout2d(drop_out_rate)
        self.convLayer1 = nn.Sequential(self.conv1, self.bn1, self.relu, self.drop_out)

    def forward(self, x):
        x = self.convLayer1(x.float())
        return x


class SkipConnectionDepthwise(nn.Module):

    def __init__(self, input_channels, out_channels) -> None:
        super(SkipConnectionDepthwise, self).__init__()

        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, groups=input_channels,bias=False)
        self.norm = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=input_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)

        self.depthwise_conv = nn.Sequential(self.conv, self.norm, self.relu, self.conv2, self.norm2, self.relu)

    def forward(self, x):
        x = self.depthwise_conv(x)
        return x


class PredictionHead(nn.Module):
    
    def __init__(self,  input_channels, K):
        super(PredictionHead, self).__init__()

        self.upsample = nn.ConvTranspose2d(in_channels=input_channels, out_channels=input_channels, kernel_size=2, stride=2, padding=0)

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=K, kernel_size=1)
        self.predictor = nn.Sequential(self.upsample, self.conv1)
    
    def forward(self,x):
        out = self.predictor(x)
        return out

class ResNetBlock(nn.Module):
    def __init__(self, layers, input_chan, output_chan, kernel_size, first_block_stride = 1):
        super(ResNetBlock, self).__init__()

        self.downsample = None
        
        if first_block_stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels= input_chan, out_channels=output_chan, kernel_size=1, stride = first_block_stride),
                nn.BatchNorm2d(output_chan)
                )
        
        self.blocks = nn.ModuleList()
        self.blocks.append(nn.Conv2d(input_chan, output_chan, kernel_size, first_block_stride, padding = 1 if first_block_stride != 1 else 'same'))
        self.blocks.append(nn.BatchNorm2d(output_chan))
        self.blocks.append(nn.ReLU())
        
        for i in range(layers -1):
            self.blocks.append(nn.Conv2d(output_chan, output_chan, kernel_size, 1, padding='same'))
            self.blocks.append(nn.BatchNorm2d(output_chan))
            self.blocks.append(nn.ReLU())
    
    def forward(self, x):
        residual = x
        
        for module in self.blocks:
            x = module.forward(x)

        if self.downsample != None:
            residual = self.downsample(residual)

        x = x + residual
        x = torch.nn.functional.relu(x, inplace=True)
        return x
    


class PatchEmbedding(nn.Module):
    def __init__(self, input_chan, embedding_dim = 64, kernel_size = 7, padding=3):
        super(PatchEmbedding, self).__init__()

        self.embedding_dim = embedding_dim

        self.conv1 = nn.Conv2d(input_chan, self.embedding_dim, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(self.embedding_dim)
        self.relu = nn.ReLU(inplace=True)
    

    def forward(self, x):
        with torch.autocast(device_type="cuda", enabled=False):        
            x = self.conv1(x)
            x = self.bn(x)
            x = self.relu(x)
                
        return x

class PatchEmbed(nn.Module):
    def __init__(self, input_chan, embedding_dim = 64, patch_size = 4):
        super(PatchEmbed, self).__init__()

        self.embedding_dim = embedding_dim

        self.conv1 = nn.Conv2d(input_chan, self.embedding_dim, kernel_size=patch_size, stride=patch_size, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(self.embedding_dim)
        self.relu = nn.ReLU(inplace=True)
    

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
            
        return x