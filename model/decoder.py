import torch
from model.modules import *


class SegmentationDecoder(nn.Module):
    def __init__(self, attention_channels: list, drop_out_rate, conv_blocks: int = 2, n_classes: int = 2, decoder_channels: list = [64, 128, 256, 512]):
        
        super(SegmentationDecoder, self).__init__()
        
        self.n_classes = n_classes
        self.attention_channels = attention_channels
        self.decoder_channels = decoder_channels

        self.conv_skip_features = nn.ModuleList()

        self.upsampling_stages = nn.ModuleList(
            (
                DeConvBlock(attention_channels[3], attention_channels[2], drop_out_rate=drop_out_rate),
                DeConvBlock(attention_channels[2], attention_channels[1], drop_out_rate=drop_out_rate),
                DeConvBlock(attention_channels[1], attention_channels[0], drop_out_rate=drop_out_rate),
                DeConvBlock(attention_channels[0], attention_channels[0] // 2, drop_out_rate=drop_out_rate)
            )
        )

                
        self.decoder_stages = nn.ModuleList()
        
        conv_decoder_stage_0 = nn.Sequential(
                ConvBlock(attention_channels[2]*2, attention_channels[2], drop_out_rate=drop_out_rate), 
                ConvBlock(attention_channels[2],attention_channels[2], drop_out_rate=drop_out_rate),
                ConvBlock(attention_channels[2],attention_channels[2], drop_out_rate=drop_out_rate)
            )        
        self.decoder_stages.append(conv_decoder_stage_0)

        conv_decoder_stage_1 = nn.Sequential(
                ConvBlock(attention_channels[1]*2, attention_channels[1], drop_out_rate=drop_out_rate), 
                ConvBlock(attention_channels[1],attention_channels[1], drop_out_rate=drop_out_rate)
        ) 
        self.decoder_stages.append(conv_decoder_stage_1)
        
        conv_decoder_stage_2 = nn.Sequential(
                ConvBlock(attention_channels[0]*2, attention_channels[0], drop_out_rate=drop_out_rate), 
                ConvBlock(attention_channels[0],attention_channels[0], drop_out_rate=drop_out_rate)
        )
        self.decoder_stages.append(conv_decoder_stage_2)

        conv_decoder_stage_4 = nn.Sequential(
                ConvBlock((attention_channels[0]// 2)*2, attention_channels[0]// 2, drop_out_rate=drop_out_rate), 
                ConvBlock(attention_channels[0]// 2, attention_channels[0]// 2, drop_out_rate=drop_out_rate),
                nn.Conv2d(attention_channels[0]// 2, n_classes, kernel_size=1, stride=1, padding=0)
        )
        self.decoder_stages.append(conv_decoder_stage_4)

    

    def forward(self, skip_connections: list, x: torch.Tensor):
        
        for i, upsample in enumerate(self.upsampling_stages):
            if i == 3:
                break
            skip_features = skip_connections.pop()
            x = upsample(x)
            x = torch.cat((x, skip_features), dim=1)
            x = self.decoder_stages[i](x)

        x = self.upsampling_stages[-1](x)
        input_image = skip_connections.pop()

        x = torch.cat((x, input_image), dim=1)

        x = self.decoder_stages[-1](x)

        return x