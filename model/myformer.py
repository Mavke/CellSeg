import copy
import numpy as np

import torch  
from torch import nn
from torchvision.models import resnet

from model.modulesh2former import MultiScaleFeatures, ChannelAttention, PatchMerg
from model.modules import ConvBlock, SkipConnectionDepthwise, PatchEmbedding, PatchEmbed
from model import swin_transformer_v2
from model.decoder import SegmentationDecoder

class MyFormer(nn.Module):
    """_
    The implementation of the novel model architecture based on H2Former and Swin-Transformer and a decoding path similar to the CellVit model.

    """

    def __init__(self, image_size, num_classes, channel_dim = 3, channel_per_stage = [64, 128, 256, 512], window_size=16, pretrained_transformer = None, drop_out_rate = 0.2):
        super(MyFormer,self).__init__()
        
        self.num_classes = num_classes
        self.drop_out_rate = drop_out_rate
        channel_per_stage = [64, 128, 256, 512]
        attention_channels = 96 #128# 
        attention_path_channels = [attention_channels, attention_channels*2, attention_channels*4,attention_channels*8]

        #patch parititon and embedding
        self.convStem = PatchEmbedding(channel_dim, kernel_size=3, padding=1)

        # self.convStem = PatchEmbedding(channel_dim, kernel_size=7, padding=3)
        self.patch_embed = PatchEmbed(3, attention_channels, patch_size=2)
        self.input_convolve = nn.Sequential(ConvBlock(input_channels=3, output_channels=attention_channels // 2, drop_out_rate=self.drop_out_rate), ConvBlock(input_channels=attention_channels // 2, output_channels=attention_channels // 2, drop_out_rate=self.drop_out_rate))

        self._norm_layer = nn.BatchNorm2d
        self.mlp_ratio = 4.0
        self.window_size = window_size

        
        temp_resnet = resnet.resnet18()
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = copy.deepcopy(temp_resnet.layer1)
        self.layer2 = copy.deepcopy(temp_resnet.layer2)
        self.layer3 = copy.deepcopy(temp_resnet.layer3)
        self.layer4 = copy.deepcopy(temp_resnet.layer4)
        
        del temp_resnet

        self.decoder = SegmentationDecoder(attention_channels=attention_path_channels, n_classes=2, drop_out_rate=self.drop_out_rate)
        self.decoder_distance = SegmentationDecoder(attention_channels=attention_path_channels, n_classes=2, drop_out_rate=self.drop_out_rate)
        self.decoder_classes = SegmentationDecoder(attention_channels=attention_path_channels, n_classes=num_classes, drop_out_rate=self.drop_out_rate)

        #modules for multi-scale attention features
        self.msca_moduel1 = nn.Sequential(MultiScaleFeatures([4,8,16], [2,4,4], channel_per_stage[0], channel_per_stage[0], drop_out_rate = self.drop_out_rate), ChannelAttention(name='msca1'))
        self.msca_moduel2 = nn.Sequential(MultiScaleFeatures([2,4], [2,2], channel_per_stage[0], channel_per_stage[1], drop_out_rate = self.drop_out_rate),ChannelAttention())
        self.msca_moduel3 = nn.Sequential(MultiScaleFeatures([2,4], [2,2], channel_per_stage[1], channel_per_stage[2], drop_out_rate = self.drop_out_rate),ChannelAttention())
        self.msca_moduel4 = nn.Sequential(MultiScaleFeatures([2,4], [2,2], channel_per_stage[2], channel_per_stage[3], drop_out_rate = self.drop_out_rate), ChannelAttention())


        self.skip_connection1 = SkipConnectionDepthwise(channel_per_stage[0], attention_path_channels[0])
        self.skip_connection2 = SkipConnectionDepthwise(channel_per_stage[1], attention_path_channels[1])
        self.skip_connection3 = SkipConnectionDepthwise(channel_per_stage[2], attention_path_channels[2])
        self.skip_connection4 = SkipConnectionDepthwise(channel_per_stage[3], attention_path_channels[3])




        self.layers = nn.ModuleList()
        self.n_layers = 4
       
        #initialize Swin Transformer
        self.layers = nn.ModuleList()
        depths = [2, 2, 6, 2]
        heads=[3, 6, 12, 24]
        # depths = [2,2,18,2]
        # heads = [4,8,16,32]
        drop_path_rate = 0.1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        for i_layer in range(self.n_layers):
            layer = swin_transformer_v2.BasicLayer(dim=int(attention_channels * 2 ** i_layer),
                                depth=depths[i_layer],num_heads=heads[i_layer],
                            input_resolution=(image_size[0] // 2**(i_layer+1), image_size[1] // 2**(i_layer+1)),
                            window_size=self.window_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=True,
                
                            drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                            norm_layer=nn.LayerNorm,
                            downsample= None)
            self.layers.append(layer)
        self.patch_merge_layers = nn.ModuleList()

        for i in range(self.n_layers-1):
            self.patch_merge_layers.append(PatchMerg(attention_path_channels[i], attention_path_channels[i+1]))

    def unfreeze(self):
        self.freeze_pretrained(True)

    def freeze_pretrained(self, req_grad = False):
        self._freeze_layer(self.layer1,req_grad)
        self._freeze_layer(self.layer2,req_grad)
        self._freeze_layer(self.layer3,req_grad)
        self._freeze_layer(self.layer4,req_grad)
        
        for layer in self.layers:
            self._freeze_layer(layer,req_grad)

    def _freeze_layer(self, layer, req_grad = False):
        for param in layer.parameters():
            param.requires_grad = req_grad

    def check_for_nan(self,x, s):
        if torch.isnan(x).any():
            print('Found nan value in forward pass' + s)
            print(torch.max(x), torch.min(x))
            return True
        if (1^ torch.isfinite(x)).any():
            print('Found not finite value in forward pass' +s)
            print(torch.max(x), torch.min(x))
            return True
        
        return False
   
    def forward(self, x):
        self.check_for_nan(x, "input")
        intermediate_results = []
        input_conv = self.input_convolve(x)
        intermediate_results.append(input_conv)
        patch = self.patch_embed(x)

        x = self.convStem(x)
        self.check_for_nan(x, "conv stem")
        local_features = self.maxpool(x)

        local_features = self.layer1(local_features)
        msca_features = self.msca_moduel1(x)
        features_left = local_features + msca_features


        x = self.skip_connection1(features_left)
        x = patch + x

        x = x.flatten(2).transpose(1, 2)

        x = self.layers[0](x)
        x = self.seqToImage(x)

        intermediate_results.append(x)
        
        x = self.patch_merge_layers[0](x)

        local_features = self.layer2(features_left)

        msca_features = self.msca_moduel2(features_left)

        features_left = local_features + msca_features

        x = self.skip_connection2(features_left) + x
        x = x.flatten(2).transpose(1, 2)
        x = self.layers[1](x)
        x = self.seqToImage(x)
        
        intermediate_results.append(x)
        
        x = self.patch_merge_layers[1](x)

        local_features = self.layer3(features_left).requires_grad_()
        msca_features = self.msca_moduel3(features_left)

        features_left = local_features + msca_features

        x = self.skip_connection3(features_left) + x
        x = x.flatten(2).transpose(1, 2)

        x = self.layers[2](x)
        x = self.seqToImage(x)
        intermediate_results.append(x)

        x = self.patch_merge_layers[2](x)

        local_features = self.layer4(features_left)
        msca_features = self.msca_moduel4(features_left)
        features_left = local_features + msca_features

        x = self.skip_connection4(features_left) + x
        x = x.flatten(2).transpose(1, 2)

        x = self.layers[3](x)

        swin_x = self.seqToImage(x)

        result_seg = self.decoder(copy.copy(intermediate_results), swin_x)
        result_distance = self.decoder_distance(copy.copy(intermediate_results), swin_x)
        result_class = self.decoder_classes(copy.copy(intermediate_results), swin_x)

        result = {}
        result['nuclei_binary_map'] = result_seg
        result['hv_map'] = result_distance
        result['class'] = result_class
        
        return result


    def seqToImage(self, x):
        B, L, C = x.shape
        x = x.view(B, int(np.sqrt(L)), int(np.sqrt(L)), C).permute(0,3,1, 2)
        return x