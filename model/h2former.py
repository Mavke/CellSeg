import torch
from torch import nn
import copy
import numpy as np


from model.modulesh2former import *
from torchvision.models import resnet
from model import swin_transformer_v2
from model import swin_transformer

class H2Former(nn.Module):
    """ Implements the H2Former model described as in the paper "H2Former: An Efficient Hierarchical Hybrid Transformer for Medical Image Segmentation" by Zhang et al.
    """
    def __init__(self, image_size, num_classes, channel_dim = 3, channel_per_stage = [64, 128, 256, 512], window_size=16, pretrained_transformer = None):
        super(H2Former,self).__init__()
        
        self.num_classes = num_classes
        channel_per_stage = [64, 128, 256, 512]
        attention_channels = 96
        attention_path_channels = [attention_channels, attention_channels*2, attention_channels*4,attention_channels*8]

        #patch parititon and embedding
        self.convStem = PatchEmbedding(channel_dim, patch_size=2)

        self._norm_layer = nn.BatchNorm2d
        self.mlp_ratio = 4.0
        self.window_size = window_size

        
        t = resnet.resnet34()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = copy.deepcopy(t.layer1)
        self.layer2 = copy.deepcopy(t.layer2)
        self.layer3 = copy.deepcopy(t.layer3)
        self.layer4 = copy.deepcopy(t.layer4)
        
        del t

        #modules for multi-scale attention features
        self.msca_moduel1 = nn.Sequential(MultiScaleFeatures([4,8,16,32], [2,4,8,8], channel_per_stage[0], channel_per_stage[0]), ChannelAttention())
        self.msca_moduel2 = nn.Sequential(MultiScaleFeatures([2,4], [2,2], channel_per_stage[0], channel_per_stage[1]),ChannelAttention())
        self.msca_moduel3 = nn.Sequential(MultiScaleFeatures([2,4], [2,2], channel_per_stage[1], channel_per_stage[2]),ChannelAttention())
        self.msca_moduel4 = nn.Sequential(MultiScaleFeatures([2,4], [2,2], channel_per_stage[2], channel_per_stage[3]), ChannelAttention())

        self.skip_connection1 = SkipConnection(channel_per_stage[0], attention_path_channels[0])
        self.skip_connection2 = SkipConnection(channel_per_stage[1], attention_path_channels[1])
        self.skip_connection3 = SkipConnection(channel_per_stage[2], attention_path_channels[2])
        self.skip_connection4 = SkipConnection(channel_per_stage[3], attention_path_channels[3])
        
        self.layers = nn.ModuleList()
        self.n_layers = 4

        if pretrained_transformer is None:
            depths = [2, 2, 2, 2]
            num_heads = [2, 4, 8, 16]
            drop_path_rate = 0.1
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]


            embedding_dim = channel_per_stage[0]
            for i in range(4):
                swin_layer = swin_transformer.BasicLayer(dim=int(embedding_dim * 2 ** i),
                    input_resolution=(image_size[0] // 2**(i+2), image_size[1] // 2**(i+2)), depth=depths[i],num_heads=num_heads[i],window_size=self.window_size,mlp_ratio=self.mlp_ratio,
                    qkv_bias=True, qk_scale=None,drop=0.0, attn_drop=0.0,
                    drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                    norm_layer=nn.LayerNorm,downsample= None,use_checkpoint=False)
                self.layers.append(swin_layer)
        else:
            self.layers = nn.ModuleList()
            depths = [2, 2, 6, 2]
            heads=[3, 6, 12, 24]
            drop_path_rate = 0.1
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
            for i_layer in range(self.n_layers):
                layer = swin_transformer_v2.BasicLayer(dim=int(96 * 2 ** i_layer),
                                   depth=depths[i_layer],num_heads=heads[i_layer],
                                input_resolution=(image_size[0] // 2**(i_layer+2), image_size[1] // 2**(i_layer+2)),
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


        
        self.patch_expanding1 = PatchExpanding(attention_path_channels[3])
        self.patch_expanding2 = PatchExpanding(attention_path_channels[2])
        self.patch_expanding3 = PatchExpanding(attention_path_channels[1])

        self.decode_conv1 = DecoderConvolution(attention_path_channels[2] + attention_path_channels[3], attention_path_channels[2])
        self.decode_conv2 = DecoderConvolution(attention_path_channels[1] + attention_path_channels[2], attention_path_channels[1])
        self.decode_conv3 = DecoderConvolution(attention_path_channels[0] + attention_path_channels[1], attention_path_channels[0])
        self.decode_conv4 = PredictionHead(attention_path_channels[0], self.num_classes)

   
    def forward(self, x):

        intermediate_results = []

        x = self.convStem(x)
        self.check_for_nan(x, "conv stem")
        local_features = self.maxpool(x)

        local_features = self.layer1(local_features)
        msca_features = self.msca_moduel1(x)
        features_left = local_features + msca_features

        x = self.skip_connection1(features_left)
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

        local_features = self.layer3(features_left)
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
        encoder_res = intermediate_results.pop()
        x = self.patch_expanding1(swin_x.float(), encoder_res.shape)    
        x = torch.cat((x, encoder_res), dim=1)
        x = self.decode_conv1(x)
        
        encoder_res =intermediate_results.pop()
        x = self.patch_expanding2(x, encoder_res.shape)       
        x = torch.cat((x, encoder_res), dim=1)
        x = self.decode_conv2(x)

        encoder_res =intermediate_results.pop()
        x = self.patch_expanding3(x, encoder_res.shape)       
        x = torch.cat((x, encoder_res), dim=1)
        x = self.decode_conv3(x)

        out = self.decode_conv4(x)
        
        return out


    def seqToImage(self, x):
        B, L, C = x.shape
        x = x.view(B, int(np.sqrt(L)), int(np.sqrt(L)), C).permute(0,3,1, 2)
        return x