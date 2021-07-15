import torch 
import torch.nn as nn
import torch.nn.functional as F 

def passthrough(x):
    return x
    
def activation(prelu, num_channels):
    if prelu:
        return nn.PReLU(num_channels)
    else:
        return nn.ELU()
        
class ConvLayer(nn.Module):
    def __init__(self, num_channels, prelu, dilation):
        super(ConvLayer, self).__init__()
        self.relu1 = activation(prelu, num_channels)
        self.conv1 = nn.Conv3d(num_channels, num_channels, kernel_size = 5, padding = dilation * 2, dilation = dilation, padding_mode = 'zeros')
        
    def forward(self, x):
        out = self.relu1(self.conv1(x))
        return out
        
def ConvLayers(num_channels, num_layers, prelu, dilation):
    layers = []
    for _ in range(num_layers):
        layers.append(ConvLayer(num_channels, prelu, dilation))
    return nn.Sequential(*layers)
    
class InputTransition(nn.Module):
    def __init__(self, out_channels, prelu, dilation = 1, attention_module = False):
        super(InputTransition, self).__init__()
        self.attention_module = attention_module
        self.conv1 = nn.Conv3d(1, out_channels, kernel_size = 5, padding = dilation * 2, dilation = dilation, padding_mode = 'zeros')
        self.relu1 = activation(prelu, out_channels)
        if self.attention_module:
          self.att_module = AttentionModule(out_channels)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        if self.attention_module:
          out = self.att_module(out)
        out = torch.add(out, x)
        return out
        
class DownTransition(nn.Module):
    def __init__(self, in_channels, num_layers, prelu, dilation = 1, attention_module = False):
        super(DownTransition, self).__init__()
        out_channels = 2 * in_channels
        self.attention_module = attention_module
        self.down_conv = nn.Conv3d(in_channels, out_channels, kernel_size = 2, stride = 2)
        self.pass1 = passthrough
        self.relu1 = activation(prelu, out_channels)
        self.ops = ConvLayers(out_channels, num_layers, prelu, dilation)
        if self.attention_module:
          self.att_module = AttentionModule(out_channels)

    def forward(self, x):
        down = self.relu1(self.down_conv(x))
        out = self.pass1(down)
        out = self.ops(out)
        if self.attention_module:
          out = self.att_module(out)
        out = torch.add(out, down)
        return out
        
class UpTransition(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, prelu, dilation = 1):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(in_channels, out_channels // 2, kernel_size = 2, stride = 2)
        self.pass1 = passthrough
        self.relu1 = activation(prelu, out_channels // 2)
        self.relu2 = activation(prelu, out_channels)
        self.ops = ConvLayers(out_channels, num_layers, prelu, dilation)
        
    def forward(self, x, skip_connection):
        out = self.pass1(x)
        skip = skip_connection
        out = self.relu1(self.up_conv(out))
        out_cat = torch.cat((out, skip), 1)
        out = self.ops(out_cat)
        out = torch.add(out, out_cat)
        return out
        
class OutputTransition(nn.Module):
    def __init__(self, in_channels, prelu):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 1, kernel_size = 1)
        self.relu1 = activation(prelu, 1)
        
    def forward(self, x):
        out = self.relu1(self.conv1(x))
        return out

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, in_channels, reduction_ratio = 4):
        super(ChannelGate, self).__init__()
        #   
        self.channel_gate_layers = []
        self.channel_gate_layers.append(Flatten())
        self.channel_gate_layers.append(nn.Linear(in_channels, in_channels // reduction_ratio))
        self.channel_gate_layers.append(nn.BatchNorm1d(in_channels // reduction_ratio))
        self.channel_gate_layers.append(nn.ReLU())
        self.channel_gate_layers.append(nn.Linear(in_channels // reduction_ratio,in_channels))
        self.channel_gate = nn.Sequential(*self.channel_gate_layers)

    def forward(self, x):
        avg_pool_out = F.avg_pool3d(x, kernel_size = (x.size(2),x.size(3),x.size(4)))
        x_out = self.channel_gate(avg_pool_out)
        c_att = torch.sigmoid(x_out).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
        out_att = x * c_att
        return x + out_att

class SpatialGate(nn.Module):
    def __init__(self, in_channels, reduction_ratio = 4, dilated_conv_layers = 2, dilation = 4):
        super(SpatialGate, self).__init__()
        self.spatial_gate_layers = []
        self.spatial_gate_layers.append(nn.Conv3d(in_channels, in_channels // reduction_ratio, kernel_size = 1))
        self.spatial_gate_layers.append(nn.BatchNorm3d(in_channels // reduction_ratio))
        self.spatial_gate_layers.append(nn.ReLU())
        for i in range(dilated_conv_layers):
            self.spatial_gate_layers.append(nn.Conv3d(in_channels // reduction_ratio, in_channels // reduction_ratio, kernel_size = 3, 
                                            padding = dilation, dilation = dilation))
            self.spatial_gate_layers.append(nn.BatchNorm3d(in_channels // reduction_ratio))
            self.spatial_gate_layers.append(nn.ReLU())
        self.spatial_gate_layers.append(nn.Conv3d(in_channels // reduction_ratio, 1, kernel_size = 1))
        self.spatial_gate = nn.Sequential(*self.spatial_gate_layers)
    
    def forward(self, x):
        x_out =  self.spatial_gate(x)
        s_att = torch.sigmoid(x_out).expand_as(x)
        out_att = x * s_att
        return x + out_att

class AttentionModule(nn.Module):
    """
    Attention mechansim specified in "Park, J. et al. (2019) ‘BAM: Bottleneck attention module’, British Machine Vision Conference 2018, BMVC 2018."
    Note: This object has differences in that it applies sequential Channel and Spatial attention compared to parallel and applies to 5D Tensors(NXCXDXHXW)
    """
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.channel_att = ChannelGate(in_channels)
        self.spatial_att = SpatialGate(in_channels)
    
    def forward(self,x):
        x_out = self.channel_att(x)
        x_out = self.spatial_att(x_out)
        return x + x_out
        
class VNet(nn.Module):
    """
    VNet architecture specified in "Milletari, F., Navab, N. and Ahmadi, S. A. (2016) ‘V-Net: Fully convolutional neural networks for volumetric 
    medical image segmentation’, Proceedings - 2016 4th International Conference on 3D Vision, 3DV 2016, pp. 565–571. doi: 10.1109/3DV.2016.79."
    Note: Consists of additions for dilation and attention modules
    """
    def __init__(self, num_layers = 1, prelu = True, dilation = 1, attention_module = False):
        super(VNet, self).__init__()
        self.input_transition = InputTransition(8, prelu, dilation, attention_module)
        self.down_transition16 = DownTransition(8, num_layers, prelu, dilation, attention_module)
        self.down_transition32 = DownTransition(16, num_layers, prelu, dilation, attention_module)
        self.down_transition64 = DownTransition(32, num_layers + 1, prelu, dilation, attention_module)
        self.down_transition128 = DownTransition(64, num_layers + 1, prelu, dilation, attention_module)
        self.up_transition128 = UpTransition(128, 128, num_layers + 1, prelu)
        self.up_transition64 = UpTransition(128, 64, num_layers + 1, prelu)
        self.up_transition32 = UpTransition(64, 32, num_layers, prelu)
        self.up_transition16 = UpTransition(32, 16, num_layers, prelu)
        self.out_transition = OutputTransition(16, prelu)
         
    def forward(self, x):
        out8 = self.input_transition(x)
        out16 = self.down_transition16(out8)
        out32 = self.down_transition32(out16)
        out64 = self.down_transition64(out32)
        out128 = self.down_transition128(out64)
        out = self.up_transition128(out128, out64)
        out = self.up_transition64(out, out32)
        out = self.up_transition32(out, out16)
        out = self.up_transition16(out, out8)
        out = self.out_transition(out)
        return out