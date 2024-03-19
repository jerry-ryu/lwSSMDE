from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from layers import *
from timm.models.layers import trunc_normal_


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, depth_bins=64, scales=range(4), num_output_channels=1, min_val = 0.001, max_val = 80):
        super().__init__()

        self.num_output_channels = num_output_channels
        self.upsample_mode = 'bilinear'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_f = [x //2 for x in self.num_ch_enc]
        self.num_ch_cum = [sum(self.num_ch_enc[i:]) for i in range(len(self.num_ch_enc))]

        self.g_ccost_volume = Generate_cost_volume()
        self.g_energy_map = Generate_energy_map()
        
        # decoder
        self.convs = torch.nn.ModuleDict()
        for i in range(2, -1, -1):
            
            self.convs[f"f_upconv_{i}_0"] = ConvBlock(self.num_ch_f[i], self.num_ch_f[i])
            self.convs[f"f_upconv_{i}_1"] = ConvBlock(self.num_ch_f[i], self.num_ch_f[i])
            
            
            self.convs[f"p_upconv_{i}_0"] = ConvBlock(self.num_ch_cum[i], self.num_ch_cum[i]//2)
            self.convs[f"p_upconv_{i}_1"] = ConvBlock(self.num_ch_cum[i]//2 , depth_bins)
            
            if i!=0:
                self.convs[f"1x1_conv_{i}"] = nn.Linear(self.num_ch_f[i], self.num_ch_f[i-1])
            
        # self.bins_regressor = nn.Sequential(nn.Linear(self.num_ch_cum[0] * self.num_ch_f[0], 8*self.num_ch_cum[0]),
        #                                nn.LeakyReLU(),
        #                                nn.Linear(8*self.num_ch_cum[0], 16*16),
        #                                nn.LeakyReLU(),
        #                                nn.Linear(16*16, depth_bins))
        
        self.bin_compressor = torch.nn.ModuleDict()
        self.bin_compressor["local_compressor"] = nn.Linear(self.num_ch_f[0],self.num_ch_f[0]//4)
        self.bin_compressor["global_compressor"] =  nn.Linear(self.num_ch_cum[0],self.num_ch_cum[0]//4)
        self.bins_regressor = nn.Sequential(nn.Linear((self.num_ch_f[0]//4) * (self.num_ch_cum[0]//4), 2*(self.num_ch_cum[0]//4)),
                                            nn.LeakyReLU(),
                                            nn.Linear(2*(self.num_ch_cum[0]//4), depth_bins))
        
        self.min_val = min_val
        self.max_val = max_val
        

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, input_features, conv_features):
        self.outputs = {}
        depth_probabilities = []
        
        for i in range(2, -1, -1):
            x =  input_features[i]
            y =  conv_features[i]
            ccost_volume = self.g_ccost_volume(x,y)
            
            if i !=2:
                ccost_volume = torch.cat([ccost_volume, post_ccost],1)
                
            y = self.convs[f"f_upconv_{i}_0"](y)
            y = upsample(y, mode='bilinear')
            y = self.convs[f"f_upconv_{i}_1"](y)
            
            energy_map = self.g_energy_map(ccost_volume, y)
            
            depth_probability = self.convs[f"p_upconv_{i}_0"](energy_map)
            depth_probability = upsample(depth_probability, mode='bilinear')
            depth_probability = self.convs[f"p_upconv_{i}_1"](depth_probability)
            depth_probabilities.append(depth_probability)
            
            if i != 0:
                post_ccost = self.convs[f"1x1_conv_{i}"](ccost_volume)
        
        
        
        ccost_volume = self.bin_compressor["local_compressor"](ccost_volume)
        ccost_volume = self.bin_compressor["global_compressor"](ccost_volume.permute(0,2,1))
        bs, L, G = ccost_volume.shape
        bins = self.bins_regressor(ccost_volume.view(bs, G*L))
        
        bins = torch.relu(bins)
        eps = 0.1
        bins = bins + eps
        bins = bins /bins.sum(dim=1, keepdim=True)
        
        bin_widths = (self.max_val - self.min_val) * bins
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val)
        bin_edges = torch.cumsum(bin_widths, dim=1)
        
        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1)
        
        for i in self.scales:
            pred = torch.sum(depth_probabilities[2-i] * centers, dim=1, keepdim=True)
            self.outputs[("disp", i)] = pred

        return self.outputs

