import torch.nn as nn
from torch.nn.functional import normalize
import torch

from torch.nn.parameter import Parameter
from TransformerViewFusion import *;
from MLPClassifier import *;

# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)

# Decoder
class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)

# SCMVC Network
class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, class_num,device):
        super(Network, self).__init__()
        self.view = view
        self.calss_num = class_num
        self.encoders = []
        self.decoders = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], feature_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

        # global features fusion layer
        self.feature_fusion_module = nn.Sequential(
            nn.Linear(self.view * feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, high_feature_dim)
        )

        # view-consensus features learning layer
        #self.common_information_module = nn.Sequential(
        #    nn.Linear(feature_dim, high_feature_dim)
        #)
        self.common_information_modules = nn.ModuleList([
            nn.Sequential(nn.Linear(feature_dim, high_feature_dim))
            for _ in range(view)
        ])



        # 视图信息融合模块
        self.view_fusion=TransformerViewFusion(feature_dim
                                               , embed_dim=high_feature_dim
                                               , num_heads=8
                                               , dropout=0.0
                                               , num_layers=2
                                               , num_views=view)
        
        
        # 对每个视角的特征，使用视角分类头
        self.view_classifier = MLPClassifier(in_dim=high_feature_dim, 
                                                num_classes=class_num,
                                                hidden_dim=256, 
                                                dropout=0.0
                                             )

        # 对融合特征 H，使用融合分类头
        self.global_classifier = MLPClassifier(in_dim=high_feature_dim,
                                                num_classes=class_num,
                                                hidden_dim=256, 
                                                dropout=0.0
                                            )


        
    # global feature fusion
    def feature_fusion(self, zs, zs_gradient):
        input = torch.cat(zs, dim=1) if zs_gradient else torch.cat(zs, dim=1).detach()
        return normalize(self.feature_fusion_module(input),dim=1)

    def forward(self, xs, zs_gradient=True):
        rs = []
        xrs = []
        zs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            xr = self.decoders[v](z)
            r = normalize(self.common_information_modules[v](z),dim=1)

            rs.append(r)
            zs.append(z)
            xrs.append(xr)

        #H = self.feature_fusion(zs,zs_gradient)
        H,weight = self.view_fusion(zs)
        H = normalize(H,dim=1)
        # xrs重建损失，zs编码器输出，rs 翻译器的到该视角特征对全局的描述，H全局特征
        return xrs,zs,rs,H,weight
    # 返回视角概率和全局概率
    def calculate_s(self, xs):
        ss = []
        zs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            zs.append(z)
            # 学习到的一致性特征，并且经过了归一化处理
            r = normalize(self.common_information_modules[v](z),dim=1)
            # 将每个视角学习到的一致性特征交给分类器进行处理的到分类结果
            s = self.view_classifier(r)
            ss.append(s)
        hp,_ = self.view_fusion(zs)
        hp = normalize(hp,dim=1)
        hp = self.global_classifier(hp)
        return ss,hp
    