import torch
from torch import nn

#from thop import profile
#from thop import clever_format


"""
1 : nn.Conv3d(in_planes,out_planes,kernel_size=1,stride=stride,bias=False)
3 : nn.Conv3d(in_planes,out_planes,kernel_size=3,stride=stride,padding=1,bias=False)
"""
def get_inplanes():
    return [128, 256, 512, 1024]


""" Constructor
Args:
    features: input channel dimensionality.
    M: the number of branchs.
    G: num of convolution groups.
    r: the ratio for compute d, the length of z.
    stride: stride, default 1.
    L: the minimum dim of the vector z in paper, default 32.
"""
class SKConv(nn.Module):
    # features = mid_planes
    def __init__(self, mid_planes, M=2, G=32, r=16, stride=1 ,L=32):
        super(SKConv, self).__init__()
        d = max(int(mid_planes/r), L)
        self.M = M
        self.mid_planes = mid_planes
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv3d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1+i, dilation=1+i, groups=G, bias=False),
                nn.BatchNorm3d(mid_planes),
                nn.ReLU(inplace=False)
            ))
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(nn.Conv3d(mid_planes, d, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm3d(d),
                                nn.ReLU(inplace=False))
        self.fcs = nn.Conv3d(d, mid_planes * M, kernel_size=1, stride=1, bias=False)
        """
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                 nn.Conv3d(d, mid_planes, kernel_size=1, stride=1)
            )
        """
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        batch_size = x.shape[0]


        feats = [conv(x) for conv in self.convs]   
        #print(feats[0].shape)   

        #feats = torch.cat(feats, dim=1)
        #feats = feats.view(batch_size, self.M, self.mid_planes, -1,-1,-1)
        
        #feats_U = torch.sum(feats, dim=1)
        feats_U = feats[0] + feats[1]   #hard-coding (M=2)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)

        #attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = self.fcs(feats_Z)
        #attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.M, self.mid_planes, 1, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        
        #---------- Selection
        #feats_V = torch.sum(torch.FloatTensor(feats) * attention_vectors, dim=1)
        attention_vectors = list(attention_vectors.chunk(self.M, dim=1))  # split to a and b   chunk为pytorch方法，将tensor按照指定维度切分成 几个tensor块
        attention_vectors = list(map(lambda x: torch.squeeze(x,dim=1), attention_vectors))
        feats_V = list(map(lambda x, y: x * y, attention_vectors, feats))
        feats_V = feats_V[0] + feats_V[1]
        #print(feats_V.shape)

        return feats_V

""" Constructor
Similar to ReNeXtBottleneck
Args:
    in_planes: input channel dimensionality.
    planes: output channel dimensionality.
    cardinality : resnext cardinality
    M: the number of branchs.
    G: num of convolution groups.
    r: the ratio for compute d, the length of z.
    mid_features: the channle dim of the middle conv with stride not 1, default out_features/2.
    stride: stride.
    L: the minimum dim of the vector z in paper.
"""
class SKUnit(nn.Module):
    expansion = 2

    def __init__(self, in_planes, planes, cardinality, M=2, G=32, r=16, stride=1, L=32):
        super(SKUnit, self).__init__()
        
        mid_planes = cardinality * planes // 32

        # conv1
        self.conv1 = nn.Sequential(
                        nn.Conv3d(in_planes, mid_planes,
                                kernel_size=1,
                                stride=1, bias=False),
                        nn.BatchNorm3d(mid_planes),
                        nn.ReLU(inplace=True)
        )
        # conv2
        self.conv2_sk = SKConv(mid_planes, M=M, G=G, r=r, stride=stride, L=L)
        
        # conv3
        self.conv3 = nn.Sequential(
                        nn.Conv3d(mid_planes, planes * self.expansion,
                                kernel_size=1,
                                stride=1, bias=False),
                        nn.BatchNorm3d(planes * self.expansion), # idk if i's needed or not...
        )

        if in_planes == (planes* self.expansion): # when dim not change, input_features could be added diectly to out
            self.shortcut = nn.Sequential()
        else: # when dim not change, input_features should also change dim to be added to out
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, (planes * self.expansion), 1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.conv2_sk(out)
        out = self.conv3(out)
        
        #print(out.shape, residual.shape)
        return self.relu(out + self.shortcut(residual))

# Model
class SKNet(nn.Module):
    def __init__(self,
                layers, #eg. [2,2,2,2] = nums_block_list
                block_inplanes, # from get_inplane()
                n_input_channels=3,
                
                conv1_t_size=7, # to handle temporal stride at 'basic_conv()'
                conv1_t_stride=1, # to handle temporal stride at 'basic_conv()'

                no_max_pool=False,
                shortcut_type='B',
                cardinality=32,
                n_classes=101):
        super(SKNet, self).__init__()

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        # first layer (7*7, 64, stride 2)
        self.basic_conv = nn.Sequential(
            nn.Conv3d(n_input_channels,
                self.in_planes,
                kernel_size=(conv1_t_size, 7, 7),
                stride=(conv1_t_stride, 2, 2),
                padding=(conv1_t_size // 2, 3, 3),
                bias=False),
            nn.BatchNorm3d(self.in_planes),
            nn.ReLU(inplace=True),
        )
        
        # maxpool (after first layer)
        # 3*3, stride 2
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # 4 stages --------------------------------
        #img 56*56

        self.stage_1 = self._make_layer(block_inplanes[0], cardinality,
                                        nums_block=layers[0],
                                        stride=1)
        #img 28*28
        self.stage_2 = self._make_layer(block_inplanes[1], cardinality,
                                        nums_block=layers[1],
                                        stride=2)                                
        #img 14*14
        self.stage_3 = self._make_layer(block_inplanes[2], cardinality,
                                        nums_block=layers[2],
                                        stride=2)
        #img 7*7
        self.stage_4 = self._make_layer(block_inplanes[3], cardinality,
                                        nums_block=layers[3],
                                        stride=2)
        #img 1*1
        # GAP 7*7, fc and softmax applied
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Linear(2048, n_classes)

        #--- init params here
        #-------------
        
    #_, planes, blocks, shortcut_type
    def _make_layer(self, planes, cardinality, nums_block, stride=1):
                
        layers=[SKUnit(self.in_planes, planes, cardinality, stride=stride)]
        self.in_planes = planes * SKUnit.expansion 
        for _ in range(1,nums_block):
            layers.append(SKUnit(self.in_planes, planes, cardinality))
        return nn.Sequential(*layers)

    def forward(self, x):
        fea = self.basic_conv(x)
        fea = self.maxpool(fea)
        fea = self.stage_1(fea)
        fea = self.stage_2(fea)
        fea = self.stage_3(fea)
        fea = self.stage_4(fea)
        fea = self.gap(fea)
        fea = torch.squeeze(fea)
        fea = self.classifier(fea)
        return fea

#generate model
#model = ResNeXt(ResNeXtBottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)
def generate_model(model_depth, **kwargs):
    assert model_depth in [26, 50, 101]

    if model_depth == 26:
        model = SKNet([2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = SKNet([3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = SKNet([3, 4, 23, 3], get_inplanes(), **kwargs)

    return model


"""
#input = [batch, channel, time, w, h]
if __name__ == '__main__':
    model = generate_model(50).cuda()
    print(model)

    inp = torch.randn(32, 3, 30, 112, 112).cuda()
    output = model(inp).cuda()
    print(output)
    print(output.size())
"""