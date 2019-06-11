import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import maskFOV_on_BEV


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        #out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        #out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, use_bn=True):
        super(Bottleneck, self).__init__()
        bias = not use_bn
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=bias)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        if self.use_bn:
            out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.relu(residual + out)
        return out

class BevBackBone(nn.Module):

    def __init__(self, block, num_block, geom, use_bn=True):
        super(BevBackBone, self).__init__()

        self.use_bn = use_bn
        self.fusion = geom['fusion']
        # Block 1
        self.conv1 = conv3x3(33, 32)
        self.conv2 = conv3x3(32, 32)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        # Block 2-5
        self.in_planes = 32
        self.block2 = self._make_layer(block, 24, num_blocks=num_block[0])
        self.block3 = self._make_layer(block, 48, num_blocks=num_block[1])
        self.block4 = self._make_layer(block, 64, num_blocks=num_block[2])
        self.block5 = self._make_layer(block, 96, num_blocks=num_block[3])

        # Lateral layers
        self.latlayer1 = nn.Conv2d(384, 196, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(192, 96, kernel_size=1, stride=1, padding=0)

        # Top-down layers
        self.deconv1 = nn.ConvTranspose2d(196, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 96, kernel_size=3, stride=2, padding=1, output_padding=1)

        # MLP
        self.squeeze_fusion = nn.Conv2d(2048, 256, kernel_size=1, stride=1)
        self.mlp2 = self._make_mlp(256, 96)
        self.mlp3 = self._make_mlp(256, 192)
        self.mlp4 = self._make_mlp(256, 256)
        self.mlp5 = self._make_mlp(256, 384)
        
    def forward(self, x, y, x2y):
        # x: bev input
        # y: image feature map 93, 310
        # x2y: knn input map 256, 224, k, 2
        
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        c1 = self.relu2(x)

        if self.fusion:
            image_feature = self.bev_image_fusion(y, x2y)
            # bottom up layers
            c2 = self.block2(c1)+self.mlp2(image_feature)
            #print ("m2", m2.size())
            #print ("c2", c2.size())
            c3 = self.block3(c2)+self.mlp3(image_feature[:,:,::2,::2])
            #print ("m3", m3.size())
            #print ("c3", c3.size())
            c4 = self.block4(c3)+self.mlp4(image_feature[:,:,::4,::4])
            #print ("m4", m4.size())
            #print ("c4", c4.size())
            c5 = self.block5(c4)+self.mlp5(image_feature[:,:,::8,::8])
            #print ("m5", m5.size())
            #print ("c5", c5.size())
        else:
            c2 = self.block2(c1)
            c3 = self.block3(c2)
            c4 = self.block4(c3)
            c5 = self.block5(c4)

        l5 = self.latlayer1(c5)
        l4 = self.latlayer2(c4)
        p4 = l4 + self.deconv1(l5)
        l3 = self.latlayer3(c3)
        p3 = l3 + self.deconv2(p4)

        return p3
    
    def bev_image_fusion(self, y, x2y):
        # y: image feature map -1, 128, 93, 310
        # x2y: knn input map -1, 256, 224, 16, k, 2
        # return: 256, 224, 256
        assert list(y.size())[1:] == [128,93,310]
        batch_size = y.size()[0]
        y = y.permute(0, 2, 3, 1)
        # TODO can be better
        x2y[:,:,:,:,:,0] = torch.clamp(x2y[:,:,:,:,:,0], 0, 92)
        x2y[:,:,:,:,:,1] = torch.clamp(x2y[:,:,:,:,:,1], 0, 309)
        x2y = torch.round(x2y)
        x2y = x2y.view(batch_size,-1,2)
        x2y = x2y.long()

        y_bev = torch.Tensor()
        for i in range(batch_size):
            z = y[i,x2y[i,:,0],x2y[i,:,1]]
            z = torch.unsqueeze(z, 0)
            if y_bev.size()[0] == 0:
                y_bev = z
            else:
                y_bev = torch.cat((y_bev, z), 0)
        if True:
            y_bev = y_bev.view(batch_size, 256, 224, 16, -1 ,128)
            y_bev = torch.mean(y_bev, 4).squeeze(4)
            y_bev = y_bev.view(batch_size, 256, 224, -1)
            y_bev = y_bev.permute(0, 3, 1, 2)
            y_bev = self.squeeze_fusion(y_bev)
        else:
            y_bev = y_bev.view(batch_size, 256, 224, -1 ,256)
            y_bev = torch.mean(y_bev, 3).squeeze(3)
            y_bev = y_bev.permute(0, 3, 1, 2)
        return y_bev

    def _make_layer(self, block, planes, num_blocks):
        if self.use_bn:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        else:
            downsample = nn.Conv2d(self.in_planes, planes * block.expansion,
                                   kernel_size=1, stride=2, bias=True)

        layers = []
        layers.append(block(self.in_planes, planes, stride=2, downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, stride=1))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def _make_mlp(self, channel_in, channel_out):
        mlp = nn.Sequential(
            nn.Conv2d(channel_in, channel_in//2, kernel_size=1, stride=1),
            nn.Conv2d(channel_in//2, channel_out, kernel_size=1, stride=1)
        )
        return mlp

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

class ImageBackBone(nn.Module):

    def __init__(self, block, num_block, use_bn=True):
        super(ImageBackBone, self).__init__()

        self.use_bn = use_bn

        # Block 1
        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        # Block 2-5
        self.in_planes = 64
        self.block2 = self._make_layer(block, 48, num_blocks=num_block[0])
        self.block3 = self._make_layer(block, 64, num_blocks=num_block[1])
        self.block4 = self._make_layer(block, 96, num_blocks=num_block[2])
        self.block5 = self._make_layer(block, 128, num_blocks=num_block[3])


        # Lateral layers
        self.latlayer1 = nn.Conv2d(512, 384, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(384, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 128 , kernel_size=1, stride=1, padding=0)

        # Top-down layers
        self.deconv1 = nn.ConvTranspose2d(384, 256, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=(0,1))

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        c1 = self.relu2(x)

        # bottom up layers
        c2 = self.block2(c1)
        c3 = self.block3(c2)
        c4 = self.block4(c3)
        c5 = self.block5(c4)

        l5 = self.latlayer1(c5)
        l4 = self.latlayer2(c4)
        p4 = l4 + self.deconv1(l5)
        l3 = self.latlayer3(c3)
        p3 = l3 + self.deconv2(p4)

        return p3

    def _make_layer(self, block, planes, num_blocks):
        if self.use_bn:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        else:
            downsample = nn.Conv2d(self.in_planes, planes * block.expansion,
                                   kernel_size=1, stride=2, bias=True)

        layers = []
        layers.append(block(self.in_planes, planes, stride=2, downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, stride=1))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

class Header(nn.Module):

    def __init__(self, use_bn=True):
        super(Header, self).__init__()

        self.use_bn = use_bn
        bias = not use_bn
        self.conv1 = conv3x3(96, 96, bias=bias)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = conv3x3(96, 96, bias=bias)
        self.bn2 = nn.BatchNorm2d(96)
        self.conv3 = conv3x3(96, 96, bias=bias)
        self.bn3 = nn.BatchNorm2d(96)
        self.conv4 = conv3x3(96, 96, bias=bias)
        self.bn4 = nn.BatchNorm2d(96)

        self.clshead = conv3x3(96, 1, bias=True)
        self.reghead = conv3x3(96, 6, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.conv3(x)
        if self.use_bn:
            x = self.bn3(x)
        x = self.conv4(x)
        if self.use_bn:
            x = self.bn4(x)

        cls = torch.sigmoid(self.clshead(x))
        reg = self.reghead(x)

        return cls, reg

class Decoder(nn.Module):

    def __init__(self, geom):
        super(Decoder, self).__init__()
        self.geometry = [geom["L1"], geom["L2"], geom["W1"], geom["W2"]]
        self.grid_size = 4*geom["interval"]

        self.target_mean = [0.008, 0.001, 0.202, 0.2, 0.43, 1.368]
        self.target_std_dev = [0.866, 0.5, 0.954, 0.668, 0.09, 0.111]

    def forward(self, x):
        '''

        :param x: Tensor 6-channel geometry
        6 channel map of [cos(yaw), sin(yaw), log(x), log(y), w, l]
        Shape of x: (B, C=6, H=200, W=175)
        :return: Concatenated Tensor of 8 channel geometry map of bounding box corners
        8 channel are [rear_left_x, rear_left_y,
                        rear_right_x, rear_right_y,
                        front_right_x, front_right_y,
                        front_left_x, front_left_y]
        Return tensor has a shape (B, C=8, H=200, W=175), and is located on the same device as x

        '''
        # Tensor in (B, C, H, W)

        device = torch.device('cpu')
        if x.is_cuda:
            device = x.get_device()

        for i in range(6):
            x[:, i, :, :] = x[:, i, :, :] * self.target_std_dev[i] + self.target_mean[i]

        cos_t, sin_t, dx, dy, log_w, log_l = torch.chunk(x, 6, dim=1)
        theta = torch.atan2(sin_t, cos_t)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        x = torch.arange(self.geometry[2], self.geometry[3], self.grid_size, dtype=torch.float32, device=device)
        y = torch.arange(self.geometry[0], self.geometry[1], self.grid_size, dtype=torch.float32, device=device)
        yy, xx = torch.meshgrid([y, x])
        centre_y = yy + dy
        centre_x = xx + dx
        l = log_l.exp()
        w = log_w.exp()
        rear_left_x = centre_x - l/2 * cos_t - w/2 * sin_t
        rear_left_y = centre_y - l/2 * sin_t + w/2 * cos_t
        rear_right_x = centre_x - l/2 * cos_t + w/2 * sin_t
        rear_right_y = centre_y - l/2 * sin_t - w/2 * cos_t
        front_right_x = centre_x + l/2 * cos_t + w/2 * sin_t
        front_right_y = centre_y + l/2 * sin_t - w/2 * cos_t
        front_left_x = centre_x + l/2 * cos_t - w/2 * sin_t
        front_left_y = centre_y + l/2 * sin_t + w/2 * cos_t

        decoded_reg = torch.cat([rear_left_x, rear_left_y, rear_right_x, rear_right_y,
                                 front_right_x, front_right_y, front_left_x, front_left_y], dim=1)

        return decoded_reg

class PIXOR(nn.Module):
    '''
    The input of PIXOR nn module is a tensor of [batch_size, height, weight, channel]
    The output of PIXOR nn module is also a tensor of [batch_size, height/4, weight/4, channel]
    Note that we convert the dimensions to [C, H, W] for PyTorch's nn.Conv2d functions
    '''

    def __init__(self, geom, use_bn=True, decode=False):
        super(PIXOR, self).__init__()
        self.backbone = BevBackBone(Bottleneck, [2, 2, 2, 2], geom, use_bn)
        self.resnet = ImageBackBone(Bottleneck, [2, 2, 2, 2], use_bn)
        
        self.header = Header(use_bn)
        self.corner_decoder = Decoder(geom)
        self.use_decode = decode
        self.cam_fov_mask = maskFOV_on_BEV(geom['label_shape'])
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        prior = 0.01
        self.header.clshead.weight.data.fill_(-math.log((1.0-prior)/prior))
        self.header.clshead.bias.data.fill_(0)
        self.header.reghead.weight.data.fill_(0)
        self.header.reghead.bias.data.fill_(0)

    def set_decode(self, decode):
        self.use_decode = decode

    def forward(self, x, y, x2y):
        # x: lidar 
        # y: image
        # x2y: bev to image
        
        device = torch.device('cpu')
        if x.is_cuda:
            device = x.get_device()
        
        # x = x.permute(0, 3, 1, 2)
        # Torch Takes Tensor of shape (Batch_size, channels, height, width)
        features_iamge = self.resnet(y)
        print (features_iamge.shape)

        features = self.backbone(x, features_iamge, x2y)
        
        cls, reg = self.header(features)
        self.cam_fov_mask = self.cam_fov_mask.to(device)
        cls = cls * self.cam_fov_mask
        if self.use_decode:
            decoded = self.corner_decoder(reg)
            # Return tensor(Batch_size, height, width, channels)
            #decoded = decoded.permute(0, 2, 3, 1)
            #cls = cls.permute(0, 2, 3, 1)
            #reg = reg.permute(0, 2, 3, 1)
            pred = torch.cat([cls, reg, decoded], dim=1)
        else:
            pred = torch.cat([cls, reg], dim=1)

        return pred

def test_decoder(decode = True):
    geom = {
        "fusion": True,
        "L1": -40.0,
        "L2": 40.0,
        "W1": 0.0,
        "W2": 70.0,
        "H1": -2.5,
        "H2": 1.0,
        'interval': 0.15625,
        'input_shape': (512, 448, 33),
        'knn_shape': (256, 224, 16),
        'label_shape': (128, 112, 7)
    }
    print("Testing PIXOR decoder")
    net = PIXOR(geom, use_bn=False)
    net.set_decode(decode)
    preds = net(torch.autograd.Variable(torch.randn(2, 33, 512, 448)), torch.autograd.Variable(torch.randn(2, 3, 370, 1240)), torch.autograd.Variable(torch.randn(2, 256, 224, 16, 2,2)))
    print(net)

    print("Predictions output size", preds.size())

if __name__ == "__main__":
    test_decoder()
