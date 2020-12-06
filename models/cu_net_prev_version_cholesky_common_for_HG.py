

"""
    DU-Net Model with the STN and CEN
    
    Version 1 2019-08-06 Abhinav Kumar
    Version 0 2017-02-xx Zhiqiang Tang
"""

import torch
import torch.nn as nn
import math
import numpy
from collections import OrderedDict
from torch.autograd import Variable, Function
from torch._thnn import type2backend
from torch.backends import cudnn
from functools import reduce
from operator import mul
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter

from models.spatial_transformer_network import SpatialTransformerNetwork

class BinOp():
    def __init__(self, model):
        # count the number of Conv2d
        count_Conv2d = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                count_Conv2d = count_Conv2d + 1

        start_range = 1
        end_range = count_Conv2d-2
        self.bin_range = numpy.linspace(start_range,
                end_range, end_range-start_range+1)\
                        .astype('int').tolist()
        self.num_of_params = len(self.bin_range)
        self.saved_params = []
        self.target_params = []
        self.target_modules = []
        index = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                index = index + 1
                if index in self.bin_range:
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)

    def binarization(self):
        self.meancenterConvParams()
        self.clampConvParams()
        self.save_params()
        self.binarizeConvParams()

    def meancenterConvParams(self):
        for index in range(self.num_of_params):
            s = self.target_modules[index].data.size()
            negMean = self.target_modules[index].data.mean(1).\
                    mul(-1).expand_as(self.target_modules[index].data)
            self.target_modules[index].data = self.target_modules[index].data.add(negMean)

    def clampConvParams(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.clamp(-1.0, 1.0,
                    out = self.target_modules[index].data)

    def save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def binarizeConvParams(self):
        for index in range(self.num_of_params):
            n = self.target_modules[index].data[0].nelement()
            s = self.target_modules[index].data.size()
            m = self.target_modules[index].data.norm(1, 3)\
                    .sum(2).sum(1).div(n)
            self.target_modules[index].data.sign()\
                    .mul(m.expand(s), out=self.target_modules[index].data)

    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    def updateBinaryGradWeight(self):
        for index in range(self.num_of_params):
            weight = self.target_modules[index].data
            n = weight[0].nelement()
            s = weight.size()
            m = weight.norm(1, 3)\
                    .sum(2).sum(1).div(n).expand(s)
            m[weight.lt(-1.0)] = 0 
            m[weight.gt(1.0)] = 0
            m = m.mul(self.target_modules[index].grad.data)
            m_add = weight.sign().mul(self.target_modules[index].grad.data)
            m_add = m_add.sum(3)\
                    .sum(2).sum(1).div(n).expand(s)
            m_add = m_add.mul(weight.sign())
            self.target_modules[index].grad.data = m.add(m_add).mul(1.0-1.0/s[1]).mul(n)

class _SharedAllocation(object):
    """
    A helper class which maintains a shared memory allocation.
    Used for concatenation and batch normalization.
    """
    def __init__(self, storage):
        self.storage = storage

    def type(self, t):
        self.storage = self.storage.type(t)

    def type_as(self, obj):
        if isinstance(obj, Variable):
            self.storage = self.storage.type(obj.data.storage().type())
        elif isinstance(obj, torch._TensorBase):
            self.storage = self.storage.type(obj.storage().type())
        else:
            self.storage = self.storage.type(obj.type())

    def resize_(self, size):
        if self.storage.size() < size:
            self.storage.resize_(size)
        return self

class _EfficientDensenetBottleneck(nn.Module):
    """
    A optimized layer which encapsulates the batch normalization, ReLU, and
    convolution operations within the bottleneck of a DenseNet layer.
    This layer usage shared memory allocations to store the outputs of the
    concatenation and batch normalization features. Because the shared memory
    is not perminant, these features are recomputed during the backward pass.
    """
    def __init__(self, shared_allocation_1, shared_allocation_2, num_input_channels, num_output_channels):

        super(_EfficientDensenetBottleneck, self).__init__()
        self.shared_allocation_1 = shared_allocation_1
        self.shared_allocation_2 = shared_allocation_2
        self.num_input_channels = num_input_channels

        self.norm_weight = nn.Parameter(torch.Tensor(num_input_channels))
        self.norm_bias = nn.Parameter(torch.Tensor(num_input_channels))
        self.register_buffer('norm_running_mean', torch.zeros(num_input_channels))
        self.register_buffer('norm_running_var', torch.ones(num_input_channels))
        self.conv_weight = nn.Parameter(torch.Tensor(num_output_channels, num_input_channels, 1, 1))
        self._reset_parameters()


    def _reset_parameters(self):
        self.norm_running_mean.zero_()
        self.norm_running_var.fill_(1)
        self.norm_weight.data.uniform_()
        self.norm_bias.data.zero_()
        stdv = 1. / math.sqrt(self.num_input_channels)
        self.conv_weight.data.uniform_(-stdv, stdv)


    def forward(self, inputs):
        if isinstance(inputs, Variable):
            inputs = [inputs]
        fn = _EfficientDensenetBottleneckFn(self.shared_allocation_1, self.shared_allocation_2,
            self.norm_running_mean, self.norm_running_var,
            stride=1, padding=0, dilation=1, groups=1,
            training=self.training, momentum=0.1, eps=1e-5)
        return fn(self.norm_weight, self.norm_bias, self.conv_weight, *inputs)

class _DenseLayer(nn.Sequential):

    def __init__(self, shared_allocation_1, shared_allocation_2, in_num, neck_size, growth_rate):
        super(_DenseLayer, self).__init__()
        self.shared_allocation_1 = shared_allocation_1
        self.shared_allocation_2 = shared_allocation_2

        self.add_module('bottleneck', _EfficientDensenetBottleneck(shared_allocation_1, shared_allocation_2,
                                                           in_num, neck_size * growth_rate))
        self.add_module('norm.2', nn.BatchNorm2d(neck_size * growth_rate))
        self.add_module('relu.2', nn.ReLU(inplace=True))
        self.add_module('conv.2', nn.Conv2d(neck_size * growth_rate, growth_rate,
                                            kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        if isinstance(x, Variable):
            prev_features = [x]
        else:
            prev_features = x
        # print(len(prev_features))
        new_features = super(_DenseLayer, self).forward(prev_features)

        return new_features

class _DenseBlock(nn.Module):
    def __init__(self, in_num, neck_size, growth_rate, layer_num, max_link,
                 storage_size=1024, requires_skip=True, is_up=False):
        input_storage_1 = torch.Storage(storage_size)
        input_storage_2 = torch.Storage(storage_size)
        self.shared_allocation_1 = _SharedAllocation(input_storage_1)
        self.shared_allocation_2 = _SharedAllocation(input_storage_2)
        self.saved_features = []
        self.max_link = max_link
        self.requires_skip = requires_skip
        super(_DenseBlock, self).__init__()
        max_in_num = in_num + max_link * growth_rate
        self.final_num_features = max_in_num
        self.layers = []
        #print('layer number is %d' % layer_num)
        for i in range(0, layer_num):
            if i < max_link:
                tmp_in_num = in_num + i * growth_rate
            else:
                tmp_in_num = max_in_num
            #print('layer %d input channel number is %d' % (i, tmp_in_num))
            self.layers.append(_DenseLayer(self.shared_allocation_1, self.shared_allocation_2,
                                           tmp_in_num, neck_size, growth_rate))
        self.layers = nn.ModuleList(self.layers)
        self.adapters_ahead = []
        adapter_in_nums = []
        adapter_out_num = in_num
        if is_up:
            adapter_out_num = adapter_out_num / 2
        for i in range(0, layer_num):
            if i < max_link:
                tmp_in_num = in_num + (i+1) * growth_rate
            else:
                tmp_in_num = max_in_num + growth_rate
            adapter_in_nums.append(tmp_in_num)
            #print('adapter %d input channel number is %d' % (i, adapter_in_nums[i]))
            self.adapters_ahead.append(_EfficientDensenetBottleneck(self.shared_allocation_1,
                                                                    self.shared_allocation_2,
                                                                    adapter_in_nums[i], adapter_out_num))
        self.adapters_ahead = nn.ModuleList(self.adapters_ahead)
        #print('adapter output channel number is %d' % adapter_out_num)
        if requires_skip:
            print('creating skip layers ...')
            self.adapters_skip = []
            for i in range(0, layer_num):
                self.adapters_skip.append(_EfficientDensenetBottleneck(self.shared_allocation_1,
                                                                       self.shared_allocation_2,
                                                                       adapter_in_nums[i], adapter_out_num))
            self.adapters_skip = nn.ModuleList(self.adapters_skip)

    def forward(self, x, i):
        if i == 0:
            #print '_DenseBlock start i = ', i, 'x size = ', x.size()
            self.saved_features = []
            if isinstance(x, Variable):
                # Update storage type
                self.shared_allocation_1.type_as(x)
                self.shared_allocation_2.type_as(x)
                # Resize storage
                final_size = list(x.size())
            elif isinstance(x, list):
                self.shared_allocation_1.type_as(x[0])
                self.shared_allocation_2.type_as(x[0])
                # Resize storage
                final_size = list(x[0].size())
            else:
                print('invalid type in the input of _DenseBlock module. exiting ...')
                exit()
            # print(final_size)
            final_size[1] = self.final_num_features
            # print(final_size)
            final_storage_size = reduce(mul, final_size, 1)
            # print(final_storage_size)
            self.shared_allocation_1.resize_(final_storage_size)
            self.shared_allocation_2.resize_(final_storage_size)

        
        if isinstance(x, Variable):
            x = [x]
        #print '_DenseBlock start i = ', i, 'x length = ', len(x)
        x = x + self.saved_features

        out = self.layers[i](x)
        if i < self.max_link:
            self.saved_features.append(out)
        elif len(self.saved_features) != 0:
            self.saved_features.pop(0)
            self.saved_features.append(out)
        x.append(out)
        out_ahead = self.adapters_ahead[i](x)
        if self.requires_skip:
            out_skip = self.adapters_skip[i](x)
            return out_ahead, out_skip
        else:
            return out_ahead

class _IntermediaBlock(nn.Module):
    def __init__(self, in_num, out_num, layer_num, max_link, storage_size=1024):
        input_storage_1 = torch.Storage(storage_size)
        input_storage_2 = torch.Storage(storage_size)
        self.shared_allocation_1 = _SharedAllocation(input_storage_1)
        self.shared_allocation_2 = _SharedAllocation(input_storage_2)
        max_in_num = in_num + out_num * max_link
        self.final_num_features = max_in_num
        self.saved_features = []
        self.max_link = max_link
        super(_IntermediaBlock, self).__init__()
        print('creating intermedia block ...')
        self.adapters = []
        for i in range(0, layer_num-1):
            if i < max_link:
                tmp_in_num = in_num + (i+1) * out_num
            else:
                tmp_in_num = max_in_num
            print('intermedia layer %d input channel number is %d' % (i, tmp_in_num))
            self.adapters.append(_EfficientDensenetBottleneck(self.shared_allocation_1,
                                                              self.shared_allocation_2,
                                                              tmp_in_num, out_num))
        self.adapters = nn.ModuleList(self.adapters)
        print('intermedia layer output channel number is %d' % out_num)

    def forward(self, x, i):
        if i == 0:
            #print 'start intermdedia while i = ', i
            self.saved_features = []
            if isinstance(x, Variable):
                # Update storage type
                self.shared_allocation_1.type_as(x)
                self.shared_allocation_2.type_as(x)
                # Resize storage
                final_size = list(x.size())
                if self.max_link != 0:
                    self.saved_features.append(x)
            elif isinstance(x, list):
                self.shared_allocation_1.type_as(x[0])
                self.shared_allocation_2.type_as(x[0])
                # Resize storage
                final_size = list(x[0].size())
                if self.max_link != 0:
                    self.saved_features = self.saved_features + x
            else:
                print('invalid type in the input of _DenseBlock module. exiting ...')
                exit()
            final_size[1] = self.final_num_features
            #print 'final size of intermedia block is ', final_size
            final_storage_size = reduce(mul, final_size, 1)
            #print 'final_storage_size in intermedia block', final_storage_size
            self.shared_allocation_1.resize_(final_storage_size)
            self.shared_allocation_2.resize_(final_storage_size)
            #print('middle list length is %d' % len(self.saved_features))
            return x

        #print 'start intermdedia while i = ', i
        if isinstance(x, Variable):
            # self.saved_features.append(x)
            x = [x]
        x = x + self.saved_features
        out = self.adapters[i-1](x)
        if i < self.max_link:
            self.saved_features.append(out)
        elif len(self.saved_features) != 0:
            self.saved_features.pop(0)
            self.saved_features.append(out)
        #print('middle list length is %d' % len(self.saved_features))
        return out

class _Bn_Relu_Conv1x1(nn.Sequential):
    """
    A helper block which converts the 128 output channels of each hourglass to 
    68 output channels. This can then be associated with each point. Also, some
    of the channels have a bright patch at the lower left of the image. Passing 
    through batchnorm removes the constant values.

    Input : batch_size x 128 x 64 x 64
    Output: batch_size x 68  x 64 x 64
    """
    def __init__(self, in_num, out_num):
        super(_Bn_Relu_Conv1x1, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_num))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_num, out_num,
                                          kernel_size=1, stride=1, bias=False))

# class _TransitionDown(nn.Module):
#     def __init__(self, in_num_list, out_num, num_units):
#         super(_TransitionDown, self).__init__()
#         self.adapters = []
#         for i in range(0, num_units):
#             self.adapters.append(_Bn_Relu_Conv1x1(in_num=in_num_list[i], out_num=out_num))
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#
#     def forward(self, x, i):
#         x = self.adapters[i](x)
#         out = self.pool(x)
#         return out
#
# class _TransitionUp(nn.Module):
#     def __init__(self, in_num_list, out_num_list, num_units):
#         super(_TransitionUp, self).__init__()
#         self.adapters = []
#         for i in range(0, num_units):
#             self.adapters.append(_Bn_Relu_Conv1x1(in_num=in_num_list[i], out_num=out_num_list[i]))
#         self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
#
#     def forward(self, x, i):
#         x = self.adapters[i](x)
#         out = self.upsample(x)
#         return out


class _CU_Net(nn.Module):
    """
    A block which simulates a stack of 4 Hourglasses (HG). To make stacking of 
    8 hourglasses, the first stacking of 4 hourglasses is re-used. It gives two 
    outputs - one is the final 128 dimensional heatmaps and also the output 
    before the bottlenecks of each of the hourglass

    Input : batch_size x 128 x 64 x 64
    Output: batch_size x 128 x  4 x  4, batch_size x 128 x 64 x 64 
    """
    def __init__(self, in_num, neck_size, growth_rate, layer_num, max_link):
        super(_CU_Net, self).__init__()
        self.down_blocks = []
        self.up_blocks = []

        # CUNet HG consists of 4 hourglasses.  
        self.num_blocks = 4
        self.layer_num = layer_num
        print('creating hg ...')
        for i in range(0, self.num_blocks):
            print('creating down block %d ...' % i)
            self.down_blocks.append(_DenseBlock(in_num=in_num, neck_size=neck_size,
                                      growth_rate=growth_rate, layer_num=layer_num,
                                      max_link=max_link, requires_skip=True))
            print('creating up block %d ...' % i)
            self.up_blocks.append(_DenseBlock(in_num=in_num*2, neck_size=neck_size,
                                      growth_rate=growth_rate, layer_num=layer_num,
                                      max_link=max_link, requires_skip=False, is_up=True))
        self.down_blocks = nn.ModuleList(self.down_blocks)
        self.up_blocks = nn.ModuleList(self.up_blocks)
        print('creating neck block ...')
        self.neck_block = _DenseBlock(in_num=in_num, neck_size=neck_size,
                                     growth_rate=growth_rate, layer_num=layer_num,
                                     max_link=max_link, requires_skip=False)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x, i):

        skip_list = [None] * self.num_blocks
        #print 'input x size is ', x.size()
    
        for j in range(0, self.num_blocks):
            #print('using down block %d ...' % j)            
            x, skip_list[j] = self.down_blocks[j](x, i)

            #print 'x size is ', x.size()
            #print 'skip size is ', skip_list[j].size()

            x = self.maxpool(x)
            y = x

            #print 'x size is before return', x.size()
            # x size is before return 24 x 128 x 32 x 32
            # x size is before return 24 x 128 x 16 x 16
            # x size is before return 24 x 128 x  8 x  8
            # x size is before return 24 x 128 x  4 x  4

        y = self.neck_block(y, i)

        #print('using neck block ...')
        #print 'output size is ', y.size()                
        # output size is 24 x 128 x 4 x 4

        for j in list(reversed(range(0, self.num_blocks))):
            y = self.upsample(y)
            #x = nn.functional.interpolate(x,scale_factor=2, mode='nearest')
            #print('using up block %d ...' % j)
            y = self.up_blocks[j]([y, skip_list[j]], i)
            #print 'output y sizesss is ', y.size()
        
        # x contains the bottleneck outputs, y contains the Hourglass outputs   
        return x, y

class _CholeskyBlock(nn.Sequential):
    """
    A helper block which converts the output of neck block to 
    class_num*3 coefficients

    Input of this layer: batch_size x 128 x 4 x 4
    Output of the layer: batch_size x 68  x 3
    """
    def __init__(self, class_num, mlp_tot_layers= 1, mlp_hidden_units= 4096, get_mean_from_mlp= False):
        super(_CholeskyBlock, self).__init__()

        if get_mean_from_mlp:
            num_coefficients_to_output = 5
        else:
            num_coefficients_to_output = 3

        if mlp_tot_layers > 1:    
            self.add_module('fc_1' , nn.Linear(128*4*4, mlp_hidden_units))
            self.add_module('relu1', nn.ReLU())

            # Run for minus 2 hidden layers since one layer is output and other 
            # is input.
            for i in range(mlp_tot_layers-2):
                self.add_module('fc_' + str(i+2), nn.Linear(mlp_hidden_units   , mlp_hidden_units))
                self.add_module('relu' + str(i+2), nn.ReLU())

            self.add_module('fc_'+str(mlp_tot_layers) , nn.Linear(mlp_hidden_units, class_num* num_coefficients_to_output))
        else:
           self.add_module('fc_1' , nn.Linear(128*4*4, class_num* num_coefficients_to_output)) 

        #for m in self.modules():
        #    if isinstance(m, nn.Linear):
        #        torch.nn.init.kaiming_normal_(m.weight.data)
        #        m.bias.data.fill_(0.0)


class _CU_Net_Wrapper(nn.Module):
    def __init__(self, init_chan_num, neck_size, growth_rate,
                 class_num, layer_num, order, loss_num, use_spatial_transformer= False, 
                 mlp_tot_layers= 2, mlp_hidden_units= 4096, get_mean_from_mlp= False):
        assert loss_num <= layer_num and loss_num >= 1
        loss_every = float(layer_num) / float(loss_num)
        self.loss_anchors = []
        for i in range(0, loss_num):
            tmp_anchor = int(round(loss_every * (i + 1)))
            if tmp_anchor <= layer_num:
                self.loss_anchors.append(tmp_anchor)

        assert layer_num in self.loss_anchors
        assert loss_num == len(self.loss_anchors)

        if order >= layer_num:
            print 'order is larger than the layer number.'
            exit()
        print('layer number is %d' % layer_num)
        print('loss number is %d' % loss_num)
        print('loss anchors are: ', self.loss_anchors)
        print('order is %d' % order)
        print('growth rate is %d' % growth_rate)
        print('neck size is %d' % neck_size)
        print('class number is %d' % class_num)
        print('initial channel number is %d' % init_chan_num)
        num_chans = init_chan_num
        super(_CU_Net_Wrapper, self).__init__()
        self.layer_num = layer_num
        self.class_num = class_num
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, init_chan_num, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(init_chan_num)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=2, stride=2)),
        ]))
        # self.denseblock0 = _DenseBlock(layer_num=4, in_num=init_chan_num,
        #                                neck_size=neck_size, growth_rate=growth_rate)
        # hg_in_num = init_chan_num + growth_rate * 4
        print('channel number is %d' % num_chans)
        self.hg = _CU_Net(in_num=num_chans, neck_size=neck_size, growth_rate=growth_rate,
                             layer_num=layer_num, max_link=order)

        #self.hg_down = _CU_Net_0(in_num=num_chans, neck_size=neck_size, growth_rate=growth_rate,
        #                     layer_num=layer_num, max_link=order)

        self.linears = []
        for i in range(0, layer_num):
            self.linears.append(_Bn_Relu_Conv1x1(in_num=num_chans, out_num= self.class_num))
        self.linears = nn.ModuleList(self.linears)
        # intermedia_in_nums = []
        # for i in range(0, num_units-1):
        #     intermedia_in_nums.append(num_chans * (i+2))
        self.intermedia = _IntermediaBlock(in_num=num_chans, out_num=num_chans,
                                           layer_num=layer_num, max_link=order)        

        self.use_spatial_transformer = use_spatial_transformer
        if self.use_spatial_transformer:
            print("Using spatial transformer on heatmaps")
            self.spatial_transformer_network = SpatialTransformerNetwork()

        self.mlp_tot_layers   = mlp_tot_layers
        self.mlp_hidden_units = mlp_hidden_units
        self.get_mean_from_mlp = get_mean_from_mlp
        self.cholesky = _CholeskyBlock(class_num= self.class_num, mlp_tot_layers= self.mlp_tot_layers, mlp_hidden_units= self.mlp_hidden_units, get_mean_from_mlp= self.get_mean_from_mlp)
        self.elu_plus = ELU_plus()
        print(self.cholesky)

        # Used for predicting probability
        self.probability = nn.Linear(128*4*4, self.class_num)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                stdv = 1/math.sqrt(n)
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.uniform_(-stdv, stdv)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.uniform_()
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)

        output_at_each_HG   = []
        cholesky_at_each_HG = [] 

        for i in range(0, self.layer_num):
            #print('using HG %d ...' % i)
            #print("x_size before intermediate layer is", x.size())
            # x size before intermedia layer is  24 x 128 x 64 x 64
            # x size before intermedia layer is  24 x 128 x 64 x 64
            # x size before intermedia layer is  24 x 128 x 64 x 64
            # x size before intermedia layer is  24 x 128 x 64 x 64
            # x size before intermedia layer is  24 x 128 x 64 x 64
            # x size before intermedia layer is  24 x 128 x 64 x 64
            # x size before intermedia layer is  24 x 128 x 64 x 64
            # x size before intermedia layer is  24 x 128 x 64 x 64

            x = self.intermedia(x, i) 
            #print 'x size after intermedia layer is ', x.size()
            # x size after intermedia layer is  24 x 128 x 64 x 64
            # x size after intermedia layer is  24 x 128 x 64 x 64
            # x size after intermedia layer is  24 x 128 x 64 x 64
            # x size after intermedia layer is  24 x 128 x 64 x 64
            # x size after intermedia layer is  24 x 128 x 64 x 64
            # x size after intermedia layer is  24 x 128 x 64 x 64
            # x size after intermedia layer is  24 x 128 x 64 x 64
            # x size after intermedia layer is  24 x 128 x 64 x 64

            yy, x = self.hg(x, i)
            #print 'x size after hg is ', x.size()
            # x size after hg is  24 x 128 x 64 x 64
            # x size after hg is  24 x 128 x 64 x 64
            # x size after hg is  24 x 128 x 64 x 64
            # x size after hg is  24 x 128 x 64 x 64
            # x size after hg is  24 x 128 x 64 x 64
            # x size after hg is  24 x 128 x 64 x 64
            # x size after hg is  24 x 128 x 64 x 64
            # x size after hg is  24 x 128 x 64 x 64

            #print "yy size after hg is ", yy.size()
            # yy size after hg is 24 x 128 x 4 x 4
            # yy size after hg is 24 x 128 x 4 x 4
            # yy size after hg is 24 x 128 x 4 x 4
            # yy size after hg is 24 x 128 x 4 x 4
            # yy size after hg is 24 x 128 x 4 x 4
            # yy size after hg is 24 x 128 x 4 x 4
            # yy size after hg is 24 x 128 x 4 x 4
            # yy size after hg is 24 x 128 x 4 x 4

            batch_size = yy.shape[0]
            bottleneck = yy
            bottleneck = bottleneck.view(batch_size, -1)

            # ======================= Probabilities ====================
            probabilities = self.sigmoid(self.probability(bottleneck))
            
            # =================== Cholesky coefficients =================
            y = self.cholesky(bottleneck)
            
            if self.get_mean_from_mlp:
                # There are two mean coefficients in addition to three Cholesky
                # coefficients + 1 probability of visibile
                landmarks_reshape = y.view(batch_size, self.class_num, 5)
                update_y   = Variable(torch.zeros((batch_size, self.class_num, 6)).float())
            else:
                landmarks_reshape = y.view(batch_size, self.class_num, 3)
                update_y   = Variable(torch.zeros((batch_size, self.class_num, 4)).float())
           
            if yy.is_cuda:
                update_y = update_y.cuda()

            # Pass the diagonal entries of lower triangular matrix through ELU so
            # is never zero. For the lower triangular matrix L 
            #  _  _
            # |a  0|
            # |b  c|
            #  -  -
            # the Covariance matrix = LL^T =
            #  _         _
            # |a^2      ab|
            # |ab  b^2+c^2|
            #  --       --
            # This is full rank if |LL^T| =  |a^2c^2| > 0. So, none of the a or c 
            # should be zero.
            update_y[:, :, 1] = landmarks_reshape[:, :, 1]
            update_y[:, :, 0] = self.elu_plus(landmarks_reshape[:,:,0])
            update_y[:, :, 2] = self.elu_plus(landmarks_reshape[:,:,2])

            update_y[:, :, 3] = probabilities.unsqueeze(-1)

            if self.get_mean_from_mlp:
                # There are two mean coefficients in addition to three Cholesky
                # coefficients and probability of visible
                # probability comes from a separate network (VEN)
                update_y[:, :, 4] = landmarks_reshape[:, :, 3]
                update_y[:, :, 5] = landmarks_reshape[:, :, 4]

            cholesky_at_each_HG.append(update_y)

            if (i + 1) in self.loss_anchors:
                tmp_out = self.linears[i](x)

                # Pass the heatmaps through the spatial transformer
                if self.use_spatial_transformer:
                    tmp_out = self.spatial_transformer_network(tmp_out)

                # print 'tmp output size is ', tmp_out.size()
                output_at_each_HG.append(tmp_out)

        assert len(self.loss_anchors) == len(output_at_each_HG)
        return output_at_each_HG, cholesky_at_each_HG

def create_cu_net(neck_size, growth_rate, init_chan_num,
                  class_num, layer_num, order, loss_num, use_spatial_transformer= False,
                  mlp_tot_layers= 1, mlp_hidden_units= 4096, get_mean_from_mlp= False ):
    net = _CU_Net_Wrapper(init_chan_num=init_chan_num, neck_size=neck_size,
                            growth_rate=growth_rate, class_num=class_num,
                            layer_num=layer_num, order=order, loss_num=loss_num,
                            use_spatial_transformer= use_spatial_transformer, 
                            mlp_tot_layers= mlp_tot_layers, mlp_hidden_units= mlp_hidden_units,
                            get_mean_from_mlp= get_mean_from_mlp)
    return net

class ELU_plus(nn.modules.Module):
    """
    ELU function plus alpha. 
    ELU_plus  maps x in the range (0, infinity]. Similar to ELU and ReLU
    ELU       maps x in the range (-alpha, infinity]
    RELU_plus maps x in the range [alpha,  infinity] but has a flat region
    ReLU      maps x in the range [0,      infinity]
    """
    def __init__(self, alpha= 1., inplace= False):
        super(ELU_plus, self).__init__()
        self.alpha   = alpha
        self.inplace = inplace

    def forward(self, input):
        return F.elu(input, self.alpha, self.inplace) + self.alpha

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + '(' + 'alpha=' + str(self.alpha) + inplace_str + ')'

class _EfficientDensenetBottleneckFn(Function):
    """
    The autograd function which performs the efficient bottlenck operations.
    Each of the sub-operations -- concatenation, batch normalization, ReLU,
    and convolution -- are abstracted into their own classes
    """
    def __init__(self, shared_allocation_1, shared_allocation_2,
            running_mean, running_var,
            stride=1, padding=0, dilation=1, groups=1,
            training=False, momentum=0.1, eps=1e-5):

        self.efficient_cat = _EfficientCat(shared_allocation_1.storage)
        self.efficient_batch_norm = _EfficientBatchNorm(shared_allocation_2.storage, running_mean, running_var,
                training, momentum, eps)
        self.efficient_relu = _EfficientReLU()
        self.efficient_conv = _EfficientConv2d(stride, padding, dilation, groups)

        # Buffers to store old versions of bn statistics
        self.prev_running_mean = self.efficient_batch_norm.running_mean.new()
        self.prev_running_mean.resize_as_(self.efficient_batch_norm.running_mean)
        self.prev_running_var = self.efficient_batch_norm.running_var.new()
        self.prev_running_var.resize_as_(self.efficient_batch_norm.running_var)
        self.curr_running_mean = self.efficient_batch_norm.running_mean.new()
        self.curr_running_mean.resize_as_(self.efficient_batch_norm.running_mean)
        self.curr_running_var = self.efficient_batch_norm.running_var.new()
        self.curr_running_var.resize_as_(self.efficient_batch_norm.running_var)


    def forward(self, bn_weight, bn_bias, conv_weight, *inputs):
        self.prev_running_mean.copy_(self.efficient_batch_norm.running_mean)
        self.prev_running_var.copy_(self.efficient_batch_norm.running_var)

        bn_input = self.efficient_cat.forward(*inputs)
        bn_output = self.efficient_batch_norm.forward(bn_weight, bn_bias, bn_input)
        relu_output = self.efficient_relu.forward(bn_output)
        conv_output = self.efficient_conv.forward(conv_weight, None, relu_output)
        #conv_output = self.efficient_conv.forward(conv_weight, bn_bias, relu_output) #  

        self.bn_weight = bn_weight
        self.bn_bias = bn_bias
        self.conv_weight = conv_weight
        self.inputs = inputs
        return conv_output


    def backward(self, grad_output):
        # Turn off bn training status, and temporarily reset statistics
        training = self.efficient_batch_norm.training
        self.curr_running_mean.copy_(self.efficient_batch_norm.running_mean)
        self.curr_running_var.copy_(self.efficient_batch_norm.running_var)
        # self.efficient_batch_norm.training = False
        self.efficient_batch_norm.running_mean.copy_(self.prev_running_mean)
        self.efficient_batch_norm.running_var.copy_(self.prev_running_var)

        # Recompute concat and BN
        cat_output = self.efficient_cat.forward(*self.inputs)
        bn_output = self.efficient_batch_norm.forward(self.bn_weight, self.bn_bias, cat_output)
        relu_output = self.efficient_relu.forward(bn_output)

        # Conv backward
        conv_weight_grad, _, conv_grad_output = self.efficient_conv.backward(
                self.conv_weight, None, relu_output, grad_output)

        # ReLU backward
        relu_grad_output = self.efficient_relu.backward(bn_output, conv_grad_output)

        # BN backward
        self.efficient_batch_norm.running_mean.copy_(self.curr_running_mean)
        self.efficient_batch_norm.running_var.copy_(self.curr_running_var)
        bn_weight_grad, bn_bias_grad, bn_grad_output = self.efficient_batch_norm.backward(
                self.bn_weight, self.bn_bias, cat_output, relu_grad_output)

        # Input backward
        grad_inputs = self.efficient_cat.backward(bn_grad_output)

        # Reset bn training status and statistics
        self.efficient_batch_norm.training = training
        self.efficient_batch_norm.running_mean.copy_(self.curr_running_mean)
        self.efficient_batch_norm.running_var.copy_(self.curr_running_var)

        return tuple([bn_weight_grad, bn_bias_grad, conv_weight_grad] + list(grad_inputs))


# The following helper classes are written similarly to pytorch autograd functions.
# However, they are designed to work on tensors, not variables, and therefore
# are not functions.


class _EfficientBatchNorm(object):
    def __init__(self, storage, running_mean, running_var,
            training=False, momentum=0.1, eps=1e-5):
        self.storage = storage
        self.running_mean = running_mean
        self.running_var = running_var
        self.training = training
        self.momentum = momentum
        self.eps = eps

    def forward(self, weight, bias, input):
        # Assert we're using cudnn
        for i in ([weight, bias, input]):
            if i is not None and not(cudnn.is_acceptable(i)):
                raise Exception('You must be using CUDNN to use _EfficientBatchNorm')

        # Create save variables
        self.save_mean = self.running_mean.new()
        self.save_mean.resize_as_(self.running_mean)
        self.save_var = self.running_var.new()
        self.save_var.resize_as_(self.running_var)

        # Do forward pass - store in input variable
        res = type(input)(self.storage)
        res.resize_as_(input)
        #print('hahahahahah--------', input.size(), ' ', res.size(), ' ', weight.size(), ' ', bias.size(), ' ',self.running_mean.size(), ' ', 
        #    self.running_var.size(), ' ',self.save_mean.size(), ' ', self.save_var.size(), ' ', type(self.training), type(self.momentum), type(self.eps))
        torch._C._cudnn_batch_norm_forward(
            input, res, weight, bias, self.running_mean, self.running_var, self.save_mean, self.save_var, self.training, self.momentum, self.eps
        )

        return res

    def recompute_forward(self, weight, bias, input):
        # Do forward pass - store in input variable
        res = type(input)(self.storage)
        res.resize_as_(input)
        torch._C._cudnn_batch_norm_forward(
            input, res, weight, bias, self.running_mean, self.running_var,
            self.save_mean, self.save_var, self.training, self.momentum, self.eps
        )

        return res

    def backward(self, weight, bias, input, grad_output):
        # Create grad variables
        grad_weight = weight.new()
        grad_weight.resize_as_(weight)
        grad_bias = bias.new()
        grad_bias.resize_as_(bias)

        # Run backwards pass - result stored in grad_output
        grad_input = grad_output
        torch._C._cudnn_batch_norm_backward(
            input, grad_output, grad_input, grad_weight, grad_bias,
            weight, self.running_mean, self.running_var, self.save_mean,
            self.save_var, self.training, self.eps
        )

        # Unpack grad_output
        res = tuple([grad_weight, grad_bias, grad_input])
        return res


class _EfficientCat(object):
    def __init__(self, storage):
        self.storage = storage

    def forward(self, *inputs):
        # Get size of new varible
        self.all_num_channels = [input.size(1) for input in inputs]
        size = list(inputs[0].size())
        for num_channels in self.all_num_channels[1:]:
            size[1] += num_channels

        # Create variable, using existing storage
        if (self.storage).is_cuda == False:
            self.storage = self.storage.cuda()

        res = type(inputs[0])(self.storage).resize_(size)
        torch.cat(inputs, dim=1, out=res)
        return res

    def backward(self, grad_output):
        # Return a table of tensors pointing to same storage
        res = []
        index = 0
        for num_channels in self.all_num_channels:
            new_index = num_channels + index
            res.append(grad_output[:, index:new_index])
            index = new_index
        return tuple(res)


class _EfficientReLU(object):
    def __init__(self):
        pass

    def forward(self, input):
        backend = type2backend[type(input)]
        output = input
        backend.Threshold_updateOutput(backend.library_state, input, output, 0, 0, True)
        return output

    def backward(self, input, grad_output):
        grad_input = grad_output
        grad_input.masked_fill_(input <= 0, 0)
        return grad_input


class _EfficientConv2d(object):
    def __init__(self, stride=1, padding=0, dilation=1, groups=1):
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def _output_size(self, input, weight):
        channels = weight.size(0)
        output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = self.padding
            kernel = self.dilation * (weight.size(d + 2) - 1) + 1
            stride = self.stride
            output_size += ((in_size + (2 * pad) - kernel) // stride + 1,)
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError("convolution input is too small (output would be {})".format(
                             'x'.join(map(str, output_size))))
        return output_size

    def forward(self, weight, bias, input):
        # Assert we're using cudnn
        for i in ([weight, bias, input]):
            if i is not None and not(cudnn.is_acceptable(i)):
                raise Exception('You must be using CUDNN to use _EfficientBatchNorm')

        res = input.new(*self._output_size(input, weight))
        self._cudnn_info = torch._C._cudnn_convolution_full_forward(input, weight, bias, res,(self.padding, self.padding),(self.stride, self.stride),(self.dilation, self.dilation), self.groups, cudnn.benchmark, False) #wenxuan False added

        return res

    def backward(self, weight, bias, input, grad_output):
        grad_input = input.new()
        grad_input.resize_as_(input)
        torch._C._cudnn_convolution_backward_data(
            grad_output, grad_input, weight, self._cudnn_info,
            cudnn.benchmark, False) #wenxuan False added

        grad_weight = weight.new().resize_as_(weight)
        torch._C._cudnn_convolution_backward_filter(grad_output, input, grad_weight, self._cudnn_info,
                                                    cudnn.benchmark, False)#wenxuan False

        if bias is not None:
            grad_bias = bias.new().resize_as_(bias)
            torch._C._cudnn_convolution_backward_bias(grad_output, grad_bias, self._cudnn_info)
        else:
            grad_bias = None

        return grad_weight, grad_bias, grad_input
