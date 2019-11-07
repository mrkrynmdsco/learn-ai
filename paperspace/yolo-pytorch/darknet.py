
import util
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


class EmptyLayer(nn.Module):
    def __init__(self):
        super().__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super().__init__()
        self.anchors = anchors


def parse_cfg(cfgfile):
    """
    Takes a configuration file

    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list

    """
    with open(cfgfile) as cfg:
        lines = cfg.read().splitlines()
        lines = [x for x in lines if len(x) > 0]        # remove the empty lines
        lines = [x for x in lines if x[0] != '#']       # remove comments
        lines = [x.rstrip().lstrip() for x in lines]    # remove the fringe whitespaces

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":                          # This marks the start of a new block
            if len(block) != 0:                     # If block is not empty, implies it is storing values of previous block
                blocks.append(block)                # add it into the blocks list
                block = {}                          # re-init the block
            block['type'] = line[1:-1].rstrip()
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()

    blocks.append(block)

    return blocks


def create_modules(blocks):
    net_info = blocks[0]                            # Captures the information about the input and pre-processing
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        # check the type of block
        # create a new module for the block
        # append to module_list

        if (x['type'] == 'convolutional'):
            # Get the info about the layer
            activation = x['activation']
            try:
                batch_normalize = int(x['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x['filters'])
            padding = int(x['pad'])
            kernel_size = int(x['size'])
            stride = int(x['stride'])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # Add the convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module('conv_{0}'.format(index), conv)

            # Add the batch norm layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module('batch_norm_{0}'.format(index), bn)

            # Check the activation (either Leaky or ReLU for YOLO)
            if activation == 'leaky':
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module('leaky_{0}'.format(index), activn)

        # if it's an upsampling layer (we use Bilinear2dUpsampling)
        elif x['type'] == 'upsample':
            stride = int(x['stride'])
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            module.add_module('upsample_{0}'.format(index), upsample)

        # if it's a route layer
        elif x['type'] == 'route':
            x['layers'] = x['layers'].split(',')

            # start of a route
            start = int(x['layers'][0])

            # end, if there exists one
            try:
                end = int(x['layers'][1])
            except:
                end = 0

            # positive annotation
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index

            route = EmptyLayer()
            module.add_module('route_{0}'.format(index), route)

            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        # shortcut corresponds to skip connection
        elif x['type'] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module('shourtcut_{0}'.format(index), shortcut)

        # YOLO is the detection layer
        elif x['type'] == 'yolo':
            mask = x['mask'].split(',')
            mask = [int(x) for x in mask]

            anchors = x['anchors'].split(',')
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module('detection_{0}'.format(index), detection)

        else:
            print('Something I dunno')
            assert False

        # book keeping
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)


def get_test_input():
    img = cv2.imread('dog-cycle-car.png')
    img = cv2.resize(img, (608, 608))               # Resize to the input dimension
    img_ = img[:, :, ::-1].transpose((2, 0, 1))     # BGR -> RGB | H x W x C -> C x H x W
    img_ = img_[np.newaxis, :, :, :] / 255.0        # Add a channel at 0 (for batch) | Normalize
    img_ = torch.from_numpy(img_).float()           # Convert to float
    img_ = Variable(img_)                           # Convert to Variable
    return img_


class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super().__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

        self.header = None
        self.seen = None

    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}                # We cache the outputs for the route layer
        detections = []

        write = 0

        for i, module in enumerate(modules):
            module_type = module['type']

            if module_type in ['convolutional', 'upsample']:
                x = self.module_list[i](x)

            elif module_type in ['route']:
                layers = module['layers']
                layers = [int(a) for a in layers]

                if layers[0] > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + layers[0]]

                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]

                    x = torch.cat((map1, map2), 1)

            elif module_type in ['shortcut']:
                from_ = int(module['from'])
                x = outputs[i - 1] + outputs[i + from_]

            elif module_type in ['yolo']:
                anchors = self.module_list[i][0].anchors

                # Get the input dimensions
                in_dim = int(self.net_info['height'])

                # Get the number of classes
                num_classes = int(module['classes'])

                # Transform
                x = x.data
                x = util.predict_transform(x, in_dim, anchors, num_classes, CUDA)

                if not write:       # if no collector has been initialized
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x

        return detections

    def load_weights(self, weightfile):
        h = None    # header
        w = None    # weights

        # Open the weights file
        with open(weightfile) as wt:
            # The first 5 values are header information
            # 1. Major version number
            # 2. Minor Version Number
            # 3. Subversion number
            # 4,5. Images seen by the network (during training)
            h = np.fromfile(wt, dtype=np.int32, count=5)

            # Weights
            w = np.fromfile(wt, dtype=np.float32)

        self.header = torch.from_numpy(h)
        self.seen = self.header[3]

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]['type']

            # if module_type is convolutional, load weights, otherwise ignore
            if module_type == 'convolutional':
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i + 1]['batch_normalize'])
                except:
                    batch_normalize = 0

                conv = model[0]

                if batch_normalize:
                    bn = model[1]

                    # Get the number of wweights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    # Load the weights
                    bn_biases = torch.from_numpy(w[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(w[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(w[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(w[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # Cast the loaded weights into dims of model weights
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    # Number of biases
                    num_biases = conv.bias.numel()

                    # Load the weights
                    conv_biases = torch.from_numpy(w[ptr:ptr + num_biases])
                    ptr += num_biases

                    # Re-shape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # Copy the data
                    conv.bias.data.copy_(conv_biases)

                # Load the weights of the Convolutional Layers
                num_weights = conv.weight.numel()

                # Do the same as above for weights
                conv_weights = torch.from_numpy(w[ptr:ptr + num_weights])
                ptr += num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)


def test_forward_pass():
    model = Darknet('cfg/yolov3.cfg')
    inp = get_test_input()
    cuda = torch.cuda.is_available()
    pred = model(inp, cuda)
    print(pred)
    print(pred.size())


def test_load_weights():
    model = Darknet('cfg/yolov3.cfg')
    model.load_weights('weights/yolov3.weights')


# test_forward_pass()
# test_load_weights()
