# Code to produce colored segmentation output in Pytorch for all cityscapes subsets
# Sept 2017
# Eduardo Romera
#######################

import numpy as np
import torch
import os
import importlib

from matplotlib import pyplot as plt

import torch.nn.functional as F

from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage

from dataset import cityscapes
from erfnet import ERFNet
from enet import ENet
from bisenetv1 import BiSeNetV1
from erfnet_isomax_plus import ERFNetIsomaxPlus
from transform import Relabel, ToLabel, Colorize

import glob
import re


NUM_CHANNELS = 3
NUM_CLASSES = 20

image_transform = ToPILImage()
input_transform_cityscapes = Compose(
    [
        Resize((512, 1024), Image.BILINEAR),
        ToTensor(),
        # Normalize([.485, .456, .406], [.229, .224, .225]),
    ]
)
target_transform_cityscapes = Compose(
    [
        Resize((512, 1024), Image.NEAREST),
        ToLabel(),
        Relabel(255, 19),  # ignore label to 19
    ]
)

cityscapes_trainIds2labelIds = Compose(
    [
        Relabel(19, 255),
        Relabel(18, 33),
        Relabel(17, 32),
        Relabel(16, 31),
        Relabel(15, 28),
        Relabel(14, 27),
        Relabel(13, 26),
        Relabel(12, 25),
        Relabel(11, 24),
        Relabel(10, 23),
        Relabel(9, 22),
        Relabel(8, 21),
        Relabel(7, 20),
        Relabel(6, 19),
        Relabel(5, 17),
        Relabel(4, 13),
        Relabel(3, 12),
        Relabel(2, 11),
        Relabel(1, 8),
        Relabel(0, 7),
        Relabel(255, 0),
        ToPILImage(),
    ]
)


def main(args):

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights
    modelname = args.loadModel.rstrip(".py")

    print("Loading model: " + modelpath)
    print("Loading weights: " + weightspath)

    # Import ERFNet model from the folder
    # Net = importlib.import_module(modelpath.replace("/", "."), "ERFNet")
    if modelname == "erfnet":
        model = ERFNet(NUM_CLASSES)
    elif modelname == 'erfnet_isomax_plus':
        model = ERFNetIsomaxPlus(NUM_CLASSES)
    elif modelname == "enet":
        model = ENet(NUM_CLASSES)
    elif modelname == "bisenetv1":
        model = BiSeNetV1(NUM_CLASSES)
        model.aux_mode = 'eval'

    model = torch.nn.DataParallel(model)
    if not args.cpu:
        model = model.cuda()

    def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith('module.'):
                    own_state[name.split('module.')[-1]].copy_(param)
                else:
                    print(name, ' not loaded')
                    continue
            else:
                own_state[name].copy_(param)
        return model

    if modelname == 'enet':
        model = load_my_state_dict(model.module, torch.load(weightspath)['state_dict'])
    elif modelname == 'bisenetv1':
        if args.loadWeights.endswith('.tar'):
            model = load_my_state_dict(model, torch.load(weightspath)['state_dict'])
        else:
            model = load_my_state_dict(model, torch.load(weightspath))
        mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda()
        std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda()
    else:
        if args.loadWeights.endswith('.tar'):
            model = load_my_state_dict(model, torch.load(weightspath)['state_dict'])
        else:
            model = load_my_state_dict(model, torch.load(weightspath))
    print('Model and weights LOADED successfully')
    model.eval()

    for path in glob.glob(os.path.expanduser(args.input)):
        print(path)
        images = input_transform_cityscapes((Image.open(path).convert('RGB'))).unsqueeze(0).float().cuda()

        with torch.no_grad():
            if modelname == 'bisenetv1':
                images = images - mean
                images = images / std

            if modelname == "erfnet" or modelname == "erfnet_isomax_plus":
                result = model(images).squeeze(0)
            elif modelname == "enet":
                result = torch.roll(model(images).squeeze(0), -1, 0)
            elif modelname == "bisenetv1":
                result = model(images)[0].squeeze(0)

        if args.method == 'void':
            anomaly_result = F.softmax(result, dim=0)[-1]
        else:
            # discard 20th class output
            anomaly_tmp = result[:-1]
            if args.method == 'msp':
                # MSP with temperature scaling
                anomaly_result = 1.0 - torch.max(F.softmax(anomaly_tmp / args.temperature, dim=0), dim=0)[0]
            elif args.method == 'maxlogit':
                anomaly_result = -torch.max(anomaly_tmp, dim=0)[0]
            elif args.method == 'maxentropy':
                anomaly_result = torch.div(
                    torch.sum(-F.softmax(anomaly_tmp, dim=0) * F.log_softmax(anomaly_tmp, dim=0), dim=0),
                    torch.log(torch.tensor(anomaly_tmp.size(0))),
                )

        result = result.max(0)[1].byte().cpu().data
        # label_cityscapes = cityscapes_trainIds2labelIds(label.unsqueeze(0))
        label_color = Colorize()(result.unsqueeze(0))

        filenameSave = (
            "./save_color/"
            + f'{modelname}/{args.method}/'
            + re.split("leftImg8bit/|validation_dataset/", path)[2 if 'leftImg8bit' in path else 1]
        )
        os.makedirs(os.path.dirname(filenameSave), exist_ok=True)
        # image_transform(label.byte()).save(filenameSave)
        label_save = ToPILImage()(label_color)
        label_save.save(filenameSave)

        plt.imsave(filenameSave.rstrip('.png') + '_colormap.png', anomaly_result.cpu().numpy(), cmap='bwr')

        os.system(
            f'python evalAnomaly_for_color.py --method {args.method} --input {path} --loadWeights {args.loadWeights} --loadModel {args.loadModel}'
        )


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument(
        '--input',
        type=str,
        default='../validation_dataset/RoadObsticle21/images/*.webp',
        help='A single glob pattern such as "directory/*.jpg"',
    )
    parser.add_argument('--loadDir', default='../trained_models/')
    parser.add_argument('--loadWeights', default='erfnet_pretrained.pth')
    parser.add_argument('--loadModel', default='erfnet.py')
    parser.add_argument('--subset', default='val')  # can be val or train (must have labels)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--method', type=str, default='msp')
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('--width', type=int, default=2048)

    main(parser.parse_args())
