# Code to produce colored segmentation output in Pytorch for all cityscapes subsets
# Sept 2017
# Eduardo Romera
#######################

import numpy as np
import torch
import os
import importlib

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

import visdom


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

    modelname = args.loadModel.rstrip(".py")
    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print('Loading model: ' + modelpath)
    print('Loading weights: ' + weightspath)
    print(modelname)
    if modelname == "erfnet":
        net = ERFNet(NUM_CLASSES)
    elif modelname == "erfnet_isomax_plus":
        net = ERFNetIsomaxPlus(NUM_CLASSES)
    elif modelname == "enet":
        net = ENet(NUM_CLASSES)
    elif modelname == "bisenetv1":
        net = BiSeNetV1(NUM_CLASSES)
        net.aux_mode = 'eval'

    if not args.cpu:
        model = torch.nn.DataParallel(net).cuda()

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
    print("Model and weights LOADED successfully")

    model.eval()

    if not os.path.exists(args.datadir):
        print("Error: datadir could not be loaded")

    loader = DataLoader(
        cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset=args.subset),
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # For visualizer:
    # must launch in other window "python3.6 -m visdom.server -port 8097"
    # and access localhost:8097 to see it
    if args.visualize:
        vis = visdom.Visdom()

    for step, (images, labels, filename, filenameGt) in enumerate(loader):
        if not args.cpu:
            images = images.cuda()
            # labels = labels.cuda()

        inputs = Variable(images)
        # targets = Variable(labels)
        with torch.no_grad():
            if modelname == 'bisenetv1':
                inputs = images - mean
                inputs = images / std

            if modelname == "erfnet" or modelname == "erfnet_isomax_plus":
                outputs = model(inputs)
            elif modelname == "enet":
                outputs = torch.roll(model(inputs), -1, 0)
            elif modelname == "bisenetv1":
                outputs = model(inputs)[0]

        label = outputs[0].max(0)[1].byte().cpu().data
        # label_cityscapes = cityscapes_trainIds2labelIds(label.unsqueeze(0))
        label_color = Colorize(NUM_CLASSES)(label.unsqueeze(0))

        filenameSave = './save_color/' + f'{modelname}/' + filename[0].split("leftImg8bit/")[1]
        os.makedirs(os.path.dirname(filenameSave), exist_ok=True)
        # image_transform(label.byte()).save(filenameSave)
        label_save = ToPILImage()(label_color)
        label_save.save(filenameSave)

        if args.visualize:
            vis.image(label_color.numpy())
        print(step, filenameSave)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--state')

    parser.add_argument('--loadDir', default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  # can be val, test, train, demoSequence

    parser.add_argument('--datadir', default=os.getenv("HOME") + "/datasets/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')

    parser.add_argument('--visualize', action='store_true')
    main(parser.parse_args())
