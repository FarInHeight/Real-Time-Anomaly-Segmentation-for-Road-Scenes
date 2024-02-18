# Code to calculate IoU (mean and per-class) in a dataset
# Nov 2017
# Eduardo Romera
#######################

import numpy as np
import torch
import torch.nn.functional as F
import os
import importlib
import time

from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage

from dataset import cityscapes
from erfnet import ERFNet
from erfnet_isomax_plus import ERFNetIsomaxPlus
from enet import ENet
from bisenetv1 import BiSeNetV1
from transform import Relabel, ToLabel, Colorize
from iouEval import iouEval, getColorEntry

NUM_CHANNELS = 3
NUM_CLASSES = 20

image_transform = ToPILImage()
input_transform_cityscapes = Compose(
    [
        Resize(512, Image.BILINEAR),
        ToTensor(),
    ]
)
input_transform_bisenetv1_cityscapes = Compose(
    [
        Resize(512, Image.BILINEAR),
        ToTensor(),
        Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])),
    ]
)
target_transform_cityscapes = Compose(
    [
        Resize(512, Image.NEAREST),
        ToLabel(),
        Relabel(255, 19),  # ignore label to 19
    ]
)


def main(args):
    if not os.path.exists('iou_results.txt'):
        open('iou_results.txt', 'w').close()

    modelname = args.loadModel.rstrip(".py")
    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print("Loading model: " + modelpath)
    print("Loading weights: " + weightspath)

    global input_transform_cityscapes
    if modelname == "erfnet":
        net = ERFNet(NUM_CLASSES)
    if modelname == "erfnet_isomax_plus":
        net = ERFNetIsomaxPlus(NUM_CLASSES)
    elif modelname == "enet":
        net = ENet(NUM_CLASSES)
    elif modelname == "bisenetv1":
        net = BiSeNetV1(NUM_CLASSES)
        net.aux_mode = 'eval'
        input_transform_cityscapes = input_transform_bisenetv1_cityscapes

    # model = torch.nn.DataParallel(model)
    if not args.cpu:
        model = torch.nn.DataParallel(net).cuda()

    def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
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
    else:
        if args.loadWeights.endswith('.tar'):
            model = load_my_state_dict(model, torch.load(weightspath)['state_dict'])
        else:
            model = load_my_state_dict(model, torch.load(weightspath))

    # print(model.module.state_dict().eys())

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

    iouEvalVal = iouEval(NUM_CLASSES)

    start = time.time()

    for step, (images, labels, filename, filenameGt) in enumerate(loader):
        if not args.cpu:
            images = images.cuda()
            labels = labels.cuda()

        inputs = Variable(images)
        with torch.no_grad():
            outputs = model(inputs)

        if modelname == 'enet':
            outputs = torch.roll(outputs, -1, 1)
        elif modelname == 'bisenetv1':
            outputs = outputs[0]

        iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, labels)

        filenameSave = filename[0].split("leftImg8bit/")[1]

        print(step, filenameSave)

    iouVal, iou_classes = iouEvalVal.getIoU()

    iou_classes_str = []
    for i in range(iou_classes.size(0)):
        # iouStr = getColorEntry(iou_classes[i])+'{:0.2f}'.format(iou_classes[i]*100) + '\033[0m'
        iouStr = '{:0.2f}'.format(iou_classes[i] * 100)
        iou_classes_str.append(iouStr)

    with open('iou_results.txt', 'a') as f:
        print("---------------------------------------", file=f)
        print("Model", modelname.upper(), "Took", time.time() - start, "seconds", file=f)
        print("=======================================", file=f)
        # print("TOTAL IOU: ", iou * 100, "%", file=f)
        print("Per-Class IoU:", file=f)
        print(iou_classes_str[0], "Road", file=f)
        print(iou_classes_str[1], "sidewalk", file=f)
        print(iou_classes_str[2], "building", file=f)
        print(iou_classes_str[3], "wall", file=f)
        print(iou_classes_str[4], "fence", file=f)
        print(iou_classes_str[5], "pole", file=f)
        print(iou_classes_str[6], "traffic light", file=f)
        print(iou_classes_str[7], "traffic sign", file=f)
        print(iou_classes_str[8], "vegetation", file=f)
        print(iou_classes_str[9], "terrain", file=f)
        print(iou_classes_str[10], "sky", file=f)
        print(iou_classes_str[11], "person", file=f)
        print(iou_classes_str[12], "rider", file=f)
        print(iou_classes_str[13], "car", file=f)
        print(iou_classes_str[14], "truck", file=f)
        print(iou_classes_str[15], "bus", file=f)
        print(iou_classes_str[16], "train", file=f)
        print(iou_classes_str[17], "motorcycle", file=f)
        print(iou_classes_str[18], "bicycle", file=f)
        print("=======================================", file=f)
        # iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
        iouStr = '{:0.2f}'.format(iouVal * 100)
        print("MEAN IoU: ", iouStr, "%", file=f)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--state')

    parser.add_argument('--loadDir', default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  # can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')

    main(parser.parse_args())
