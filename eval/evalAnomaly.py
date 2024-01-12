# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet
from erfnet import ERFNet
from enet import ENet
from bisenetv1 import BiSeNetV1
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr, plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
import torch.nn.functional as F

seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20
# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--input',
        type=str,
        default='/content/validation_dataset/RoadObsticle21/images/*.webp',
        help='A single glob pattern such as "directory/*.jpg"',
    )  
    parser.add_argument('--loadDir',default='../trained_models/')
    parser.add_argument('--loadWeights', default='erfnet_pretrained.pth')
    parser.add_argument('--loadModel', default='erfnet.py')
    parser.add_argument('--subset', default='val')  #can be val or train (must have labels)
    parser.add_argument('--datadir', default='/content/cityscapes')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--method', type=str, default='msp')
    parser.add_argument('--temperature', type=float, default=1)
    args = parser.parse_args()
    anomaly_score_list = []
    ood_gts_list = []

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')

    modelname = args.loadModel.rstrip(".py")
    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ('Loading model: ' + modelpath)
    print ('Loading weights: ' + weightspath)

    if modelname == "erfnet":
        model = ERFNet(NUM_CLASSES)
    elif modelname == "enet":
        model = ENet(NUM_CLASSES)
    elif modelname == "bisenetv1":
        model = BiSeNetV1(NUM_CLASSES)

    if not args.cpu:
        model = torch.nn.DataParallel(model).cuda()

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
        model = load_my_state_dict(model, torch.load(weightspath))
    else:
        model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print ('Model and weights LOADED successfully')
    model.eval()

    for path in glob.glob(os.path.expanduser(args.input)):
        print(path)
        images = torch.from_numpy(np.array(Image.open(path).convert('RGB'))).unsqueeze(0).float().cuda()
        images = images.permute(0, 3, 1, 2)
        with torch.no_grad():
            if modelname == "erfnet":
                result = model(images).squeeze(0)
            elif modelname == "enet":
                result = torch.roll(model(images).squeeze(0), -1, 0)
            elif modelname == "bisenetv1":
                result = model(images)[0].squeeze(0)
        if args.method == 'msp':
            # MSP with temperature scaling
            anomaly_result = 1.0 - torch.max(F.softmax(result / args.temperature, dim=0), dim=0)[0]
        elif args.method == 'maxlogit':
            anomaly_result = 1.0 - torch.max(result, dim=0)[0]
        elif args.method == 'maxentropy':
            anomaly_result = torch.div(torch.sum(- F.softmax(result, dim=0) * F.log_softmax(result, dim=0), dim=0), torch.log(torch.tensor(result.size(0))))
        elif args.method == 'void':
            anomaly_result = F.softmax(result, dim=0)[-1]
        anomaly_result = anomaly_result.data.cpu().numpy()
        pathGT = path.replace('images', 'labels_masks')
        if 'RoadObsticle21' in pathGT:
            pathGT = pathGT.replace('webp', 'png')
        if 'fs_static' in pathGT:
            pathGT = pathGT.replace('jpg', 'png')                
        if 'RoadAnomaly' in pathGT:
            pathGT = pathGT.replace('jpg', 'png')  

        mask = Image.open(pathGT)
        ood_gts = np.array(mask)

        if 'RoadAnomaly' in pathGT:
            ood_gts = np.where((ood_gts == 2), 1, ood_gts)
        if 'LostAndFound' in pathGT:
            ood_gts = np.where((ood_gts == 0), 255, ood_gts)
            ood_gts = np.where((ood_gts == 1), 0, ood_gts)
            ood_gts = np.where((ood_gts > 1) & (ood_gts < 201), 1, ood_gts)

        if 'Streethazard' in pathGT:
            ood_gts = np.where((ood_gts == 14), 255, ood_gts)
            ood_gts = np.where((ood_gts < 20), 0, ood_gts)
            ood_gts = np.where((ood_gts == 255), 1, ood_gts)

        if 1 not in np.unique(ood_gts):
            continue
        else:
            ood_gts_list.append(ood_gts)
            anomaly_score_list.append(anomaly_result)
        del result, anomaly_result, ood_gts, mask, images
        torch.cuda.empty_cache()

    file.write('\n')

    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    ood_mask = ood_gts == 1
    ind_mask = ood_gts == 0

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))

    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    dataset = args.input.split("/")[-3]
    print(f'Model: {modelname.upper()}')
    print(f'Method: {args.method}')
    print(f'Dataset: {dataset}')
    if args.method == 'msp':
        print(f'Temperature: {args.temperature}')
    print(f'AUPRC score: {prc_auc*100.0}')
    print(f'FPR@TPR95: {fpr*100.0}')

    file.write(f'Model: {modelname.upper()}    Method: {args.method}     Dataset: {dataset}{f"   Temperature: {args.temperature}" if args.method == "msp" else ""}    AUPRC score: {prc_auc * 100.0}   FPR@TPR95: {fpr * 100.0}')
    file.close()


if __name__ == '__main__':
    main()