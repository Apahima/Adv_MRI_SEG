import sys
import numpy as np
import random
import torch
import logging
import os
import time
import copy

from torchsummary import summary
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from collections import defaultdict

import Common.Unet_Medical_Parse as UnetParser
from Common.Unet_Medical_Parse import Args
from Common.MedicalDataLoading import MedicalDataLoading
from Common.Saving import save_model
from Common import MedicalVisualization

from datetime import datetime
from datetime import date
from tensorboardX import SummaryWriter
import torchvision
import shutil
import matplotlib.pyplot as plt

from Models.Unet import Pytorch_Unet as pytorch_unet
from Models.Unet.Loss import dice_loss
from Models.Losses.DiceLoss.dice_loss import WBCE_DiceLoss, BinaryDiceLoss,WBCEWithLogitLoss
from Models.Losses.TverskyLoss.binarytverskyloss import FocalBinaryTverskyLoss, BinaryTverskyLossV2


import pathlib


logging.basicConfig(level=logging.DEBUG,filemode='w', filename=os.getcwd()+'/Model_Debug.log')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# def calc_loss(pred, target, metrics, bce_weight=0.5):
#     bce = F.binary_cross_entropy_with_logits(pred, target)
#
#     pred = F.sigmoid(pred)
#     dice = dice_loss(pred, target)
#
#     loss = bce * bce_weight + dice * (1 - bce_weight)
#
#     metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
#     metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
#     metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
#
#     return loss

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


def train_model(model, optimizer, scheduler, dataloaders, args, criterion, writer, folder_name, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    # criterion_WBCE_DiceLoss = WBCE_DiceLoss(alpha=args.WBCE_diceloss, reduction='mean')
    # criterion_Tversky = BinaryTverskyLossV2(alpha=args.tversky_alpha, beta=args.tversky_beta)



    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                # scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    loss = criterion(outputs, labels, metrics)
                    metrics[args.loss] += loss.data.cpu().numpy() * labels.size(0)
                    # calc_loss(outputs, labels, metrics)
                    # loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0) # I would like to summerized the entire samples in one epoch and devide the accumulate loss by the number of the samples.
                                                # Actually I'm averaging the loss over the samples

            if phase == 'train': #saving the Loss just for the traning phase
                Towrite = metrics.copy()
                MedicalVisualization.WriteToTensorboard(Towrite, epoch_samples, writer,epoch)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics[args.loss] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())


        time_elapsed = time.time() - since

        # #writing the results to TensorboardX
        # writer.add_scalar('BCE', metrics['bce'], epoch)
        # writer.add_scalar('DICE', metrics['dice'], epoch)
        # writer.add_scalar('LOSS', metrics['loss'], epoch)
        # writer.add_scalar('Dice Mean Similarity', metrics['Dice_Mean_Similarity'], epoch)
        # writer.add_scalar('Tversky', metrics['Tversky'], epoch)
        # writer.add_scalar('Tversky Mean Similarity', metrics['Tversky_Mean_Similarity'], epoch)

        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    save_model(args, args.exp_dir, model, optimizer, best_loss,folder_name)
    return model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def main(args):
    #Construct the summary:
    now = datetime.now()
    construct_time_stamp = now.strftime("-%I-%M-%S-%p")
    Data_path = args.data_path
    folder_name = str(date.today()) + str(construct_time_stamp)

    #Document model parameter for later investigation
    writer = SummaryWriter(log_dir=args.exp_dir / folder_name)
    # writer.add_text('Model parameters', 'Unet-Channels {}, --lr ={}, --epochs - {}, --Pools ={}'.format(args.num_chans, args.lr, args.num_epochs, args.num_pools))
    if args.loss == 'WBCE_DiceLoss':
        writer.add_text('Model parameters', 'Unet-Channels {}, --lr ={}, --epochs - {}, --Pools ={}'
                                            '  \nLoss function: {}'
                                            '  \nDice Weight: {}'.format(args.num_chans, args.lr, args.num_epochs, args.num_pools, args.loss, args.WBCE_diceloss))
    if args.loss == 'Tversky':
        writer.add_text('Model parameters', 'Unet-Channels {}, --lr ={}, --epochs - {}, --Pools ={}'
                                            '  \nLoss function: {}'
                                            '  \nAlpha Coef: {}'
                                            '  \nBeta Coef: {}'.format(args.num_chans, args.lr, args.num_epochs,
                                                                       args.num_pools, args.loss, args.tversky_alpha, args.tversky_beta))


    train_set, val_set, test_set = MedicalDataLoading(Data_path)

    image_datasets = {
        'train': train_set, 'val': val_set, 'test': test_set
    }

    dataloaders = {
        'train': DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=0),
        'test': DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    }

    dataset_sizes = {
        x: len(image_datasets[x]) for x in image_datasets.keys()
    }
    print('Dataset size', dataset_sizes)

    if args.loss == 'WBCE_DiceLoss':
        criterion = WBCE_DiceLoss(alpha=args.WBCE_diceloss, reduction='mean')
    if args.loss == 'Tversky':
        criterion = BinaryTverskyLossV2(alpha=args.tversky_alpha, beta=args.tversky_beta)

    def build_optim(args, params):
        optimizer = optim.Adam(params, args.lr)
        return optimizer

    def build_model(args):
        model = pytorch_unet.UnetModel(in_chans=1, # define the argument
            out_chans=1, # define the argument
            chans=args.num_chans,
            num_pool_layers=args.num_pools,
            drop_prob=args.drop_prob
        ).to(args.device)
        return model

    model = build_model(args)
    summary(model, (1, 256, 256))
    # Observe that all parameters are being optimized
    optimizer_ft = build_optim(args, model.parameters())
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=25, gamma=0.1)

    if args.eval != True:
        model = train_model(model, optimizer_ft, exp_lr_scheduler, dataloaders, args,criterion, writer, folder_name, num_epochs=args.num_epochs)
    else:
        model.load_state_dict(torch.load(os.path.join(args.exp_dir, args.eval_folder, 'Model.pt')))

    model.eval()  # Set model to evaluate mode

    ### Prepare for visualization
    if args.loss == 'WBCE_DiceLoss':
        SegmentationLoss = BinaryDiceLoss()
    if args.loss == 'Tversky':
        SegmentationLoss = criterion

    MedicalVisualization.visualize(args, model, dataloaders,SegmentationLoss, writer)

    ### Block for saving plot side by side
    inputs, labels = next(iter(dataloaders['test']))  # next(iter()) gives batch of images from dataloader with size of actual batch size
    inputs = inputs.to(args.device)
    labels = labels.to(args.device)

    pred = model(inputs)

    inputs = inputs.cpu().numpy()
    pred = pred.data.cpu().numpy()
    labels = labels.cpu().numpy()
    print('Number of scans to plot side by side:', pred.shape)

    #MedicalVisualization.plot_side_by_side([inputs, labels, pred], args)
    ###

    writer.close()


if __name__ == '__main__':
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    args = UnetParser.create_arg_parser().parse_args(sys.argv[1:])
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)