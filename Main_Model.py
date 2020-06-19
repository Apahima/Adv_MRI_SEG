import sys
import numpy as np
import random
from Common.Unet_Medical_Parse import Args
from Common.MedicalDataLoading import MedicalDataLoading
import torch
import logging
import os
from datetime import datetime
from datetime import date
from tensorboardX import SummaryWriter
import torchvision
import shutil


from torch.utils.data import DataLoader
from Models.Unet import Pytorch_Unet as pytorch_unet
from torchsummary import summary
import torch.nn.functional as F
from collections import defaultdict
from Models.Unet.Loss import dice_loss
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import pathlib
from Common import MedicalImageShow

logging.basicConfig(level=logging.DEBUG,filemode='w', filename=os.getcwd()+'/Model_Debug.log')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


def train_model(model, optimizer, scheduler, dataloaders, args, writer, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

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
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())


        time_elapsed = time.time() - since

        #writing the results to TensorboardX
        writer.add_scalar('BCE', metrics['bce'], epoch)
        writer.add_scalar('DICE', metrics['dice'], epoch)
        writer.add_scalar('LOSS', metrics['loss'], epoch)

        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    save_model(args, args.exp_dir, model, optimizer, best_loss)
    return model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def main(args):

    #Construct the summary:
    now = datetime.now()
    construct_time_stamp = now.strftime("_%I-%M-%S %p")
    Data_path = args.data_path
    folder_name = str(date.today()) + str(construct_time_stamp)

    writer = SummaryWriter(log_dir=args.exp_dir / folder_name / 'Unet-Channels {}, --lr ={}, --epochs - {}, --Pools ={}'.format(args.num_chans, args.lr, args.num_epochs, args.num_pools))

    train_set, val_set = MedicalDataLoading(Data_path)

    image_datasets = {
        'train': train_set, 'val': val_set
    }

    dataloaders = {
        'train': DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    }

    dataset_sizes = {
        x: len(image_datasets[x]) for x in image_datasets.keys()
    }
    print('Dataset size', dataset_sizes)

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

    model = train_model(model, optimizer_ft, exp_lr_scheduler, dataloaders, args, writer, num_epochs=args.num_epochs)

    model.eval()  # Set model to evaluate mode

    visualize(args, model, dataloaders, writer)


    ### Block for saving plot side by side
    inputs, labels = next(iter(dataloaders['val']))  # next(iter()) gives batch of images from dataloader with size of actual batch size
    inputs = inputs.to(args.device)
    labels = labels.to(args.device)

    pred = model(inputs)

    inputs = inputs.cpu().numpy()
    pred = pred.data.cpu().numpy()
    labels = labels.cpu().numpy()
    print('Number of scans to plot side by side:', pred.shape)

    MedicalImageShow.plot_side_by_side([inputs, labels, pred], args)

    ###
    writer.close()

def visualize(args, model, dataloaders, writer):
    def save_image(image, tag):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid)

    model.eval()
    with torch.no_grad():
        inputs, labels = next(iter(dataloaders['val']))  # next(iter()) gives batch of images from dataloader with size of actual batch size
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        pred = model(inputs)

        save_image(labels, 'Ground Throuth Segmentation')
        save_image(pred, 'Segmentation')
        save_image(inputs, 'Original Scan')

        print('Visualization DONE')



def save_model(args, exp_dir, model, optimizer, best_loss):
    torch.save(
        {
            'epoch': args.num_epochs,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_loss': best_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    # if is_new_best:
    #     shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


def create_arg_parser():
    parser = Args()
    # model parameters
    parser.add_argument('--num-pools', type=int, default=3, help='Number of U-Net pooling layers')
    parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--num-chans', type=int, default=2, help='Number of U-Net channels')

    parser.add_argument('--batch-size', default=16, type=int, help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=5000, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=40,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')
    parser.add_argument('--momentum', type=float, default=0.,
                        help='Momentum factor')

    parser.add_argument('--report-interval', type=int, default=100, help='Period of loss reporting')
    parser.add_argument('--data-parallel', action='store_true',
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--exp-dir', type=pathlib.Path, default='checkpoints',
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--output-chans',type=int, default=1,
                        help='For multi-segmentation used Chans -> #Classes for single segmentation used Chans -> 1')


    # parser.add_argument('--data-split', choices=['val', 'test'], required=True,
    #                     help='Which data partition to run on: "val" or "test"')
    # This feature is dedicate for supervised learning and H5 files
    return parser


if __name__ == '__main__':
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    args = create_arg_parser().parse_args(sys.argv[1:])
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)