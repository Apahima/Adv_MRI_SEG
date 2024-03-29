import matplotlib.pyplot as plt
import numpy as np
import torchvision
import os
from datetime import datetime
from datetime import date
import torch
from Models.Unet.Loss import dice_loss
from Models.Losses.DiceLoss.dice_loss import BinaryDiceLoss
from Common import Evaluations_Param as EvalP


def plot_img_array(img_array, ncol=3):
    nrow = len(img_array) // ncol

    f, plots = plt.subplots(nrow, ncol, sharex='all', sharey='all', figsize=(ncol * 4, nrow * 4))

    for i in range(len(img_array)):
        plots[i // ncol, i % ncol]
        plots[i // ncol, i % ncol].imshow(np.sum(img_array[i], axis=0), cmap='gray')

    plt.savefig(r'SidebySide_SegmentationBestFit.png')
    # plt.show()

from functools import reduce
def plot_side_by_side(img_arrays,args):
    flatten_list = reduce(lambda x,y: x+y, zip(*img_arrays))

    plot_img_array(np.array(flatten_list), ncol=len(img_arrays))




def visualize(args, model, dataloaders,SegmentationLoss, writer):
    """

    :param args:
    :param model:
    :param dataloaders:
    :param SegmentationLoss: Get loss function constructor, need to put only Output and Label (GT)
    :param writer:
    :return: None
    """
    def save_image_to_writer(image, tag):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid)

    def save_as_unified_grid(ScanLabelPred, tag, index):
        ScanLabelPred -= ScanLabelPred.min()
        ScanLabelPred /= ScanLabelPred.max()
        grid = torchvision.utils.make_grid(ScanLabelPred, nrow=3, pad_value=1)
        writer.add_image(tag, grid, index)

    def save_image_as_file(image,tag,args,writer):
        for i, image in enumerate(image):
            image = np.squeeze(image.cpu().numpy(), axis=0)
            timest = datetime.now().strftime("%I-%M-%S.%f")[:-3]
            path, _ = list(writer.all_writers.items())[0]
            plt.imsave(os.path.join(path,'{}-{}-{}.png'.format(tag,i,timest)), image, cmap='gray')

    # def save_as_embbeded_seg(inputs,labels,pred):
    #     emb_label = inputs.detach().numpy()
    #     emb_pred = inputs.detach().numpy()
    #     labels =labels.detach().numpy()
    #     emb_label = np.dstack([emb_label,emb_label,emb_label])
    #
    #     emb_label[0,:,:][labels[0,:,:] > 0] = 255
    #     emb_pred[pred] = [0,255,0]

    with torch.no_grad():
        inputs, labels = next(iter(dataloaders['test']))  # next(iter()) gives batch of images from dataloader with size of actual batch size
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        output = model(inputs)
        pred = torch.sigmoid(output) #Scaling the output image to be (0,1)

        save_image_to_writer(labels, 'Ground Throuth Segmentation')
        save_image_to_writer(pred, 'Segmentation')
        save_image_to_writer(inputs, 'Original Scan')


        if args.savetestfile == 'Y':
            save_image_as_file(labels, 'Ground Throuth Segmentation', args,writer)
            save_image_as_file(pred, 'Segmentation', args,writer)
            save_image_as_file(inputs, 'Original Scan', args,writer)
            print('Save Images DONE')

        for Unified, _ in enumerate(inputs):
            ScanLabelPred = torch.cat((inputs[Unified,:].unsqueeze(0),labels[Unified,:].unsqueeze(0),pred[Unified,:].unsqueeze(0)), dim=0)
            save_as_unified_grid(ScanLabelPred, 'Unified Visualization', Unified)

            loss = SegmentationLoss(pred[Unified,:].unsqueeze(0),labels[Unified,:].unsqueeze(0))

            gt = labels[Unified,:].data.cpu().numpy() #the arguments should be w\0 batch index value [C,H,W]
            prd = pred[Unified,:].data.cpu().numpy() #the arguments should be w\0 batch index value [C,H,W]
            MSE = EvalP.mse(gt, prd)
            NMSE = EvalP.nmse(gt, prd)
            PSNR = EvalP.psnr(gt, prd)
            SSIM = EvalP.ssim(gt, prd)

            if args.loss == 'WBCE_DiceLoss':
                writer.add_text('Img Parameters - Test Phase:', 'Dice loss calculation: {:.3}  \nMSE: {:.3}  \nNormalized MSE: {:.3}'
                                           '  \nPSNR: {:.3}  \nSSIM: {:.3}'.format(loss,MSE,NMSE,PSNR,SSIM), Unified)
            else:
                writer.add_text('Img Parameters - Test Phase:', '{} loss calculation: {:.3}  \nMSE: {:.3}  \nNormalized MSE: {:.3}'
                                           '  \nPSNR: {:.3}  \nSSIM: {:.3}'.format(args.loss, loss,MSE,NMSE,PSNR,SSIM), Unified)
            # save_as_embbeded_seg(inputs[Unified,:],labels[Unified,:],pred[Unified,:])

    print('Visualization DONE')

def WriteToTensorboard(metrics, epoch_samples,writer,epoch):

    for k in metrics.keys():
        metrics[k] /= epoch_samples

    # writing the results to TensorboardX
    writer.add_scalar('BCE', metrics['bce'], epoch)
    writer.add_scalar('Dice Loss', metrics['dice'], epoch)
    writer.add_scalar('WBCE and DiceLoss', metrics['WBCE_DiceLoss'], epoch)
    writer.add_scalar('Dice Similarity', 1-metrics['dice'], epoch)
    writer.add_scalar('Tversky Loss', metrics['Tversky'], epoch)
    writer.add_scalar('Tversky Similarity', 1-metrics['Tversky'], epoch)

