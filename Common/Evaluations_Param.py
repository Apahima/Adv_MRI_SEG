import numpy as np
# from skimage.measure import compare_psnr
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

import torch
from torch import nn


def mse(gt, pred):
    """ Compute Mean Squared Error (MSE) """
    return np.mean((gt - pred) ** 2)


def nmse(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return compare_psnr(gt, pred, data_range=gt.max())


def ssim(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM). """
    return structural_similarity(
        gt.transpose(1, 2, 0), pred.transpose(1, 2, 0), multichannel=True, data_range=gt.max()
    )



def evaluation():
    input = torch.rand((3, 1, 32, 32, 32))
    model = nn.Conv3d(1, 1, 3, padding=1)
    target = torch.randint(0, 1, (3, 1, 32, 32, 32)).float()
    # target = make_one_hot(target, num_classes=4)

    mse(input.data.cpu().numpy(),target.data.cpu().numpy())

    # input = torch.zeros((1, 2, 32, 32, 32))
    # input[:, 0, ...] = 1
    # target = torch.ones((1, 1, 32, 32, 32)).long()
    # target_one_hot = make_one_hot(target, num_classes=2)
    # # print(target_one_hot.size())
    # criterion = DiceLoss()
    # loss = criterion(input, target_one_hot)
    # print(loss.item())


if __name__ == '__main__':
    evaluation()