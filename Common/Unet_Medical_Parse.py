
import argparse
import pathlib


class Args(argparse.ArgumentParser):
    """
    Defines global default arguments.
    """

    def __init__(self, **overrides):
        """
        Args:
            **overrides (dict, optional): Keyword arguments used to override default argument values
        """

        super().__init__(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.add_argument('--seed', default=42, type=int, help='Seed for random number generators')
        self.add_argument('--resolution', default=320, type=int, help='Resolution of images')

        # Data parameters
        self.add_argument('--challenge', choices=['singlecoil', 'multicoil'], required=True,
                          help='Which challenge')
        self.add_argument('--data-path', type=pathlib.Path, required=True,
                          help='Path to the dataset from Working Directory')
        self.add_argument('--sample-rate', type=float, default=1.,
                          help='Fraction of total volumes to include')

        # Mask parameters
        self.add_argument('--accelerations', nargs='+', default=[4, 8], type=int,
                          help='Ratio of k-space columns to be sampled. If multiple values are '
                               'provided, then one of those is chosen uniformly at random for '
                               'each volume.')
        self.add_argument('--center-fractions', nargs='+', default=[0.08, 0.04], type=float,
                          help='Fraction of low-frequency k-space columns to be sampled. Should '
                               'have the same length as accelerations')

        # Override defaults with passed overrides
        self.set_defaults(**overrides)


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
    parser.add_argument('--savetestfile', type=str,
                        help='Flag if you want to save evaluation images, Y - for saving')

    ### Choose wether the model evaluate or not
    parser.add_argument('--eval', type=bool, default=False,
                        help='Flag as True if you already save the mpdel and would like to skip the trainig phase')
    parser.add_argument('--eval-folder', type=pathlib.Path, default='checkpoints',
                        help='Path where model and results should be loaded')


    # parser.add_argument('--data-split', choices=['val', 'test'], required=True,
    #                     help='Which data partition to run on: "val" or "test"')
    # This feature is dedicate for supervised learning and H5 files
    return parser