
import argparse
import pathlib


class RawDataPreProcessing_Args(argparse.ArgumentParser):
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

        # Data parameters
        self.add_argument('--data-path', type=pathlib.Path, required=True,
                          help='Path to the RawData from Working Directory')
        self.add_argument('--image-path', type=pathlib.Path, required=True,
                          help='Path to where save the coresponding images from Working Directory')
        self.add_argument('--mask-path', type=pathlib.Path, required=True,
                          help='Path to where save the coresponding masks from Working Directory')
        self.add_argument('--scanpreprocess', type=bool, default=False,
                          help='Flag if Preproces for IScan segmentation is needed')


        # Override defaults with passed overrides
        self.set_defaults(**overrides)