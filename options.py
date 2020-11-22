import argparse


class Options:

    def __init__(self):
        self.options = None

        self.parser = argparse.ArgumentParser()

        # Universal Options

        self.parser.add_argument('--config_path',
                                 help='path to config file containing dataset info',
                                 type=str,
                                 default='paths_config.yaml')

        self.parser.add_argument('--mode',
                                 help='training or inference mode',
                                 type=str,
                                 choices=['train', 'inference'],
                                 default='train')

        self.parser.add_argument('--height',
                                 help='height of input images',
                                 type=int,
                                 default=320)
        self.parser.add_argument('--width',
                                 help='width of input images',
                                 type=int,
                                 default=608)

        self.parser.add_argument('--disable_synthetic_augmentation',
                                 action='store_true')

        # Network Options

        self.parser.add_argument('--network',
                                 choices=['hourglass'],
                                 default='hourglass')

        self.parser.add_argument('--max_disparity',
                                 help='maximum disparity',
                                 type=int,
                                 default=192)

        self.parser.add_argument('--psm_no_SPP',
                                 help='whether to use spatial pyramid pooling from PSM',
                                 action='store_true')
        self.parser.add_argument('--big_SPP',
                                 help='standard PSM SPP module breaks for lower resolution images,'
                                      'so by default we use smaller windows',
                                 action='store_true')
        self.parser.add_argument('--disable_normalisation',
                                 action='store_true')
        self.parser.add_argument('--disable_sharpening',
                                 action='store_true')
        self.parser.add_argument('--disable_background',
                                 action='store_true')
        self.parser.add_argument('--monodepth_model',
                                 type=str,
                                 default='midas',
                                 choices=['midas', 'megadepth'])
        self.parser.add_argument('--data_sampling',
                                 type=float,
                                 default=1.0)

        # Training Options

        self.parser.add_argument('--training_datasets',
                                 help='datasets to train from',
                                 nargs='+',
                                 choices=['ADE20K', 'diode', 'diw', 'mapillary', 'mscoco',
                                          'sceneflow', 'kitti2015'],
                                 default=['ADE20K', 'diode', 'diw', 'mapillary', 'mscoco'])

        self.parser.add_argument('--training_steps',
                                 help='number of steps to train for',
                                 type=int,
                                 default=250000)

        self.parser.add_argument('--log_freq',
                                 help='sets the frequency of logs to tensorboard',
                                 type=int,
                                 default=250)

        self.parser.add_argument('--val_batches',
                                 help='number of validation batches to run and average over',
                                 type=int,
                                 default=1)

        self.parser.add_argument('--batch_size',
                                 help='number of images in each batch',
                                 type=int,
                                 default=2)

        self.parser.add_argument('--lr',
                                 help='the learning rate',
                                 type=float,
                                 default=1e-3)

        self.parser.add_argument('--lr_step_size',
                                 type=int,
                                 default=5)

        self.parser.add_argument('--num_workers',
                                 help=' number of workers for dataloaders',
                                 type=int,
                                 default=6)

        self.parser.add_argument('--model_name',
                                 help='the name of the model for saving',
                                 type=str,
                                 default='model')

        self.parser.add_argument('--log_path',
                                 help='the path to save tensorboard events and trained models to',
                                 type=str,
                                 default='./logs')

        self.parser.add_argument('--start_step',
                                 help='step in training to start from - allows continuing from'
                                      'loaded model',
                                 default=0,
                                 type=int)

        # Test Options

        self.parser.add_argument('--load_path',
                                 help='the model path to load from',
                                 type=str)
        self.parser.add_argument('--save_disparities',
                                 help='if set, save all computed disparities',
                                 action='store_true')
        self.parser.add_argument('--test_data_types',
                                 choices=['eth3d', 'middlebury', 'kitti2015', 'kitti2012',
                                          'flicker', 'kitti2015submission', 'sceneflow'],
                                 nargs='+',
                                 default=['eth3d', 'middlebury', 'kitti2015', 'kitti2012'])

    def parse(self):
        """ Parse arguments """
        self.options = self.parser.parse_args()
        return self.options
