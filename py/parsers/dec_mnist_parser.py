import argparse


def dec_mnist_parser():
    parser = argparse.ArgumentParser(
        description="Original DEC Training Script")
    parser.add_argument("--hardware_acc",
                        dest="cuda_flag",
                        action='store_true',
                        help="Flag for hardware acceleration using cuda (if available)")
    parser.add_argument('--gpus',
                        dest='gpus',
                        required=False,
                        nargs='+',
                        type=int,
                        default=0,
                        help='Id for the gpu(s) to use')
    parser.add_argument("--lim_cores",
                        dest="lim_cores",
                        action='store_true',
                        help="Flag for limiting cores")
    parser.add_argument("--folder",
                        dest="out_folder",
                        type=type(str('')),
                        help="Folder to output images")
    parser.add_argument('--update_interval',
                        dest='update_interval',
                        required=False,
                        type=int,
                        default=160,
                        action='store',
                        help='set the update interval for the clusters distribution')
    parser.add_argument('--seed',
                        dest='seed',
                        required=False,
                        type=int,
                        default=51550,
                        action='store',
                        help='set the seed for the random generator of the whole dataset')
    parser.add_argument('-v', '--verbose',
                        dest='verbose',
                        required=False,
                        action='store_true',
                        help='Flag for verbosity')
    return parser
