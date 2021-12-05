import argparse


def dec_mnist_parser():
    parser = argparse.ArgumentParser(
        description='MNIST Training Script')
    parser.add_argument('--hardware_acc',
                        dest='cuda_flag',
                        action='store_true',
                        help='Flag for hardware acceleration using cuda (if available)')
    parser.add_argument('--gpus',
                        dest='gpus',
                        nargs='+',
                        type=int,
                        default=0,
                        help='Id for the gpu(s) to use')
    parser.add_argument('--lim_cores',
                        dest='lim_cores',
                        action='store_true',
                        help='Flag for limiting cores')
    parser.add_argument('--folder',
                        dest='out_folder',
                        type=type(str('')),
                        help='Folder to output images')
    parser.add_argument('--seed',
                        dest='seed',
                        type=int,
                        default=51550,
                        action='store',
                        help='set the seed for the random generator of the whole dataset')
    parser.add_argument('-v', '--verbose',
                        dest='verbose',
                        action='store_true',
                        help='Flag for verbosity')
    # AE arguments
    # TODO: descriptions
    parser.add_argument('--tied',
                        dest='tied',
                        action='store_true',
                        help='')
    parser.add_argument('--u_norm',
                        dest='u_norm',
                        action='store_true',
                        help='')
    parser.add_argument('--ortho',
                        dest='ortho',
                        action='store_true',
                        help='')
    parser.add_argument('--uncoll',
                        dest='uncoll',
                        action='store_true',
                        help='')
    parser.add_argument('--use_bias',
                        dest='use_bias',
                        action='store_true',
                        help='')
    parser.add_argument('--dropout',
                        dest='dropout',
                        type=float,
                        default=0.2,
                        action='store',
                        help='')
    parser.add_argument('--noise',
                        dest='noise',
                        type=float,
                        default=0.5,
                        action='store',
                        help='')
    # Clustering arguments
    parser.add_argument('--update_interval',
                        dest='update_interval',
                        type=int,
                        default=160,
                        action='store',
                        help='set the update interval for the clusters distribution')
    return parser
