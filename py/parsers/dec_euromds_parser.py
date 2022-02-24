import argparse
# from py.losses import get_keras_loss_names

"""
cuda, gpu_id, batch_size, pretrain_epochs, finetune_epochs, testing_mode, out_folder,
         glw_pretraining, is_tied, ae_main_loss, ae_mod_loss, alpha, input_do, hidden_do, beta,
         gaus_noise, ae_opt, lr, path_to_data
         
    config = {
        'linears': 'dec',
        'f_dim': 10,
        'activation': ReLU(),
        'final_activation': Sigmoid(),
        'dropout': 0.0,
        'epochs': 150,
        'n_clusters': 6,
        'ae_batch_size': 8,
        'update_interval': 50,
        'optimizer': 'yogi',
        'lr': None,
        'main_loss': 'mse',
        'mod_loss': 'bce+dice',
        'beta': 0.4,
        'corruption': 0.0,
        'noising': 0.0,
        'train_dec': 'yes',
        'alpha': 1,
        'scaler': 'standard',
        'use_emp_centroids': 'yes',
    }
"""

def dec_euromds_parser():
    
    parser = argparse.ArgumentParser(description="UDE Training Script")
    parser.add_argument("--batch_size",
                        dest="batch_size",
                        default=64,
                        type=int,
                        help="Batch size")
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
    parser.add_argument('--groups', dest='groups',
                        required=True,
                        action='append',
                        help='which groups of variables to use for EUROMDS dataset')
    parser.add_argument('--ex_col',
                        dest='ex_col',
                        required=True,
                        action='append', help='which columns to exclude for EUROMDS dataset')
    parser.add_argument('--fill',
                        dest='fill',
                        required=False,
                        action='store_true',
                        help='Flag for fill NaNs in dataset')
    parser.add_argument("--n_clusters",
                        dest="n_clusters",
                        default=10,
                        type=int, help="Define the number of clusters to identify")
    parser.add_argument('--ae_epochs',
                        dest='ae_epochs',
                        required=False,
                        type=int,
                        default=50000,
                        action='store',
                        help='number of epochs for the autoencoder pre-training')
    parser.add_argument('--ae_loss',
                        dest='ae_loss',
                        required=True,
                        type=type(''),
                        default='mse',
                        # choices=get_keras_loss_names(),
                        action='store',
                        help='Loss function for autoencoder training')
    parser.add_argument('--cl_epochs',
                        dest='cl_epochs',
                        required=False,
                        type=int,
                        default=10000,
                        action='store',
                        help='number of epochs for the clustering step')
    parser.add_argument('--binary',
                        dest='binary',
                        required=False,
                        action='store_true',
                        help='Flag for using probabilistic binary neurons')
    parser.add_argument('--tied',
                        dest='tied',
                        required=False,
                        action='store_true',
                        help='Flag for using tied layers in autoencoder')
    parser.add_argument('--plotting',
                        dest='plotting',
                        required=False,
                        action='store_true',
                        help='Flag for plotting confusion matrix')
    parser.add_argument('--dropout',
                        dest='dropout',
                        type=float,
                        default=0.20,
                        required=False,
                        action='store',
                        help='Flag for dropout layer in autoencoder')
    parser.add_argument('--ran_flip',
                        dest='ran_flip',
                        type=float,
                        default=0.20,
                        required=False,
                        action='store',
                        help='Flag for RandomFlipping layer in autoencoder')
    parser.add_argument('--ortho',
                        dest='ortho',
                        required=False,
                        action='store_true',
                        help='Flag for orthogonality regularizer in autoencoder (tied only)')
    parser.add_argument('--u_norm',
                        dest='u_norm',
                        required=False,
                        action='store_true',
                        help='Flag for unit norm constraint in autoencoder (tied only)')
    parser.add_argument('--cl_lr',
                        dest='cl_lr',
                        required=False,
                        type=float,
                        default=0.001,
                        action='store',
                        help='clustering model learning rate')
    parser.add_argument('--update_interval',
                        dest='update_interval',
                        required=False,
                        type=int,
                        default=20,
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
