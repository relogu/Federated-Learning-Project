import argparse
from py.dec.torch.utils import LOSS_DICT, BINARY_MOD_LOSS_DICT

def fdec_feuromds_parser():
    parser = argparse.ArgumentParser(
        description='Federeated DEC for FEMNIST'
    )
    ## EUROMDS data set parameters
    parser.add_argument(
        '--data-folder',
        dest='data_folder',
        required=False,
        type=str,
        default=None,
        help='Path to data folder, if None deafalt is used'
    )
    parser.add_argument(
        '--groups',
        dest='groups',
        required=False,
        action='append',
        help='Which groups of variables to use for EUROMDS dataset')
    parser.add_argument(
        '--ex_col',
        dest='ex_col',
        required=False,
        action='append',
        help='Which columns to exclude for EUROMDS dataset')
    parser.add_argument(
        '--fill-nans',
        dest='fill_nans',
        required=False,
        type=int,
        default=2044,
        help='Maximum number of NaNs for accepting a column')
    parser.add_argument(
        '--balance',
        dest='balance',
        required=False,
        type=int,
        default=-1,
        help='Skewness of the distribution of samples along clients, if negative they are uniformly distributed')
    ## TSAE params
    parser.add_argument(
        '--linears',
        dest='linears',
        required=False,
        type=str,
        default='dec',
        choices=['dec', 'google', 'curves'],
        help='Architecture of linears in TSAE'
    )
    parser.add_argument(
        '--activation',
        dest='activation',
        required=False,
        type=str,
        default='relu',
        choices=['relu', 'sigmoid'],
        help='Activation function for hidden nodes in TSAE'
    )
    parser.add_argument(
        '--final-activation',
        dest='final_activation',
        required=False,
        type=str,
        default='relu',
        choices=['relu', 'sigmoid'],
        help='Final activation function for TSAE'
    )
    parser.add_argument(
        '--noising',
        dest='noising',
        required=False,
        type=float,
        default=0.0,
        help='Standard deviation of gaussian noising in input of TSAE training'
    )
    parser.add_argument(
        '--corruption',
        dest='corruption',
        required=False,
        type=float,
        default=0.0,
        help='Rate of corruption in input of TSAE training'
    )
    parser.add_argument(
        '--hidden-dropout',
        dest='hidden_dropout',
        required=False,
        type=float,
        default=0.0,
        help='Rate of dropout for hidden layers in TSAE'
    )
    parser.add_argument(
        '--hidden-dimensions',
        dest='hidden_dimensions',
        required=False,
        type=int,
        default=10,
        help='Number of hidden dimension of the feature space'
    )
    ## optimizer for both TSAE and DEC training
    parser.add_argument(
        '--optimizer',
        dest='optimizer',
        required=False,
        type=str,
        default='sgd',
        choices=['sgd', 'adam', 'yogi'],
        help='Name of the optimizer in training both TSAE and DEC'
    )
    ## TSAE training
    parser.add_argument(
        '--ae-batch-size',
        dest='ae_batch_size',
        required=False,
        type=int,
        default=8,
        help='Batch size used for TSAE training'
    )
    parser.add_argument(
        '--main-loss',
        dest='main_loss',
        required=False,
        type=str,
        default='mse',
        choices=list(LOSS_DICT.keys()),
        help='Name of the main loss function in training TSAE'
    )
    parser.add_argument(
        '--mod-loss',
        dest='mod_loss',
        required=False,
        type=str,
        default=None,
        choices=list(BINARY_MOD_LOSS_DICT.keys())+[None],
        help='Name of the mod loss function in training TSAE'
    )
    parser.add_argument(
        '--beta',
        dest='beta',
        required=False,
        type=float,
        default=0.0,
        help='Fraction of the mod loss contribution w.r.t. the main loss'
    )
    parser.add_argument(
        '--ae-lr',
        dest='ae_lr',
        required=False,
        type=float,
        default=None,
        help='Learning rate for TSAE optimizer, if None the default (choosen via hyperparamete tuning) is set'
    )
    parser.add_argument(
        '--pretrain-epochs',
        dest='pretrain_epochs',
        required=False,
        type=int,
        default=150,
        help='Number of epochs for pretraining TSAE'
    )
    parser.add_argument(
        '--finetune-epochs',
        dest='finetune_epochs',
        required=False,
        type=int,
        default=150,
        help='Number of epochs for finetuning TSAE, used with noising only'
    )
    # Train dec flag
    parser.add_argument(
        '--train-dec',
        dest='train_dec',
        required=False,
        type=bool,
        default=True,
        help='Flag to set whether to train DEC or not'
    )
    # Number of cluster to search for
    parser.add_argument(
        '--n-clusters',
        dest='n_clusters',
        required=False,
        type=int,
        default=6,
        help='Number of cluster to search for'
    )
    ## KMeans parameters
    parser.add_argument(
        '--n-init',
        dest='n_init',
        required=False,
        type=int,
        default=20,
        help='Number of inititialization for KMeans fit'
    )
    parser.add_argument(
        '--kmeans-agg',
        dest='kmeans_agg',
        required=False,
        type=str,
        default='max_min',
        choices=['max_min', 'double_kmeans', 'random', 'random_weighted'],
        help='Set the aggregation method for centroids'
    )
    ## scaler
    parser.add_argument(
        '--scaler',
        dest='scaler',
        required=False,
        type=str,
        default=None,
        choices=['standard', 'normal-l1', 'normal-l2'],
        help='Name fo the scaler before to run KMeans'
    )
    ## DEC param
    parser.add_argument(
        '--alpha',
        dest='alpha',
        required=False,
        type=int,
        default=5,
        help='Alpha parameter of DEC model (best should be n_cluster-1)'
    )
    ## DEC training
    parser.add_argument(
        '--dec-epochs',
        dest='dec_epochs',
        required=False,
        type=int,
        default=25,
        help='Number of federated epochs for DEC training'
    )
    parser.add_argument(
        '--dec-batch-size',
        dest='dec_batch_size',
        required=False,
        type=int,
        default=64,
        help='Batch size used for DEC training'
    )
    parser.add_argument(
        '--dec-lr',
        dest='dec_lr',
        required=False,
        type=float,
        default=None,
        help='Learning rate for DEC optimizer, if None the default (choosen via hyperparamete tuning) is set'
    )
    ## General
    parser.add_argument(
        '--n-local-epochs',
        dest='n_local_epochs',
        required=False,
        type=int,
        default=1,
        help='set the number of local epochs')
    parser.add_argument(
        '--seed',
        dest='seed',
        required=False,
        type=int,
        default=51550,
        help='Seed for initializing random generators'
    )
    parser.add_argument(
        '--n-cpus',
        dest='n_cpus',
        required=False,
        type=int,
        default=6,
        help='Set the number of cpus per client to set ray resources'
    )
    parser.add_argument(
        '--out-folder',
        dest='out_folder',
        required=False,
        type=str,
        default=None,
        help='Path to output folder'
    )
    parser.add_argument(
        '--n-clients',
        dest='n_clients',
        required=False,
        type=int,
        default=10,
        help='Number of clients that participate the federated training'
    )
    parser.add_argument(
        '--dump-metrics',
        dest='dump_metrics',
        required=False,
        type=bool,
        default=True,
        help='Flag to set whether to dump metrics or not along training'
    )
    parser.add_argument(
        '--verbose',
        dest='verbose',
        required=False,
        type=bool,
        default=True,
        help='Flag to set verbosity'
    )
    return parser
