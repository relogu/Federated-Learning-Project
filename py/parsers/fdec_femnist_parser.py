import argparse

def fdec_femnist_parser():
    parser = argparse.ArgumentParser(
        description='Federeated DEC for FEMNIST'
    )
    parser.add_argument(
        '--n-local-epochs',
        dest='n_local_epochs',
        required=False,
        type=int,
        default=1,
        help='set the number of local epochs'
    )
    parser.add_argument(
        '--ae-batch-size',
        dest='ae_batch_size',
        required=False,
        type=int,
        default=256,
        help='Batch size used for TSAE training'
    )
    ## SDAE params
    parser.add_argument(
        '--linears',
        dest='linears',
        required=False,
        type=str,
        default='dec',
        choices=['dec', 'google', 'curves'],
        help='architecture of linears in SDAE'
    )
    parser.add_argument(
        '--noising',
        dest='noising',
        required=False,
        type=float,
        default=0.0,
        help='stddev of gaussian noising in input of SDAE training'
    )
    parser.add_argument(
        '--corruption',
        dest='corruption',
        required=False,
        type=float,
        default=0.0,
        help='rate of corruption in input of SDAE training'
    )
    parser.add_argument(
        '--hidden-dropout',
        dest='hidden_dropout',
        required=False,
        type=float,
        default=0.0,
        help='rate of dropout for hidden layers in SDAE training'
    )
    parser.add_argument(
        '--hidden-dimensions',
        dest='hidden_dimensions',
        required=False,
        type=int,
        default=10,
        help='number of hidden dimension of the feature space'
    )
    ## SDAE training
    parser.add_argument(
        '--optimizer',
        dest='optimizer',
        required=False,
        type=str,
        default='sgd',
        choices=['sgd', 'adam', 'yogi'],
        help='Optimizer to use on training TSAE and DEC'
    )
    parser.add_argument(
        '--ae-lr',
        dest='ae_lr',
        required=False,
        type=float,
        default=None,
        help='learning rate for SDAE optimizer, if None the default (choosen via hyperparamete tuning) is set'
    )
    parser.add_argument(
        '--ae-epochs',
        dest='ae_epochs',
        required=False,
        type=int,
        default=300,
        help='federated epochs to run in SDAE training'
    )
    ## KMeans parameters
    parser.add_argument(
        '--n-init',
        dest='n_init',
        required=False,
        type=int,
        default=20,
        help='number of inititialization for KMeans fit'
    )
    parser.add_argument(
        '--max-iter',
        dest='max_iter',
        required=False,
        type=int,
        default=300,
        help='maximum number of iterations for KMeans fit'
    )
    parser.add_argument(
        '--use-emp-centroids',
        dest='use_emp_centroids',
        required=False,
        type=bool,
        default=False,
        help='flag to set whether to use or not empirical centroids'
    )
    ## scaler
    parser.add_argument(
        '--scaler',
        dest='scaler',
        required=False,
        type=str,
        default=None,
        choices=['standard', 'normal-l1', 'normal-l2'],
        help='name fo the scaler before to run KMeans'
    )
    ## DEC param
    parser.add_argument(
        '--alpha',
        dest='alpha',
        required=False,
        type=int,
        default=1,
        help='alpha parameter of DEC model'
    )
    ## DEC training
    parser.add_argument(
        '--dec-epochs',
        dest='dec_epochs',
        required=False,
        type=int,
        default=20,
        help='number of federated epochs for DEC training'
    )
    parser.add_argument(
        '--dec-batch-size',
        dest='dec_batch_size',
        required=False,
        type=int,
        default=None,
        help='Batch size used for DEC training, if None the best is chosen (from hyperparameter tuning)'
    )
    parser.add_argument(
        '--dec-lr',
        dest='dec_lr',
        required=False,
        type=float,
        default=None,
        help='Learning rate for DEC optimizer, if None the default (choosen via hyperparamete tuning) is set'
    )
    ## general
    parser.add_argument(
        '--seed',
        dest='seed',
        required=False,
        type=int,
        default=51550,
        help='seed for initializing random generators'
    )
    parser.add_argument(
        '--n-cpus',
        dest='n_cpus',
        required=False,
        type=int,
        default=1,
        help='set the number of cpus per client to set ray resources'
    )
    parser.add_argument(
        '--out-folder',
        dest='out_folder',
        required=False,
        type=str,
        default=None,
        help='path to output folder'
    )
    parser.add_argument(
        '--data-folder',
        dest='data_folder',
        required=False,
        type=str,
        default=None,
        help='path to data folder'
    )
    parser.add_argument(
        '--n-clients',
        dest='n_clients',
        required=False,
        type=int,
        default=100,
        help='number of clients that participate the federated training'
    )
    parser.add_argument(
        '--min-clients',
        dest='min_clients',
        required=False,
        type=int,
        default=-1,
        help='set the minimum number of clients available per round'
    )
    parser.add_argument(
        '--dump-metrics',
        dest='dump_metrics',
        required=False,
        type=bool,
        default=False,
        help='flag to set whether to dump metrics or not along training'
    )
    parser.add_argument(
        '--verbose',
        dest='verbose',
        required=False,
        type=bool,
        default=False,
        help='flag to set verbosity'
    )
    return parser
