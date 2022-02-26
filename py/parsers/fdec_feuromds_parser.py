import argparse

def fdec_feuromds_parser():
    parser = argparse.ArgumentParser(
        description='Federeated DEC for FEMNIST'
    )
    ## EUROMDS dataset
    parser.add_argument(
        '--groups',
        dest='groups',
        required=False,
        action='append',
        help='which groups of variables to use for EUROMDS dataset')
    parser.add_argument(
        '--ex_col',
        dest='ex_col',
        required=False,
        action='append',
        help='which columns to exclude for EUROMDS dataset')
    parser.add_argument(
        '--fill-nans',
        dest='fill_nans',
        required=False,
        type=int,
        default=2044,
        help='maximum number of NaNs for accepting a column')
    parser.add_argument(
        '--balance',
        dest='balance',
        required=False,
        type=int,
        default=-1,
        help='skewness of the distribution of samples along clients, if negative they are uniformly distributed')
    parser.add_argument(
        '--n-local-epochs',
        dest='n_local_epochs',
        required=False,
        type=int,
        default=1,
        help='set the number of local epochs')
    parser.add_argument(
        '--ae-batch-size',
        dest='ae_batch_size',
        required=False,
        type=int,
        default=8,
        help='batch size used for SDAE training and DEC clustering'
    )
    ## SDAE params
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
        '--linears',
        dest='linears',
        required=False,
        type=str,
        default='dec',
        choices=['dec', 'google', 'curves'],
        help='architecture of linears in SDAE'
    )
    parser.add_argument(
        '--ae-opt',
        dest='ae_opt',
        required=False,
        type=str,
        default='sgd',
        choices=['sgd', 'adam', 'yogi'],
        help='name of the SDAE optimizer'
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
        default=500,
        help='federated epochs to run in SDAE training'
    )
    ## KMeans parameters
    parser.add_argument(
        '--n-clusters',
        dest='n_clusters',
        required=False,
        type=int,
        default=6,
        help='number of cluster to search for'
    )
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
    parser.add_argument(
        '--kmeans-agg',
        dest='kmeans_agg',
        required=False,
        type=str,
        default='max_min',
        choices=['max_min', 'double_kmeans', 'random', 'random_weighted'],
        help='aggregation method for centroids'
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
        default=5,
        help='alpha parameter of DEC model'
    )
    ## DEC training
    parser.add_argument(
        '--dec-epochs',
        dest='dec_epochs',
        required=False,
        type=int,
        default=25,
        help='number of federated epochs for DEC training'
    )
    parser.add_argument(
        '--dec-batch-size',
        dest='dec_batch_size',
        required=False,
        type=int,
        default=64,
        help='batch size used for clustering step training'
    )
    parser.add_argument(
        '--dec-lr',
        dest='dec_lr',
        required=False,
        type=float,
        default=1e-2,
        help='local learning rate used for clustering step training'
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
        default=6,
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
        '--in-folder',
        dest='in_folder',
        required=False,
        type=str,
        default=None,
        help='path to input folder'
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
        default=10,
        help='number of clients that participate the federated training'
    )
    parser.add_argument(
        '--dump-metrics',
        dest='dump_metrics',
        required=False,
        type=bool,
        default=True,
        help='flag to set whether to dump metrics or not along training'
    )
    parser.add_argument(
        '--verbose',
        dest='verbose',
        required=False,
        type=bool,
        default=True,
        help='flag to set verbosity'
    )
    return parser
