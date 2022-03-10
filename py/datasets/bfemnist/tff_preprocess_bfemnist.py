import argparse
import pathlib
import collections
import json
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff


def preprocess(dataset):

  def batch_format_fn(element):
    """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
    return collections.OrderedDict(
        x=tf.reshape(element['pixels'], [-1, 784]),
        y=tf.reshape(element['label'], [-1, 1]))

  return dataset.repeat(1).shuffle(100, seed=1).map(batch_format_fn).prefetch(10)

def get_parser():
    parser = argparse.ArgumentParser(
        description="Script for building the folder containing Binary Federated EMNIST (BFEMNIST)")
    parser.add_argument("--only-digits",
                        dest="only_digits",
                        type=bool,
                        default=True,
                        help="Flag for using only digits")
    parser.add_argument("--cache-folder",
                        dest="cache_folder",
                        type=str,
                        default=None,
                        help="Folder to caching the download of images")
    parser.add_argument("--out-folder",
                        dest="out_folder",
                        type=str,
                        default=None,
                        help="Folder to output images")
    return parser


if __name__ == "__main__":

    args = get_parser().parse_args()
    # Define output folder
    if args.out_folder is None:
        path_to_out = pathlib.Path(__file__).parent.parent.absolute()/'output'
    else:
        path_to_out = pathlib.Path(args.out_folder)
    # Define caching folder
    cache_folder = None
    if args.cache_folder is not None:
        cache_folder = pathlib.Path(args.out_folder)
    print('Output folder {}'.format(path_to_out))
    (path_to_out/'bfemnist'/'train').mkdir(parents=True, exist_ok=True)
    (path_to_out/'bfemnist'/'test').mkdir(parents=True, exist_ok=True)
    
    train, test = tff.simulation.datasets.emnist.load_data(
        only_digits=args.only_digits, cache_dir=cache_folder
    )
    
    for client in train.client_ids:
        c_train = preprocess(train.create_tf_dataset_for_client(client))
        c_test = preprocess(test.create_tf_dataset_for_client(client))
        c_x_train = np.array([d['x'].numpy() for d in c_train])
        c_y_train = np.array([d['y'].numpy() for d in c_train])
        c_x_test = np.array([d['x'].numpy() for d in c_test])
        c_y_test = np.array([d['y'].numpy() for d in c_test])
        c_train = {
            'x': np.round(c_x_train).tolist(),
            'y': np.round(c_y_train).tolist(),
        }
        with open(path_to_out/'bfemnist'/'train'/'{}.json'.format(client), 'x') as file:
            json.dump(c_train, file)
        c_test = {
            'x': np.round(c_x_test).tolist(),
            'y': np.round(c_y_test).tolist(),
        }
        with open(path_to_out/'bfemnist'/'test'/'{}.json'.format(client), 'x') as file:
            json.dump(c_test, file)
        