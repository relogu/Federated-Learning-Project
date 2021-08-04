#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wen Aug 4 10:37:10 2021

@author: relogu
"""
import pathlib
import pandas as pd
from pathlib import Path
from typing import Union, Dict


def dump_learning_curve(filename: str, round: int, loss: float, accuracy: float):
    """Dump the learning curve.
    The function dumps to the file given by complete path
    (relative or absolute) the row composed by:
    filename,round,loss,accuracy
    If round == 1, the function dumps also the header:
    \"client,round,loss,accuracy\"

    Args:
        filename ([str]): path to file to dump
        round ([int]): current round of the learning
        loss ([float]): current loss of the learning
        accuracy ([float]): current accuracy of the learning
    """
    # get file path
    path_to_file = pathlib.Path(__file__).parent.absolute()
    path_to_file = path_to_file/"output"/(filename+".dat")
    # touching file
    path_to_file.touch()
    with open(path_to_file, "a") as outfile:
        # write line(s)
        if round == 1:
            print("client,round,loss,accuracy", file=outfile)
        print(filename+","+str(round)+","+str(loss) +
              ","+str(accuracy), file=outfile)


def dump_result_dict(filename: str,
                     result: Dict,
                     path_to_out: Union[Path, str] = None,
                     verbose: int = 0):
    """Dump the result dictionary.
    The function dumps to the file given by complete path
    (relative or absolute) the row composed by results.values(),
    separated by a comma
    If result[\'round\'] == 1, the function dumps also the headers of 
    the dictionary, contained in results.keys(), separated by a comma.

    Args:
        filename ([str]): path to file to dump
        result ([Dict]): dictionary containing the values to dump
    """
    if result.get('round') is None:
        raise KeyError("The mandatory key \'round\' is missing.")
    # get file path
    if path_to_out is None:
        path_to_out = pathlib.Path(__file__).parent.parent.absolute()/'output'
    else:
        path_to_out = pathlib.Path(path_to_out)
    path_to_file = path_to_out/(filename+".dat")
    # touching file
    path_to_file.touch()
    if verbose > 0:
        print("Dumping results at "+str(path_to_file))
    with open(path_to_file, "a") as outfile:
        # write line(s)
        if result['round'] == 1:
            print(','.join(list(result.keys())), file=outfile)
        print(','.join(map(str, list(result.values()))), file=outfile)


def dump_pred_dict(filename: str,
                   pred: Dict,
                   path_to_out: Union[Path, str] = None,
                   verbose: int = 0):
    # get file path
    if path_to_out is None:
        path_to_out = pathlib.Path(__file__).parent.parent.absolute()/'output'
    else:
        path_to_out = pathlib.Path(path_to_out)
    path_to_file = path_to_out/(filename+".csv")
    if verbose > 0:
        print("Dumping results at "+str(path_to_file))
    df = pd.DataFrame(pred)
    df.to_csv(path_to_file)


def dump_labels_euromds(labels,
                        name: str = None,
                        path_to_data: Union[Path, str] = None):
    # set the path
    if path_to_data is None:
        parent = pathlib.Path(__file__).parent.parent.absolute()
        data_folder = parent/'data'/'euromds'
    else:
        data_folder = path_to_data
    # set the filename
    if name is None:
        filename = 'labels.csv'
    else:
        filename = 'labels_'+name+'.csv'
    # get the main dataframe
    main_df = pd.read_csv(data_folder/'dataFrame.csv')
    # selected the IDs
    main_df = main_df[main_df.columns[0]]
    main_df.replace('EUROMDS', '')
    # building the dataframe
    main_df = pd.DataFrame({'ID': main_df.asint(),
                            'label': labels})
    # saving the file
    main_df.to_csv(data_folder/filename)
