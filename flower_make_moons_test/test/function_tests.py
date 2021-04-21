#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 14:57:21 2021

@author: relogu
"""
import math
import pathlib
import sys
import unittest
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.testing.compare import compare_images
sys.path.append('../')
import common_fn as my_fn

class TestMethods(unittest.TestCase):

    def test_rotate_moons(self):
        x = np.array([[0.0,0.0],[0.0,0.0]])
        theta = math.pi/10
        theta1 = math.pi/12
        x_f = my_fn.rotate_moons(theta, x)
        self.assertAlmostEqual(x_f.all(), x.all())
        x = np.array([[1.0,1.0],[1.0,1.0]])
        x_f = my_fn.rotate_moons(-theta, my_fn.rotate_moons(theta, x))
        self.assertAlmostEqual(x_f.all(), x.all())
        x = np.array([[1.0,1.0],[1.0,1.0]])
        x = my_fn.rotate_moons(theta1, my_fn.rotate_moons(theta, x))
        x_f = my_fn.rotate_moons(theta, my_fn.rotate_moons(theta1, x))
        self.assertAlmostEqual(x_f.all(), x.all())
        x = np.array([[1.0,1.0],[1.0,1.0]])
        x = my_fn.rotate_moons(2*math.pi, x)
        self.assertAlmostEqual(x_f.all(), x.all())

    def test_translate_moons(self):
        x = np.array([[0.0,0.0],[0.0,0.0]])
        x_f = my_fn.translate_moons(0.0, 0.0, x)
        self.assertAlmostEqual(x_f.all(), x.all())
        x_f = my_fn.translate_moons(-1.0, -1.0, my_fn.translate_moons(1.0, 1.0, x))
        self.assertAlmostEqual(x_f.all(), x.all())
        x_f = my_fn.translate_moons(2.0, 2.0, my_fn.translate_moons(1.0, 1.0, x))
        x = my_fn.translate_moons(3.0, 3.0, x)
        self.assertAlmostEqual(x_f.all(), x.all())
        x = np.array([[0.0,0.0],[0.0,0.0]])
        x_f = my_fn.translate_moons(2.0, 2.0, my_fn.translate_moons(1.0, 1.0, x))
        x = my_fn.translate_moons(1.0, 1.0, my_fn.translate_moons(2.0, 2.0, x))
        self.assertAlmostEqual(x_f.all(), x.all())

    def test_dump_learning_curve(self):
        path_to_file = pathlib.Path(__file__).parent.absolute()
        filename = "/../output/abc.dat"
        my_fn.dump_learning_curve("abc", 1, 1, 1)
        test_file = str(path_to_file)+"/test0.dat"
        file_to_test = str(path_to_file)+filename
        test_file = open(test_file).read()
        file_to_test = open(file_to_test).read()
        os.remove(str(path_to_file)+filename)
        self.assertMultiLineEqual( test_file, file_to_test, "not equal files")
        my_fn.dump_learning_curve("abc", 1, 1, 1)
        my_fn.dump_learning_curve("abc", 2, 2, 2)
        test_file = str(path_to_file)+"/test1.dat"
        file_to_test = str(path_to_file)+filename
        test_file = open(test_file).read()
        file_to_test = open(file_to_test).read()
        os.remove(filename[1:])
        self.assertMultiLineEqual( test_file, file_to_test, "not equal files")
        
    def test_plot_points(self):
        x_train = np.array([[0.0,0.0],[2.0,2.0]])
        y_train = np.array([0, 1])
        x_test = np.array([[-1.0,-1.0],[1.0,1.0]])
        y_test = np.array([0, 1])
        images_path = '../output/*.png'
        path_to_pass = '../output'
        my_fn.plot_client_dataset(0, x_train, y_train, x_test, y_test, path_to_pass)
        files =  glob.glob(images_path)
        compare_images('test_data_client_0.png', files[0], 1.0)
        os.remove(files[0])
        x_test = np.array([[-1.0,-1.0],[1.0,1.0]])
        y_test = np.array([0, 1])
        model = my_fn.create_keras_model()
        model.load_weights("model.h5")
        my_fn.plot_decision_boundary(model, x_test, y_test,
                                     client_id=0, fed_iter=1, path=path_to_pass)
        files =  glob.glob(images_path)
        compare_images('test_dec_bound_c0_e1.png', files[0], 1.0)
        os.remove(files[0])
        my_fn.plot_decision_boundary(model, x_test, y_test,
                                     client_id=0, path=path_to_pass)
        files =  glob.glob(images_path) 
        compare_images('test_dec_bound_c0.png', files[0], 1.0)
        os.remove(files[0])
        my_fn.plot_decision_boundary(model, x_test, y_test,
                                     path=path_to_pass)
        files =  glob.glob(images_path)
        compare_images('test_dec_bound_nofed.png', files[0], 1.0)
        os.remove(files[0])
        

if __name__ == '__main__':
    unittest.main()