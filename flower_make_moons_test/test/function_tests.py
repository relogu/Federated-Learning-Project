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
path = pathlib.Path(__file__).parent.absolute()
path_to_test = str(path)
path_parent = str(path.parent)
sys.path.append(path_parent)
import common_fn as my_fn

# removing files in the output folder if present
files = glob.glob(path_parent+'/output/*')
for file in files:
    os.remove(file)

class TestMethods(unittest.TestCase):

    def test_rotate_moons1(self):
        x = np.array([[0.0,0.0],[0.0,0.0]])
        theta = math.pi/10
        x_f = my_fn.rotate_moons(theta, x)
        self.assertAlmostEqual(x_f.all(), x.all())
        
    def test_rotate_moons2(self):
        x = np.array([[1.0,1.0],[1.0,1.0]])
        theta = math.pi/10
        x_f = my_fn.rotate_moons(-theta, my_fn.rotate_moons(theta, x))
        self.assertAlmostEqual(x_f.all(), x.all())
        
    def test_rotate_moons3(self):
        x = np.array([[1.0,1.0],[1.0,1.0]])
        theta = math.pi/10
        theta1 = math.pi/12
        x = my_fn.rotate_moons(theta1, my_fn.rotate_moons(theta, x))
        x_f = my_fn.rotate_moons(theta, my_fn.rotate_moons(theta1, x))
        self.assertAlmostEqual(x_f.all(), x.all())
        
    def test_rotate_moons4(self):
        x = np.array([[1.0,1.0],[1.0,1.0]])
        x_f = my_fn.rotate_moons(2*math.pi, x)
        self.assertAlmostEqual(x_f.all(), x.all())
        
    def test_rotate_moons5(self):
        x = np.array([[1.0,1.0],[1.0,1.0]])
        theta = math.pi/10
        theta1 = math.pi/12
        x = my_fn.rotate_moons(theta1, my_fn.rotate_moons(theta, x))
        x_f = my_fn.rotate_moons(theta+theta1, x)
        self.assertAlmostEqual(x_f.all(), x.all())
        
    def test_rotate_moons6(self):
        x = np.array([[1.0,1.0,1.0],[1.0,1.0,1.0]])
        with self.assertRaises(TypeError):
            my_fn.rotate_moons(0.0, x)

    def test_translate_moons1(self):
        x = np.array([[0.0,0.0],[0.0,0.0]])
        x_f = my_fn.translate_moons(0.0, 0.0, x)
        self.assertAlmostEqual(x_f.all(), x.all())
        
    def test_translate_moons2(self):
        x = np.array([[0.0,0.0],[0.0,0.0]])
        x_f = my_fn.translate_moons(-1.0, -1.0, my_fn.translate_moons(1.0, 1.0, x))
        self.assertAlmostEqual(x_f.all(), x.all())
        
    def test_translate_moons3(self):
        x = np.array([[0.0,0.0],[0.0,0.0]])
        x_f = my_fn.translate_moons(2.0, 2.0, my_fn.translate_moons(1.0, 1.0, x))
        x = my_fn.translate_moons(3.0, 3.0, x)
        self.assertAlmostEqual(x_f.all(), x.all())
        
    def test_translate_moons4(self):
        x = np.array([[0.0,0.0],[0.0,0.0]])
        x_f = my_fn.translate_moons(2.0, 2.0, my_fn.translate_moons(1.0, 1.0, x))
        x = my_fn.translate_moons(1.0, 1.0, my_fn.translate_moons(2.0, 2.0, x))
        self.assertAlmostEqual(x_f.all(), x.all())
        
    def test_translate_moons5(self):
        x = np.array([[1.0,1.0,1.0],[1.0,1.0,1.0]])
        with self.assertRaises(TypeError):
            my_fn.translate_moons(1.0, 1.0, x)

    def test_dump_learning_curve1(self):
        file_to_test = path_parent+"/output/abc.dat"
        my_fn.dump_learning_curve("abc", 1, 1, 1)
        test_file = path_to_test+"/test0.dat"
        test_lines = open(test_file).read()
        lines = open(file_to_test).read()
        os.remove(file_to_test)
        self.assertMultiLineEqual(test_lines, lines, "not equal files")
        
    def test_dump_learning_curve2(self):
        file_to_test = path_parent+"/output/abc.dat"
        my_fn.dump_learning_curve("abc", 1, 1, 1)
        my_fn.dump_learning_curve("abc", 2, 2, 2)
        test_file = path_to_test+"/test1.dat"
        test_lines = open(test_file).read()
        lines = open(file_to_test).read()
        os.remove(file_to_test)
        self.assertMultiLineEqual(test_lines, lines, "not equal files")

    def test_build_dataset_fn(self):
        x, y = my_fn.build_dataset(2, 8, 0.1)
        self.assertEqual(len(x.shape), 2, 'wrong dimensions of points')
        self.assertEqual(len(y.shape), 1, 'wrong dimensions of labels')
        self.assertEqual(x.shape[0], 8, 'wrong number of points')
        self.assertEqual(y.shape[0], 8, 'wrong number of labels')
        for xx in x:
            self.assertEqual(len(xx), 2, 'wrong number of coordinates')
        for yy in y:
            self.assertTrue(yy==1 or yy==0, 'wrong label')
            
    def test_get_dataset_fn1(self):
        x, y = my_fn.build_dataset(2, 8, 0.1)
        for client_id in [0,1]:
            x_c, y_c = my_fn.get_client_dataset(client_id, 2, x, y)
            self.assertEqual(len(x_c.shape), 2, 'wrong dimensions of points')
            self.assertEqual(len(y_c.shape), 1, 'wrong dimensions of labels')
            self.assertEqual(x_c.shape[0], 4, 'wrong number of points')
            self.assertEqual(y_c.shape[0], 4, 'wrong number of labels')
            for xx in x_c:
                self.assertEqual(len(xx), 2, 'wrong number of coordinates')
            for yy in y_c:
                self.assertTrue(yy==1 or yy==0, 'wrong label')

    def test_get_dataset_fn2(self):
        x, y = my_fn.build_dataset(2, 8, 0.1)
        with self.assertRaises(TypeError):
            my_fn.get_client_dataset(-1, 2, x, y)
            
    def test_get_dataset_fn3(self):
        _, y = my_fn.build_dataset(2, 8, 0.1)
        with self.assertRaises(TypeError):
            my_fn.get_client_dataset(0, 2, np.array([]), y)
            
    def test_get_dataset_fn4(self):
        x, _ = my_fn.build_dataset(2, 8, 0.1)
        with self.assertRaises(TypeError):
            my_fn.get_client_dataset(0, 2, x, np.array([]))

    def test_plotters(self):
        x_train = np.array([[0.0,0.0],[2.0,2.0]])
        y_train = np.array([0, 1])
        x_test = np.array([[-1.0,-1.0],[1.0,1.0]])
        y_test = np.array([0, 1])
        images_path = path_parent+'/output/*.png'
        path_to_pass = path_parent+'/output'
        my_fn.plot_client_dataset(0, x_train, y_train, x_test, y_test, path_to_pass)
        files = glob.glob(images_path)
        compare_images(path_to_test+'/test_data_client_0.png', files[0], 1.0)
        os.remove(files[0])
        x_test = np.array([[-1.0,-1.0],[1.0,1.0]])
        y_test = np.array([0, 1])
        model = my_fn.create_keras_model()
        model.load_weights(path_to_test+"/model.h5")
        my_fn.plot_decision_boundary(model, x_test, y_test,
                                     client_id=0, fed_iter=1, path=path_to_pass)
        files =  glob.glob(images_path)
        compare_images(path_to_test+'/test_dec_bound_c0_e1.png', files[0], 1.0)
        os.remove(files[0])
        my_fn.plot_decision_boundary(model, x_test, y_test,
                                     client_id=0, path=path_to_pass)
        files =  glob.glob(images_path) 
        compare_images(path_to_test+'/test_dec_bound_c0.png', files[0], 1.0)
        os.remove(files[0])
        my_fn.plot_decision_boundary(model, x_test, y_test,
                                     path=path_to_pass)
        files =  glob.glob(images_path)
        compare_images(path_to_test+'/test_dec_bound_nofed.png', files[0], 1.0)
        os.remove(files[0])
        

if __name__ == '__main__':
    unittest.main()