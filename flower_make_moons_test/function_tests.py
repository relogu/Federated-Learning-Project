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
import numpy as np
sys.path.append('../')
import flower_make_moons_test.common_fn as my_fn

class TestMethods(unittest.TestCase):

    def test_rotate_moons(self):
        x = np.array([[0.0,0.0],[0.0,0.0]])
        theta = math.pi/10
        x_f = my_fn.rotate_moons(theta, x)
        self.assertAlmostEqual(x_f.all(), x.all())
        x = np.array([[1.0,1.0],[1.0,1.0]])
        x_f = my_fn.rotate_moons(-theta, my_fn.rotate_moons(theta, x))
        self.assertAlmostEqual(x_f.all(), x.all())

    def test_translate_moons(self):
        x = np.array([[0.0,0.0],[0.0,0.0]])
        dx = 0.0
        dy = 0.0
        x_f = my_fn.translate_moons(dx, dy, x)
        self.assertAlmostEqual(x_f.all(), x.all())
        dx = 1.0
        dy = 1.0
        x_f = my_fn.translate_moons(-dx, -dy, my_fn.translate_moons(dx, dy, x))
        self.assertAlmostEqual(x_f.all(), x.all())

    def test_dump_learning_curve(self):
        path_to_file = pathlib.Path(__file__).parent.absolute()
        filename = "/output/abc.dat"
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
        os.remove(str(path_to_file)+filename)
        self.assertMultiLineEqual( test_file, file_to_test, "not equal files")

if __name__ == '__main__':
    unittest.main()