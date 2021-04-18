#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 1 17:54:03 2021

@author: relogu
"""
#%% importations
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
import glob
import sys
import math
sys.path.append('../')
import flower_make_moons_test.common_fn as my_fn
#%%
(x, y) = datasets.make_moons(n_samples=840, shuffle=True, noise=0.1, random_state=42)
x_tr = my_fn.translate_moons(0.5,0.5, x.copy())
x_rot = my_fn.rotate_moons(math.pi/5, x_tr.copy())
y = y*0+20
y_tr = y.copy()*0+50
y_rot = y.copy()*0+100
x = np.concatenate((x, x_tr), axis=0)
y = np.concatenate((y, y_tr), axis=0)
x = np.concatenate((x, x_rot), axis=0)
y = np.concatenate((y, y_rot), axis=0)

cm = plt.cm.get_cmap('RdYlBu')
sns.set_style('whitegrid')
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(18,9))
ax.set_xlabel('x')
ax.set_ylabel('Y')
# Plot the samples
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cm, marker=".")
plt.draw()
plt.savefig('../docs/images/datasets_examples.png')
plt.show()
plt.close()

(x, y) = datasets.make_moons(n_samples=320, shuffle=True, random_state=42)

sns.set_style('whitegrid')
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(18,9))
ax.set_xlabel('x')
ax.set_ylabel('Y')
# Plot the samples
plt.scatter(x[:, 0], x[:, 1], marker=".")
plt.draw()
plt.savefig('../docs/images/make_moons_example.png')
plt.show()
plt.close()



# %%
