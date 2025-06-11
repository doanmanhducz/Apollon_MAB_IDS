from __future__ import print_function, division

import matplotlib as mpl
mpl.rcParams["axes.linewidth"] = 0.05  # set the value globally

# ------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %matplotlib inline
plt.style.use("ggplot")
import xgboost as xgb
import pickle
import gc
import os
import sys
import sklearn.cluster as cluster

global sess
global graph

from matplotlib import pyplot
import re
import tensorflow as tf

from tensorflow.keras import applications
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import time
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB

# Import `Sequential` from `tensorflow.keras.models`
from tensorflow.keras.models import Sequential

# Import `Dense` from `tensorflow.keras.layers`
from tensorflow.keras.layers import Dense

# Load libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from tabulate import tabulate

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import (
    Input,
    Dense,
    Reshape,
    Flatten,
    Dropout,
    multiply,
    Multiply,
)
from tensorflow.keras.layers import (
    BatchNormalization,
    Activation,
    Embedding,
    ZeroPadding2D,
)
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.optimizers import Adam
from scipy import stats

gc.collect()
import timeit
from xgboost import XGBClassifier

TEST_XGB = 1
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import recall_score, precision_score, roc_auc_score, f1_score

# Fixed pylab import - use specific imports instead
import numpy as np
import matplotlib.pyplot as plt

# Fix TensorFlow v1 compatibility - use v2 equivalent
import seaborn as sns

import datetime
from IPython.display import display
from sklearn.metrics import hamming_loss
from tensorflow.keras.optimizers import RMSprop
import tensorflow.keras.backend as K

from gym import Env
from gym.spaces import Discrete, Box
import random

from scipy.sparse import data
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import hashlib

np.random.seed(123)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Activation,
    Flatten,
    ELU,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

# Fixed RL imports - use tf-agents or stable-baselines3 instead of deprecated keras-rl
try:
    from rl.agents.dqn import DQNAgent
    from rl.agents.sarsa import SarsaAgent
    from rl.policy import BoltzmannQPolicy
    from rl.memory import SequentialMemory
except ImportError:
    print("Warning: keras-rl not available. Consider using stable-baselines3 or tf-agents instead.")
    # Alternative: from stable_baselines3 import DQN

TEST_MODEL = 0
SHOW_TIME = 1
ALL_CLASSIFIERS = 0
DEBUG = 1
SHOW_PLOTS = 1
USE_UNIFORM_NOISE = 0

ESTIMATE_CLASSIFIERS = True
POST_TRAINING_ESTIMATE_CLASSIFIERS = False

USE_RL_AT_THE_END = False
USE_RL_AFTER_EPOCH = False
USE_RL_AFTER_BATCH = True

NOISE_SIZE = 100
ROUNDS_FACTOR = 1
PLOT_AFTER_EPOCH = 1
ADVERSARIAL_TRAINING = False
DEBUG_RL = 0

# ===============================================================================================================================
no_aug_accu_list = []
real_aug_accu_list = []
SMOTE_IPF_aug_accu_list = []
ProWSyn_aug_accu_list = []
polynom_fit_SMOTE_aug_accu_list = []
uGAN_aug_accu_list = []
GAN_aug_accu_list = []

no_aug_rcl_list = []
real_aug_rcl_list = []
SMOTE_IPF_aug_rcl_list = []
ProWSyn_aug_rcl_list = []
polynom_fit_SMOTE_aug_rcl_list = []
uGAN_aug_rcl_list = []
GAN_aug_rcl_list = []

no_aug_prec_list = []
real_aug_prec_list = []
SMOTE_IPF_aug_prec_list = []
ProWSyn_aug_prec_list = []
polynom_fit_SMOTE_aug_prec_list = []
uGAN_aug_prec_list = []
GAN_aug_prec_list = []

no_aug_f1_list = []
real_aug_f1_list = []
SMOTE_IPF_aug_f1_list = []
ProWSyn_aug_f1_list = []
polynom_fit_SMOTE_aug_f1_list = []
uGAN_aug_f1_list = []
GAN_aug_f1_list = []

import preprocess
import importlib

importlib.reload(preprocess)
from preprocess import *

import classifiers
importlib.reload(classifiers)
from classifiers import *

import plot_data
importlib.reload(plot_data)
from plot_data import *

import relevagan
importlib.reload(relevagan)
from relevagan import *

import manipulate
importlib.reload(manipulate)
from manipulate import *

from sklearn.impute import SimpleImputer

# Fixed seaborn style - use updated style name
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except:
    plt.style.use("seaborn-whitegrid")

# Fixed RL import
try:
    from rl.agents import DQNAgent
except ImportError:
    print("Warning: keras-rl not available")

def save_losses(
    list_log_iteration=[],
    xgb_acc=[],
    dt_acc=[],
    nb_acc=[],
    knn_acc=[],
    rf_acc=[],
    lr_acc=[],
    xgb_rcl=[],
    dt_rcl=[],
    nb_rcl=[],
    rf_rcl=[],
    lr_rcl=[],
    knn_rcl=[],
    best_xgb_acc_index=[],
    best_xgb_rcl_index=[],
    best_dt_acc_index=[],
    best_dt_rcl_index=[],
    best_nb_acc_index=[],
    best_nb_rcl_index=[],
    best_rf_acc_index=[],
    best_rf_rcl_index=[],
    best_lr_acc_index=[],
    best_lr_rcl_index=[],
    best_knn_acc_index=[],
    best_knn_rcl_index=[],
    epoch_list_disc_loss_real=[],
    epoch_list_disc_loss_generated=[],
    epoch_list_comb_loss=[],
    GAN_type="",
):
    # Changed 'dict' to 'data_dict' to avoid overriding built-in
    data_dict = {
        "Epoch": list_log_iteration,
        "xgb_acc": xgb_acc,
        "dt_acc": dt_acc,
        "nb_acc": nb_acc,
        "rf_acc": rf_acc,
        "lr_acc": lr_acc,
        "knn_acc": knn_acc,
        "xgb_rcl": xgb_rcl,
        "dt_rcl": dt_rcl,
        "nb_rcl": nb_rcl,
        "rf_rcl": rf_rcl,
        "lr_rcl": lr_rcl,
        "knn_rcl": knn_rcl,
        "best_xgb_acc_index": best_xgb_acc_index,
        "best_xgb_rcl_index": best_xgb_rcl_index,
        "best_dt_acc_index": best_dt_acc_index,
        "best_dt_rcl_index": best_dt_rcl_index,
        "best_nb_acc_index": best_nb_acc_index,
        "best_nb_rcl_index": best_nb_rcl_index,
        "best_rf_acc_index": best_rf_acc_index,
        "best_rf_rcl_index": best_rf_rcl_index,
        "best_lr_acc_index": best_lr_acc_index,
        "best_lr_rcl_index": best_lr_rcl_index,
        "best_knn_acc_index": best_knn_acc_index,
        "best_knn_rcl_index": best_knn_rcl_index,
        "dlr": epoch_list_disc_loss_real,
        "dlg": epoch_list_disc_loss_generated,
        "comb_loss": epoch_list_comb_loss,
    }

    df = pd.DataFrame(data_dict)
    df.to_csv(GAN_type + "losses.csv")
    print("Losses file saved")