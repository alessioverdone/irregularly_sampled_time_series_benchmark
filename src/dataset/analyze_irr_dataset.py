import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read dataset
dataset = 'etth1'
dataset_param = 'seq96_pred96_mcar_sp0.300_seed0'
db = np.load(f'../../data/irregular/{dataset}/{dataset_param}/dataset.npz', allow_pickle=True)
db_config = json.load(open(f'../../data/irregular/{dataset}/{dataset_param}/config.json'))
print('Data loaded!')