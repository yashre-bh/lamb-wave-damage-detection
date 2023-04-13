import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

# # Download the data
# kg.api.authenticate()
# kg.api.dataset_download_files('g2net-gravitational-wave-detection/train/0/0/0/00000e74ad.npy', path='./data', unzip=True)
dwnfiles = np.loadtxt('training_labels.csv', dtype=str, delimiter=',', usecols=(0))
dwnfiles = dwnfiles[:263]

for dwnfile in dwnfiles:
    print()
    api.competition_download_file('g2net-gravitational-wave-detection','train/0/0/'+dwnfile[2]+'/'+dwnfile+'.npy', path='./data')
