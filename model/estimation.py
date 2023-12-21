import os
import os.path
# Set the current file's directory as the default directory
os.chdir('QPGWR/model')

import QPGWR
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import sklearn
from mgwr.gwr import GWR,MGWR
from mgwr.sel_bw import Sel_BW

########################################################
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

Important_para=np.zeros([1440,7,500])
# read the plot-level data

rawdata=pd.read_csv('../data/raw_sim_data.csv')

from tqdm import tqdm
#loop for each file to record the estimated parameters of three models
for file_num in tqdm(range(1,501)):
    
    # data=rawdata[rawdata['sim']==(file_num)]
    data=rawdata[rawdata['sim']==(file_num)].copy()

    #MGWR functions
    y = data['yield'].values
    X = data[['N','N2']].values
    u = data['X']
    v = data['Y']
    coords = list(zip(u,v))
    y = y.reshape(-1,1)

    file_num=file_num-1
    #x0, a, y0
    xy=np.array(data[['X','Y']])
    gwr_results = GWR(coords, y, X, bw=50, fixed=False,kernel='gaussian').fit()
    
    Important_para[:,0,file_num]=gwr_results.params[:,1]
    Important_para[:,1,file_num]=gwr_results.params[:,2]

    popt0=[data['yield'].min(),(data['yield'].max()-data['yield'].min())/(data['N'].max()-data['N'].min()),data['N'].mean()]
    data_LP=QPGWR.LPGWR(data,50,xy,popt0,n_var='N',y_var='yield')
    
    Important_para[:,2,file_num]=data_LP['NK_']

    popt0=[data['N'].mean(),-(data['yield'].max()-data['yield'].min())/(data['N'].mean()-data['N'].min())**2,data['yield'].max()]
    data_QP=QPGWR.QPGWR(data,50,xy,popt0,n_var='N',y_var='yield')

    data_gpd=gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.X, data.Y))
    
    Important_para[:,3,file_num]=data_QP['b1_']
    Important_para[:,4,file_num]=data_QP['b2_']

    #also record the real eonr
    Important_para[:,5,file_num]=data['b1']
    Important_para[:,6,file_num]=data['b2']

np.save('../Results3th/Important_para2.npy',Important_para)