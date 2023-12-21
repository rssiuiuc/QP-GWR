import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd

from scipy.optimize import curve_fit
from scipy.stats import norm # only for generic data with errors

def distance(xy,x0,y0):
    x=xy[:,0]
    y=xy[:,1]
    return np.sqrt((x-x0)**2+(y-y0)**2)

def dis2wi(dis_h,w_h,num=20):
    """
    Calculates the spatial weights for each observation based on their distance to the num-th nearest neighbor.
    
    Parameters:
    dis_h (numpy.ndarray): A 2D array of shape (n, n) where n is the number of observations. Each element (i, j) represents the Euclidean distance between observation i and observation j.
    w_h (numpy.ndarray): A 2D array of shape (n, n) where n is the number of observations. Each element (i, j) represents the spatial weight between observation i and observation j.
    num (int): The number of nearest neighbors to consider when calculating the spatial weights. Default is 20.
    
    Returns:
    None
    """
    for count in range(dis_h.shape[0]): 
        d=dis_h[count,:]
        
        index=np.argsort(d)
        h=d[index[num-1]]

        wi=np.exp(-0.5*(d/h)**2)
        w_h[count,:]=wi/sum(wi)
    
def quad_plateau(x, b0, b1, Nk):
    """
    Computes the quadratic plateau function for a given input x.

    Parameters:
    x (float): Input value for the function.
    b0 (float): Intercept parameter.
    b1 (float): Slope parameter.
    Nk (float): Inflection point parameter.

    Returns:
    float: The value of the quadratic plateau function for the given input x.
    """
    return b0+b1*(x-Nk)*(x<Nk)

def QPGWR(data,band_num,xy,popt0,n_var='n_rate',y_var='yild_vl'):
    """
    This function performs a quadratic plateau geographically weighted regression (QPGWR) on the given data.

    Parameters:
    data (pandas.DataFrame): The input data containing the variables to be used in the regression.
    band_num (int): The number of bands to be used in the distance matrix.
    xy (numpy.ndarray): The coordinates of the data points.
    popt0 (numpy.ndarray): The initial guess for the parameters of the regression.
    n_var (str): The name of the column in the input data containing the independent variable.
    y_var (str): The name of the column in the input data containing the dependent variable.

    Returns:
    pandas.DataFrame: The input data with additional columns for the regression parameters.
    """
    # distance matrix (point*point)
    dis_h=np.zeros([data.shape[0],data.shape[0]])
    
    for count in range(data.shape[0]):
        x0=xy[count,0]
        y0=xy[count,1]
        dis_h[:,count]=distance(xy,x0,y0)

    #weight matrix (point*point)
    w_h=np.zeros([data.shape[0],data.shape[0]])
    dis2wi(dis_h,w_h,band_num)

    # xdata and ydata with errors
    xl = np.array(data[n_var]) # xdata
    yn = np.array(data[y_var]) # ydata with errors

    #initial guess for the parameters
    x0_r=np.zeros([data.shape[0],])
    a_r=np.zeros([data.shape[0],])
    y0_r=np.zeros([data.shape[0],])
    y2=np.zeros([data.shape[0],])

    #loop over all points
    for i in range(xl.shape[0]):

        theguess = popt0
        w_i=w_h[i,:]

        yn_i=yn*np.sqrt(w_i)
        # yn_i=yn_i[T_index]

        def quad_plateau2(x, x0, a, y0): # much shorter version in this representation
            # return (y0 + a * ( x - x0 )**2 * (x < x0 ))*np.sqrt(w_i[T_index])
            return (y0 + a * ( x - x0 )**2 * (x < x0 ))*np.sqrt(w_i)

        popt, pcov = curve_fit(
            quad_plateau2,
            xl, yn_i,
            bounds=((data[n_var].min(), -np.inf, 0), (data[n_var].max(), 0, np.inf)),
            p0=theguess,maxfev=10000)

        x0_r[i]=popt[0]
        a_r[i]=popt[1]
        y0_r[i]=popt[2]

        # the predicted values
        y2[i]=y0_r[i] + a_r[i] * ( xl[i] - x0_r[i] )**2 * (xl[i] < x0_r[i] )

 
    data.loc[:,'a_']=a_r
    data.loc[:,'x0_']=x0_r
    data.loc[:,'y0_']=y0_r

    data.loc[:,'b0_'] = data['y0_']+ data['a_']*data['x0_']**2
    data.loc[:,'b1_'] = -2*data['a_']*data['x0_']
    data.loc[:,'b2_'] = data['a_']
    
    return data

def LPGWR(data,band_num,xy,popt0,n_var='n_rate',y_var='yild_vl'):
    """
    Perform a local polynomial geographically weighted regression (LPGWR) on the given data.

    Args:
    - data: pandas DataFrame containing the data to be used in the regression
    - band_num: integer representing the number of bands to be used in the distance-to-weight conversion
    - xy: numpy array containing the x and y coordinates of each data point
    - popt0: numpy array containing the initial parameter values for the regression
    - n_var (str): The name of the column in the input data containing the independent variable.
    - y_var (str): The name of the column in the input data containing the dependent variable.

    Returns:
    - data: pandas DataFrame containing the original data with additional columns for the regression coefficients
    """
    #important distance matrix: point*dist
    dis_h=np.zeros([data.shape[0],data.shape[0]])
    
    for count in range(data.shape[0]):
        x0=xy[count,0]
        y0=xy[count,1]
        # print(x0)
        dis_h[:,count]=distance(xy,x0,y0)

    #important weight matrix: point*dist
    w_h=np.zeros([data.shape[0],data.shape[0]])
    dis2wi(dis_h,w_h,band_num)

    xl = np.array(data[n_var]) # xdata
    yn = np.array(data[y_var]) # ydata with errors

    b0_=np.zeros([data.shape[0],])
    b1_=np.zeros([data.shape[0],])
    Nk_=np.zeros([data.shape[0],])
    y2=np.zeros([data.shape[0],])

    for i in range(xl.shape[0]):

        theguess = popt0
        w_i=w_h[i,:]

        yn_i=yn*np.sqrt(w_i)

        def linear_plateau(x, b0, b1, Nk): # much shorter version in this representation
            return (b0+b1*(x-Nk)*(x<Nk))*np.sqrt(w_i)

        popt, pcov = curve_fit(
            linear_plateau,
            xl, yn_i,
            bounds=((0, 0, data[n_var].min()), (np.inf, np.inf, data[n_var].max())),
            p0=popt0,maxfev=10000)

        b0_[i]=popt[0]
        b1_[i]=popt[1]
        Nk_[i]=popt[2]
        
    data.loc[:,'b0_']=b0_
    data.loc[:,'b1_']=b1_
    data.loc[:,'NK_']=Nk_
    
    return data