import pandas as pd
import numpy as np
#import pandas_datareader.data as web
import matplotlib
matplotlib.use('TkAgg') # Remove this to compare with MacOSX backend
import matplotlib.pyplot as plt
import datetime


from numpy import matrix, array, zeros, empty, sqrt, ones, dot, append, mean, cov, transpose, linspace
from numpy.linalg import inv, pinv
#from pylab import *
#from structures.quote import QuoteSeries
import scipy.optimize
import random

'''
##########################################################

formula used 

E(R) = [(τ C)-1 + PT ΩP]-1 [(τ C)-1 Π + PT ΩQ]

wherer E(R) is the expected excess return
τ is a scalar, I have used 1
C is the variance covariance matrix
P is matrix identifying which assets you have view on
Ω is uncertainty
Q is view on expected return

the view used is MSFT outperforming GE by 2%
AAPL underperforming JNJ by 2%
views = [
        ('MSFT', '>', 'GE', 0.02),
        ('AAPL', '<', 'JNJ', 0.02)
        ]
        
###########################################################

'''


def load_data_net():
        symbols = ['XOM', 'AAPL', 'MSFT', 'JNJ', 'GE', 'GOOG', 'CVX', 'PG', 'WFC']
        cap = {'^GSPC':14.90e12, 'XOM':403.02e9, 'AAPL':392.90e9, 'MSFT':283.60e9, 'JNJ':243.17e9, 'GE':236.79e9, 'GOOG':292.72e9, 'CVX':231.03e9, 'PG':214.99e9, 'WFC':218.79e9}
        n = len(symbols)
        prices_out = pd.read_csv('data_BL.csv',index_col=['date'],parse_dates=True)
        caps_out = []
        for s in symbols:
                caps_out.append(cap[s])
        return symbols, prices_out, caps_out


def assets_meanvar(names, prices, caps):
    weights = array(caps) / sum(caps)  # create weights

    rtns = prices.pct_change()
    expreturns = rtns.mean()*252 #annulised

    covars = rtns.cov()*252

    return names, weights, expreturns, covars


def print_assets(names, W,R,C):
        print("\nHistorical Returns and Market equilibrium Weights\n")
        df = pd.DataFrame(names)
        w = pd.DataFrame(W)
        r = pd.DataFrame(R)
        df = pd.concat([df, w], axis=1)
        df.columns = ['Stocks', 'Mkt eq Weights']
        df.set_index('Stocks', inplace=True)
        df = pd.concat([df, r], axis=1)
        df.columns = ['Mkt eq Weights', 'Hist Returns']
        df[['Mkt eq Weights', 'Hist Returns']] = df[['Mkt eq Weights', 'Hist Returns']].apply(lambda x: x * 100)
        df =df[['Hist Returns','Mkt eq Weights']]
        return df


def port_mean_var(W,R,C):
    port_ret = np.sum(R * W)
    port_vol = np.sqrt(np.dot(W.T, np.dot(C, W)))
    return port_ret,port_vol


# given the pairs of assets, prepare the views and link matrices. This function is created just for users' convenience
def prepare_views_and_link_matrix(names, views):
        r, c = len(views), len(names)
        Q = [views[i][3] for i in range(r)]     # view matrix
        P = zeros([r, c])                                       # link matrix
        nameToIndex = dict()
        for i, n in enumerate(names):
                nameToIndex[n] = i
        for i, v in enumerate(views):
                name1, name2 = views[i][0], views[i][2]
                P[i, nameToIndex[name1]] = +1 if views[i][1]=='>' else -1
                P[i, nameToIndex[name2]] = -1 if views[i][1]=='>' else +1
        return array(Q), P


# Load names, prices, capitalizations from the data source
names, prices, caps = load_data_net()


# Estimate assets's expected return and covariances
names, W, R, C = assets_meanvar(names, prices, caps)
rf = .015       # Risk-free rate
ann_excess_rtns = R - rf


# Print historic data
df = print_assets(names, W, R, C)
print(df)

# Calculate portfolio historical return and variance
mean, var = port_mean_var(W, R, C)


# Black-litterman reverse optimization
lmb = (mean - rf) / var                         # Calculate return/risk trade-off
Pi = 2*np.dot(np.dot(lmb, C), W)                  #implied equib excess return



# Determine views to the equilibrium returns and prepare views (Q) and link (P) matrices
views = [
        ('MSFT', '>', 'GE', 0.02),
        ('AAPL', '<', 'JNJ', 0.02)
        ]


Q, P = prepare_views_and_link_matrix(names, views)
print('\nViews Matrix')
print(Q)
print('\nLink Matrix')
print(P)
print('\n')

tau = 1 # scaling factor


# Calculate omega - uncertainty matrix about views
omega = np.dot(np.dot(np.dot(tau, P), C), transpose(P))


# Calculate equilibrium excess returns with views incorporated
A_ = inv(np.dot(tau, C))
B_ = np.dot(np.dot(transpose(P), inv(omega)), P)
C_ = np.dot(inv(np.dot(tau, C)), Pi)
D_ = np.dot(np.dot(transpose(P), inv(omega)), Q)
BL_rtns = np.dot(inv(A_ + B_), (C_ + D_))


Z = np.dot(inv(C),BL_rtns)
BL_implied_weights = Z/sum(Z)


def summarize(Pi,BL_rtns,BL_implied_weights,df):
    pass
    Pi = pd.DataFrame(Pi) ; BL_rtns = pd.DataFrame(BL_rtns) ; BL_weights = pd.DataFrame(BL_implied_weights)
    zz = pd.concat([Pi,BL_rtns,BL_weights],axis=1)
    zz['Stocks'] = names
    zz.set_index('Stocks',inplace=True)
    zz.columns = ['Market eq returns','BL implied returns','BL implied weights']
    zz[['Market eq returns','BL implied returns']] = zz[['Market eq returns','BL implied returns']].apply(lambda x: x*100)
    df = pd.concat([df,zz],axis=1)
    df['BL implied weights'] = df['BL implied weights'].apply(lambda x: x * 100)
    df = df.round(2)
    return(df)

df = summarize(Pi,BL_rtns,BL_implied_weights,df)
print(df)


portfolio_return = np.dot(df['BL implied weights'],df['BL implied returns'])