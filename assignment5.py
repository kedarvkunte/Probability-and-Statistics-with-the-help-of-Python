import numpy as np
from scipy.special import erfinv
import unittest
import matplotlib.pyplot as plt
import seaborn as sns

###########################################
# Problem 1
###########################################


def inverse_normal_cdf(u,mu,sigma2):
    F_inv = mu + (np.sqrt(2*sigma2))*erfinv((2*u)-1)
            ## Add code here ##
    return np.array(F_inv)
    
###########################################
# Problem 2
###########################################

def normal_sampler(n_samples,mu,sigma2):
    ## Add code here ##
    CDF_Inverse_SAMPLE = []
    low = 0.0
    high = 1.0

    u_array = np.random.uniform(low,high,n_samples)

    #print(u_array)
    for i,u in np.ndenumerate(u_array):
        #print(u)
        CDF_Inverse_SAMPLE.append(inverse_normal_cdf(u,mu,sigma2))

    return np.array(CDF_Inverse_SAMPLE)
    
###########################################
# Problem 3
###########################################

def fun(x):
    return x**2

def mcev(fun,n_samples,mu,sigma2):
    ## Add code here ##
    sampler_array = normal_sampler(n_samples,mu,sigma2)
    funct_val = np.empty(0)
    for index, value in np.ndenumerate(sampler_array):
        funct_val = np.append(funct_val,fun(value))
        #print("fun value ",fun(value))
    mean_val = np.mean(funct_val)
    #print("SHAPE ",funct_val.shape)
    return mean_val

#u = np.random.rand()

Sample_of_1000 = normal_sampler(1000,0,2)
#print("Sample of 1000 \n",Sample_of_1000)

#print(Sample_of_1000)
plt.hist(Sample_of_1000)
plt.title('Histogram of the samples')
plt.show()

expect = mcev(fun,1000,0.0,2.0)
print("Expected Value of fun object = ",expect)
#print(fun(5))

