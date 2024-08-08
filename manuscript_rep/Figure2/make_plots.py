import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import ast 
from matplotlib.lines import Line2D


smaller_size = 13
medium_size = 14
bigger_size = 16

plt.rc('font', size=bigger_size)          # controls default text sizes
plt.rc('axes', titlesize=medium_size)     # fontsize of the axes title
plt.rc('axes', labelsize=medium_size)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=smaller_size)    # fontsize of the tick labels
plt.rc('ytick', labelsize=smaller_size)    # fontsize of the tick labels
plt.rc('legend', fontsize=smaller_size)    # legend fontsize
plt.rc('figure', titlesize=bigger_size)  # fontsize of the figure title

id = "Pytorch_RP_fig2_rep_5thattempt_18b9f662238ffa48b9457ee6d5b5f90f" 
index='000'

loss = pd.read_csv('record_eval.csv',usecols=range(8))
noi_index = np.genfromtxt("node_Index.csv", 
                          dtype = str)[[2,3,82,31,40,83]]



def moving_average(sequence, n=5) :
    ret = np.cumsum(sequence, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


# Loss Plot
idx = np.where([x!='None' for x in loss['train_mse']])[0]
nma = 10
plt.plot(np.arange(len(idx)-nma+1), 
         moving_average(np.array([float(x) for x in loss['train_mse'][idx]]),n=nma), 
         alpha = 0.8, color="C2")
plt.plot(np.arange(len(idx)-nma+1), 
         moving_average(np.array([float(x) for x in loss['valid_mse'][idx]]),n=nma), 
         alpha = 0.8, color="C1")

plt.xlabel('Training iterations')
plt.ylabel('Mean Squared Error')

custom_lines = [Line2D([0], [0], color='C2'), 
                Line2D([0], [0], color='C1')]
plt.legend(custom_lines, ['Training', 'Validation'], loc='upper right',
                    frameon=False)
#plt.legend(['Training', 'Validation', 'Test'], frameon=False)
plt.xticks([0,1000,2000,3000,4000])
plt.yticks([0,0.2,0.4,0.6,0.8,1.0])
plt.title("Prediction error decreases during training", 
          weight='bold', size=15)
plt.text(-0.13,1.02,'A', weight='bold')

plt.savefig("MSE_loss.png")
plt.close()


idx = np.where([x!='None' for x in loss['train_loss']])[0]
plt.plot(np.arange(len(idx)-nma+1), 
         moving_average(np.array([float(x) for x in loss['train_loss'][idx]]),n=nma), 
         alpha = 0.8, color="C2")
plt.plot(np.arange(len(idx)-nma+1), 
         moving_average(np.array([float(x) for x in loss['valid_loss'][idx]]),n=nma), 
         alpha = 0.8, color="C1")

plt.xlabel('Training iterations')
plt.ylabel('Mean Squared Error')

custom_lines = [Line2D([0], [0], color='C2'), 
                Line2D([0], [0], color='C1')]
plt.legend(custom_lines, ['Training', 'Validation'], loc='upper right',
                    frameon=False)
plt.add_artist(legend)
#plt.legend(['Training', 'Validation', 'Test'], frameon=False)
plt.xticks([0,1000,2000,3000,4000])
plt.yticks([0,0.2,0.4,0.6,0.8,1.0])
plt.title("Prediction error decreases during training", 
          weight='bold', size=15)
plt.text(-0.13,1.02,'A', weight='bold')

plt.savefig("MSE_loss.png")
plt.close()



