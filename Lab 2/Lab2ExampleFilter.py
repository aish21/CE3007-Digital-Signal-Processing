# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 13:10:08 2017

@author: Chng Eng Siong
Date: Dec 2017, using python to solve DSP problems
purpose - to remove dependency on Matlab
"""

# this is lab 2 example, in which we will load a waveform,
# listen to it
# plot its time domain sequence,
# plot its spectrogram

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile  as wavfile
import winsound
from scipy import signal





# this is FIR filter assuming A=[1]
def myfilter_Bonly(B,X):

    numNum = len(B)
    memX = np.zeros(numNum)
    B_np  = np.array(B)

    y = np.zeros(len(X))
    for i in np.arange(len(X)):
        # doing the left side of DF1 structure
        memX[0] = X[i]   #rolling in the memory X input
        vec_left_op = np.multiply(memX,B_np)
        y[i] = np.sum(vec_left_op)
        memX = np.roll(memX,1) # getting ready for the next step
        memX[0] = 0  # we use roll, so circular shift, lets 0 shifted in element 0

    return y


print("Example of Lab2 for filter operation CE3007")


# generate the impulse response using scipy.signal.lfilter module
# where num and den are the coefficients of the linear constant coefficients
numSamples = 50
n = np.arange(0,numSamples,1)
impulseX = np.zeros(numSamples)
impulseX[0] = 1


num = [3.1, 2.1, -1.0]
den = [1]
my = myfilter_Bonly(num,impulseX)
# the above filter only has feedforward (num) coeff B,
# the feedback (den) coeff A = [1].

# you will need to implement this mylfilter
# which also has feedback coeff (den), WITH condition that den[0] = 1
#num = [1, -0.7653668, 0.99999]
#den = [1, -0.722744, 0.888622]
#my = mylfilter(num,den,impulseH)

y = signal.lfilter(num, den, impulseX)

plt.stem(n, y,'r')
plt.plot(my,'g--')
plt.show()


#sanity check that my implemetation is the same as signal.lfilter
# lets check of the two sequences are the same
if np.array_equal(np.around(y,decimals=10),np.around(my,decimals=10)):
    print('PASS = my op sequence same as signal.lfilter')
else:
    print('ERROR = my op sequence differ as signal.lfilter')
    for i in np.arange(len(my)):
        if y[i] != my[i]:
            print(i,y[i],my[i])