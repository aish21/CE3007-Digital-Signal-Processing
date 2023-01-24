# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 13:10:08 2017

@author: Chng Eng Siong
Date: Dec 2017, using python to solve DSP problems
purpose - to remove dependency on Matlab
"""

"""
  My env: use anaconda, install matplotlib via 
  conda install matplotlib
  conda install scipy
  conda install winsound
  
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile  as wavfile
import winsound
# plotting 3D complex plane
from mpl_toolkits.mplot3d import Axes3D


# This is an example of using python to generate a discrete time sequence
# with time, value pair of a sinusoid


# The following function generates a continuous time sinusoid
# given the amplitude A, F (cycles/seconds), Fs=sampling rate, start and endtime
def fnGenSampledSinusoid(A,Freq,Phi, Fs,sTime,eTime):
    # Showing off how to use numerical python library to create arange
    n = np.arange(sTime,eTime,1.0/Fs)
    y = A*np.cos(2 * np.pi * Freq * n + Phi)
    return [n,y]


# The input is a float array (should have dynamic value from -1.00 to +1.00
def fnNormalizeFloatTo16Bit(yFloat):
    y_16bit = [int(s*32767) for s in yFloat]
    return(np.array(y_16bit, dtype='int16'))

# The input is a float array (should have dynamic value from -1.00 to +1.00
def fnNormalize16BitToFloat(y_16bit):
    yFloat = [float(s/32767.0) for s in y_16bit]
    return(np.array(yFloat, dtype='float'))


def fn_mostBasicCosineSignal():
    # Lets start with the most basic
    #y(theta) = cos(theta)
    theta = np.arange(0,4*np.pi,np.pi/10.0)
    y = 2.5*np.cos(theta)
    plt.figure(1)
    plt.plot(theta/(np.pi), y,'r--o');
    plt.xlabel('theta (pi)'); plt.ylabel('y(theta)')
    plt.title('sinusoid of signal (floating point)')
    plt.grid()
    plt.show()
    print('Above figure 0 shows cosine wrt to horizontal axis of angle theta')

    t = np.arange(0,2.0,0.1)
    F = 1.0   #  1 cycle per second
    y = 1*np.cos(2*np.pi*F*t)
    plt.figure(2)
    plt.stem(t, y,'g-');
#    plt.plot(t, y,'ro');
    plt.xlabel('time in seconds'); plt.ylabel('y(t)')
    plt.title('sinusoid of signal (floating point)')
    plt.grid()
    plt.show()

    numPts = 30
    n = np.arange(0,numPts)
    Fs = 20.0   # sampling freq , ex= 10 times in 1 sec
    Ts = 1/Fs   # sampling period = 1/sampling frequency
    nT = n*Ts   #(1.0/10)
    F = 1.0   #  1 cycle per second
    yNT = 1*np.cos(2*np.pi*F*nT)
    plt.figure(3)
    plt.plot(n, yNT,'ro');
    plt.stem(n, yNT,'g-');
    plt.xlabel('samples'); plt.ylabel('y(nT)')
    plt.title('sinusoid of signal (floating point)')
    plt.grid()
    plt.show()




def fn_genCosineSignalwrtTime():
    A=0.5; F=6000; Phi = 0; Fs=16000; sTime=0; eTime = 1.4;
    [n,yfloat] = fnGenSampledSinusoid(A, F, Phi, Fs, sTime, eTime)
    # Lets plot what we have
    plt.figure(1)
    numSamples =72
    plt.plot(n[0:numSamples], yfloat[0:numSamples],'r--o');
    plt.xlabel('time in sec'); plt.ylabel('y[nT]')
    plt.title('sinusoid of signal (floating point)')
    plt.grid()
    plt.show()
    print('Above figure 1 shows sinusoid')

    plt.figure(2)
    nIdx = np.arange(0,numSamples)
    plt.plot(nIdx, yfloat[0:numSamples],'r--o');
    plt.xlabel('sample index n'); plt.ylabel('y[n]')
    plt.title('sinusoid of signal (floating point)')
    plt.grid()
    plt.show()
    print('Above figure 1 shows sinusoid')
    
    # Although we created the signal in the date type float and dynamic range -1.0:1.0
    # when we save it and when we wish to listen to it using winsound it should be in 16 bit int.
    y_16bit = fnNormalizeFloatTo16Bit(yfloat)
    
    # Lets save the file, fname, sequence, and samplingrate needed
    wavfile.write('t1_16bit.wav', Fs, y_16bit)
    wavfile.write('t1_float.wav', Fs, yfloat)   # wavfile write can save it as a 32 bit format float
    
    # Lets play the wavefile using winsound given the wavefile saved above
    #unfortunately winsound ONLY likes u16 bit values
    #thats why we had to normalize y->y_norm (16 bits) integers to play using winsounds
    winsound.PlaySound('t1_16bit.wav', winsound.SND_FILENAME)
    
    # The following at float fail to play using winsound!
    #winsound.PlaySound('t1_float.wav', winsound.SND_FILENAME)
    #The file t1_float.wav cannot be played by winsound - but can be played by audacity?




#  Second Example, plotting complex exponential!
#
# Lets generate and plot complex exponential in 2-d, polar and 3d plot
# how to include phase shift? Phi != 0?
def fn_genComplexExpSignal():

    print('Below is figure 2 shows complex exponential')
    numSamples = 200
    A=0.99; w1=2*np.pi/50.0;
    n = np.arange(0, numSamples, 1)
    y1 = np.multiply(np.power(A, n), np.exp(1j * w1 * n))
    
    
    # plotting in 2-D, the real and imag in the same figure
    plt.figure(1)
    plt.plot(n, y1[0:numSamples].real,'r--o')
    plt.plot(n, y1[0:numSamples].imag,'g--o')
    plt.xlabel('sample index n'); plt.ylabel('y[n]')
    plt.title('Complex exponential (red=real) (green=imag)')
    plt.grid()
    plt.show()
    
    # plotting in polar, understand what the spokes are
    plt.figure(2)
    for x in y1:
        plt.polar([0,np.angle(x)],[0,np.abs(x)],marker='o')
    
    plt.title('Polar plot showing phasors at n=0..N')
    plt.show()
    
    # plotting 3D complex plane
    plt.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    reVal = y1[0:numSamples].real
    imgVal = y1[0:numSamples].imag
    ax.plot(n,reVal, imgVal,  label='complex exponential phasor')
    ax.scatter(n,reVal,imgVal, c='r', marker='o')
    ax.set_xlabel('sample n')
    ax.set_ylabel('real')
    ax.set_zlabel('imag')
    ax.legend()
    plt.show()
    

#============================================================
# Main prog starts here
#========================================================

    

print("Example of Lab1 CE3007")
#fn_mostBasicCosineSignal()
#fn_genCosineSignalwrtTime()
fn_genComplexExpSignal()
#print("end of prog")
