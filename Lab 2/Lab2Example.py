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
import scipy
import winsound
from scipy import signal



# The input is a float array (should have dynamic value from -1.00 to +1.00
def fnNormalizeFloatTo16Bit(yFloat):
    y_16bit = [int(s*32767) for s in yFloat]
    return(np.array(y_16bit, dtype='int16'))

# The input is a float array (should have dynamic value from -1.00 to +1.00
def fnNormalize16BitToFloat(y_16bit):
    yFloat = [float(s/32767.0) for s in y_16bit]
    return(np.array(yFloat, dtype='float'))


print("Example of Lab2 CE3007")
ipcleanfilename = 'helloworld_16bit.wav'

print("Example using np concolve")
# note that when we process, we WILL use floating point array
# hence, dtype must be include in definition of np.array
h = np.array([1,2,3],dtype='float')
x = np.array([1,0,0,0,0,2,0,0,0,0,0],dtype='float')
y = np.convolve(x,h)
y2 = scipy.signal.lfilter(h,[1],x)
plt.figure()
plt.stem(y,  linefmt='r-o')
plt.stem(y2, linefmt='b-x')
plt.ylabel('op using convolve and scipy lfilter')
plt.xlabel('sample n')
plt.show()


#
print("Example of Lab2, Q5, CE3007")
# The first part - build the boisy signal, embed 3KHz to helloWorld
winsound.PlaySound(ipcleanfilename, winsound.SND_FILENAME)
[Fs, sampleX_16bit] = wavfile.read(ipcleanfilename)

# we wish to do everything in floating point, hence lets convert to floating point first.
# by the following normalization
sampleX_float = fnNormalize16BitToFloat(sampleX_16bit)

#e.g, lets make the entire signal a bit larger, scale by 3.0
sampleX_float = np.multiply(3.0,sampleX_float)
# remember to normalize it to +-1 dynamic range floating point before processing it.
# lets plot the noisy signal in time and spectrogram
plt.figure()
plt.plot(sampleX_float,'r')
plt.ylabel('signal (float)')
plt.xlabel('sample n')
plt.show()

# reminder - dont processed the 16bit sequence, always convert to float before doing the DSP
[f, t, Sxx_clean] = signal.spectrogram(sampleX_float, Fs, window=('blackmanharris'),nperseg=512,noverlap=int(0.9*512))
plt.pcolormesh(t, f, 10*np.log10(Sxx_clean))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('spectrogram of signal')
plt.show()


# generate the impulse response using scipy.signal.lfilter module
# where num and den are the coefficients of the linear constant coefficients
numSamples = 100
n = np.arange(0,numSamples,1)
impulseH = np.zeros(numSamples)
impulseH[0] = 1
num = [1, -0.7653668, 0.99999]
den = [1, -0.722744, 0.888622]
y = signal.lfilter(num, den, impulseH)
plt.stem(n, y)
plt.ylabel('impulse response of the IIR filter')
plt.xlabel('sample n')
plt.show()

