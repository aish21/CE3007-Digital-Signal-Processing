import numpy as np
import matplotlib.pyplot as plt

def myDTFS(x):
    X = np.zeros(len(x), dtype=complex)
    Omega = np.zeros(len(x))
    N = len(x)
    for k in np.arange(0,len(x)):
        tmpVal = 0.0
        Omega[k] = (2*np.pi/N)*k
        for n in np.arange(0,len(x)):
            tmpVal = tmpVal + x[n]*np.exp(-1j*(2*np.pi/N)*k*n)
        X[k] = tmpVal/N
    return (X,Omega)



def myIDTFS(X):
    x = np.zeros(len(X), dtype=float)
    N = len(x)
    for n in np.arange(0,len(x)):
        tmpVal = 0.0
        for k in np.arange(0,len(X)):
            tmpVal = tmpVal + X[k]*np.exp(+1j*(2*np.pi/N)*k*n)
        x[n] = np.absolute(tmpVal)
    return (x)


x = [0,0,0, 1,1, 0, 0, 0]

(X1, Omega) = myDTFS(x)
(X) = np.fft.fft(x)
print(X)
N=len(x)
print(X1*N)

# Lets save the file, fname, sequence, and samplingrate needed
absX = np.absolute(X)
angleX = np.angle(X)
titleStr = 'x[n]'


f, axarr = plt.subplots(2, sharex=True)
axarr[0].stem(np.arange(0,8), absX)
axarr[0].set_title('DTFS '+titleStr)
axarr[0].set_ylabel('mag value')


axarr[1].stem(np.arange(0,8), angleX)
axarr[1].set_xlabel('k')
axarr[1].set_ylabel('Phase (rad)')
plt.show()


f, axarr = plt.subplots(2, sharex=True)
axarr[0].stem(Omega/np.pi, absX*(2*np.pi),'C0x')
axarr[0].set_title('DTFT of '+titleStr)
axarr[0].set_ylabel('mag value')
plt.xticks(np.arange(0, 2, step=0.25))

axarr[1].stem(Omega/np.pi, angleX)
axarr[1].set_xlabel('omega*pi (rad/sample)')
axarr[1].set_ylabel('Phase (rad)')
plt.show()
