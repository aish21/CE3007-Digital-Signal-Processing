% design a low pass filter
close all;
clear all;

% Question: Q5

% Designing the FIR filter directly by inverse Fourier Transform!
N = 31;
w_c = 2*pi*2.5/10; %2*pi*f/Fs???;   

% how to calculate the normalized cut-off frequeny given the sampling rate at 10Khz and the stop-band at 2.5Khz?
% Fs = 10k, assume ideal cutoff 2.5kHz 
% omega_c = 2pi2.5/10=2pi/4
% normalize over pi yield 1/2 

n_range = -15:1:15;
midX = length(n_range)/2;
for (n=1:length(n_range))              %n count from 1 to 31
                                       %n_range(n) count from -15 to 15
  h(n) = sin(w_c*n_range(n))/(pi*n_range(n)); 
  %this is the sinc function For non-sero n  
end

h(1+((length(n_range)-1)/2)) = w_c/pi; 
%this is the h(0)
plot(h,'+-');

% Question: Q5(1)
h2 = h./sum(h); 
%WHY AH SUM(H) IS ONE CUZ SINC IS NORMALIZED  % h2 is a normalized version of h
fvtool(h2,1); 
% examine using magnitude response at y-axis (not dB or magnitude^2)?
% Passband and stopband specs are specified in linear scale.
% Although we can always do it in db and convert back
B = fir1(30,0.5, ones(31,1)','noscale')
freqz(B,1,512)
% Question: Q5 (2)
% Lets study the performance of the designed filter,
% does it satisfy criteria? passband ripple < 0.1, stopband attenuation <
% 0.01? 
fvtool(h,1);
%Passband ripple 
%We don't trust our eyes
meow = freqz(h,1,N); %This is the frequency response
cutoff_index = 15  %trial and error to get this to be the end of the ripple
ripple = max(abs(meow(1:cutoff_index)))-min(abs(meow(1:cutoff_index)))
% passband ripple is 0.1011 in this case
stopband_attn = max(abs(meow(18:length(meow))  %try and error to get the first stopband
% In this case is 0.06 so yea cannot

% Question: Q5 (3)
% Using Matab's function fir1 to design, fill in the ???
%B_ham = fir1(???,???, hamming(???),'noscale')
B_ham = fir1(N-1,w_c/pi, hamming(N),'noscale');
fvtool(B_ham,1);
