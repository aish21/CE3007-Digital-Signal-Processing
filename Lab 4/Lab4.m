% Lab 4:
% generate a signal with sampling rate of 1KHz and also 100Hz.
% we then wish to reconstruct the signal sampled at 1KHz from 100Hz.
% signal reconstruction, upsampling
% The idea is to send a interpolated signal through a low pass filter!
% answer the following questions

close all; clear all;

Duration = 1;

F1000 = 1000;
T1000 = 1/F1000;
n=1:F1000*Duration;
for n=1:F1000*Duration
  y1000(n) = 2*sin(2*pi*2*(n-1)*T1000+(pi/4));  
  t1000(n) = (n-1)*T1000;
end

F100 = 100; T100 = 1/F100;
for (n=1:F100*Duration)
  y100(n) = 2*sin(2*pi*2*(n-1)*T100+(pi/4));  
  t100(n) = (n-1)*T100;
end
%=======================

figure(1);
plot(t1000, y1000); title('Signal of 1000Hz'); hold on;

fprintf('Press any key to continue ...'); pause;

stem(t100,y100);
title('Signal of 1000Hz & 100Hz');

fprintf('Press any key to continue ...'); pause;

figure(2)
rsig = upsample(y100,10);
plot(t1000(1:length(rsig)), rsig,'r+'); title('Upsampling of 100Hz Signal');hold;

pause;

% Upsample, apply a lowpass FIR filter to rsig.

%Wn = ???;   % what is the correct value of Wn? (choose between 0..1)
            % This is the cutoff frequency of the interpolation filter                    
Freq_Signal = 2; %Hz  
Freq_Fn = 50;
Wn = Freq_Fn/F1000*2; % choose 50Hz as the cutoff (as the origianl signal is 2Hz only
                 % Wn = 0-1. where 1 corresponding to PI
%=======================
%This part will be taught in the coming weeks
%=======================
N=61; %choose N arbitrarily, which should be determined by Kaiser estimation                       
h = fir1(N,Wn);    % how does filter length affect the output?
h = F1000/F100*h;           %%% Question: why do we multiply with this constant?
%=======================
rsig2 = filter(h,1,rsig);  % what am I doing to get rsig2?
plot(t1000,rsig2,'g');         % rsig2 is plotted in green color
title('Green waveform: 1000Hz Signal after upsampling from 100Hz')