% Lab3 , Q1
close all; clear all;
x = [1 1 0 0 0 0 0 0]
N = length(x);
Ck = (1/N)*fft(x);
x_axis = [0:N-1];
figure; subplot(2,1,1); stem(x_axis, abs(Ck)); title('DTFS example  c(k)'); xlabel('k'); ylabel('Abs(Ck)');
subplot(2,1,2); stem(x_axis, angle(Ck)); ylabel('Angle(Ck)'); xlabel('k'); 

N = length(x);
x_axis = [0:2/N:2-(1/N)];
figure; 
subplot(2,1,1); h1=stem(x_axis,abs(Ck)*2*pi);
title('DTFT example  X(e^{j\omega})'); xlabel('*\pi(rad/sample)'); ylabel('Abs(Ck)');  xticks(x_axis);
h1(1).Marker = '^';

subplot(2,1,2); h2=stem(x_axis,angle(Ck)); ylabel('Angle(Ck)'); xlabel('*\pi(rad/sample)');  xticks(x_axis);
h2(1).Marker = 'diamond';
