function [E]=bls(y)
%      y=load('003.txt');
y_hht=hilbert(y);%希尔伯特变换
y_an=abs(y_hht);%包络信号
y_an=y_an-mean(y_an);%去除直流分量
y_an_nfft= 2^nextpow2(length(y_an));%包络的DFT的采样点数 取2的幂次方
y_an_ft=fft(y_an,y_an_nfft);%包络的DFT
  c=2*abs(y_an_ft(1:round(y_an_nfft/2)))/length(y_an);%包络幅值
%  c=2*abs(y_an_ft(1:y_an_nfft/2))/length(y_an_nfft)
%   c=2*abs(y_an_ft(1:y_an_nfft/2))/length(y_an);
p=(c+eps)/(sum(c+eps));
E = -sum(p.*log(p));
% plot(x,c);
