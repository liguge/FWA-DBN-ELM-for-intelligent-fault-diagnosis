function [E]=bls(y)
%      y=load('003.txt');
y_hht=hilbert(y);%ϣ�����ر任
y_an=abs(y_hht);%�����ź�
y_an=y_an-mean(y_an);%ȥ��ֱ������
y_an_nfft= 2^nextpow2(length(y_an));%�����DFT�Ĳ������� ȡ2���ݴη�
y_an_ft=fft(y_an,y_an_nfft);%�����DFT
  c=2*abs(y_an_ft(1:round(y_an_nfft/2)))/length(y_an);%�����ֵ
%  c=2*abs(y_an_ft(1:y_an_nfft/2))/length(y_an_nfft)
%   c=2*abs(y_an_ft(1:y_an_nfft/2))/length(y_an);
p=(c+eps)/(sum(c+eps));
E = -sum(p.*log(p));
% plot(x,c);
