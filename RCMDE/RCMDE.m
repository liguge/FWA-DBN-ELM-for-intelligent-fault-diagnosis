function Out_RCMDE=RCMDE(x,m,c,tau,Scale)
%
% This function calculates the refined composite multiscale dispersion entropy (RCMDE) of a univariate signal x
%
% Inputs:
%
% x: univariate signal - a vector of size 1 x N (the number of sample points)
% m: embedding dimension
% c: number of classes (it is usually equal to a number between 3 and 9 - we used c=6 in our studies)
% tau: time lag (it is usually equal to 1)
% Scale: number of scale factors
%
%Outputs:
%
% Out_RCMDE: a vector of size 1 * Scale - the RCMDE of x
%
%
% Ref:
% [1] H. Azami, M. Rostaghi, D. Abasolo, and J. Escudero, "Refined Composite Multiscale Dispersion Entropy and its Application to Biomedical
% Signals", IEEE Transactions on Biomedical Engineering, 2017.
% [2] M. Rostaghi and H. Azami, "Dispersion Entropy: A Measure for Time-Series Analysis", IEEE Signal Processing Letters. vol. 23, n. 5, pp. 610-614, 2016.
%
% If you use the code, please make sure that you cite references [1] and [2].
%
% Hamed Azami and Javier Escudero Rodriguez
% Emails: hamed.azami@ed.ac.uk and javier.escudero@ed.ac.uk
%
%  20-January-2017
%%
x=load('b2.txt');
m=3;
c=6;
tau=1;
Scale=20;
ti=toc
tic;
Out_RCMDE=NaN*ones(1,Scale);

Out_RCMDE(1)=DisEn_NCDF(x,m,c,tau);

sigma=std(x);
mu=mean(x);

for j=1:Scale
    pdf=[];
    for jj=1:j
        xs = Multi(x(jj:end),j);
        [DE, T_pdf]=DisEn_NCDF_ms(xs,m,c,mu,sigma,tau);
        pdf=[pdf ; T_pdf];
    end
    pdf=mean(pdf,1);
    pdf=pdf(pdf~=0);
    Out_RCMDE(j)=-sum(pdf .* log(pdf));
end


function M_Data = Multi(Data,S)

%  generate the consecutive coarse-grained time series
%  Input:   Data: time series;
%           S: the scale factor
% Output:
%           M_Data: the coarse-grained time series at the scale factor S

L = length(Data);
J = fix(L/S);

for i=1:J
    M_Data(i) = mean(Data((i-1)*S+1:i*S));
end
end
t2=toc
end
