% 适应度函数
function F =fun_1(X)
% 函数功能：计算该个体对应适应度值
% x           input     个体
% fitness     output    个体适应度值

filterSize=round(X(1));
termIter=30;
T=round(X(2));
M=3;
plotMode=0;
x=importdata('001.txt');
%--------------- Run actual VMD code
[y_final] = mckd(x,filterSize,termIter,T,M,plotMode);
% AU=zeros(1,4096) ;
%     AU = AU+u(kk,:);
    [E]=bls(y_final);
    F= E;  
end

