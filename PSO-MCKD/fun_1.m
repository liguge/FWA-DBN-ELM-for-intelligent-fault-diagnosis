% ��Ӧ�Ⱥ���
function F =fun_1(X)
% �������ܣ�����ø����Ӧ��Ӧ��ֵ
% x           input     ����
% fitness     output    ������Ӧ��ֵ

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

