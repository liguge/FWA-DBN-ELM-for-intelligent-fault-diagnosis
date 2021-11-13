%% �̻��㷨���к����Ż�
%%��������Сֵ
%fitness��Ӧ�Ⱥ�����N�̻�����D����ά����M���������Er��ը�뾶��En��ը��Ŀ
%LB,UB�ֱ�Ϊ�������½磬TΪ����������a,bΪ��ը��Ŀ��������
clear;clc
N=30;   % N�̻���
D=2;     % D����ά��
M=5;     % M�������
En=6;    % En��ը��Ŀ
Er=5;    % Er��ը�뾶
a=0.3;   % a,bΪ��ը��Ŀ��������
b=0.6;
T=15;   % TΪ��������

%�����ֵ�������½�
% LB=[-10,-10];
% UB=[10,10];

%����ڽ�ռ��ʼ��N���̻�λ��
x = zeros(N,D);
for i=1:N
        x(i,1)=unifrnd(100,500,1,1);  %�����ʼ��λ��
        x(i,2)=unifrnd(69,79,1,1);
end
%ѭ������
E_Spark=zeros(T,D,N);
Fit = zeros(1,N);
F = zeros(1,T);
for t=1:T
    %����ÿ���̻���Ӧ��ֵ
    for i=1:N
        Fit(i)=fun_1(x(i,:));
    end
    [F(t),~]=min(Fit);
    Fmin=min(Fit);
    Fmax=max(Fit);
    %����ÿ���̻��ı�ը�뾶E_R�ͱ�ը��ĿE_N�Լ������ı�ը��
    E_R = zeros(1,N);
    E_N = zeros(1,N);
    for i=1:N
        E_R(i)=Er*((Fit(i)-Fmin+eps)/(sum(Fit)-N*Fmin+eps));  %��ը�뾶
        E_N(i)=En*((Fmax-Fit(i)+eps)/(N*Fmax-sum(Fit)+eps));  %��ը��Ŀ
        if E_N(i)<a*En    % ��ը��Ŀ����
            E_N(i)=round(a*En);
        elseif E_N(i)>b*En
            E_N(i)=round(b*En);
        else
            E_N(i)=round(E_N(i));
        end
        %������ը�� E_Spark
        for j=2:(E_N(i)+1)              % ��i���̻�������E_N(i)����
            E_Spark(1,:,i)=x(i,:);      % ����i���̻�����Ϊ��i���������еĵ�һ������ը�����Ļ𻨴������еĵڶ�����ʼ�洢�����̻�Ϊ��ά����ÿһҳ�ĵ�һ�У�
            h=E_R(i)*(-1+2*rand(1,D));  % λ��ƫ��
            E_Spark(j,:,i)=x(i,:)+h;    % ��i���̻�����ά�����iҳ�������ĵ�j����ά�����j�У�����
%             for k=1:D   %Խ����
%                 if E_Spark(j,k,i)>UB(k)||E_Spark(j,k,i)<LB(k)  %��i���̻�����ά�����iҳ�������ĵ�j���𻨣���ά�����j�У��ĵ�k����������ά�����k�У�
%                     E_Spark(j,k,i)=LB(k)+rand*(UB(k)-LB(k));   %ӳ�����
%                 end
%             end
                if E_Spark(j,1,i)>500||E_Spark(j,1,i)<100
             E_Spark(j,1,i)=unifrnd(100,500,1,1); 
                end
                if E_Spark(j,2,i)>79||E_Spark(j,2,i)<69
             E_Spark(j,2,i)= unifrnd(69,79,1,1);
                end
        end
    end
    %������˹�����Mut_Spark,���ѡ��M���̻����б���
    Mut=randperm(N);          % �������1-N�ڵ�N����
    for m1=1:M                % M�������̻�
        m=Mut(m1);            % ���ѡȡ�̻�
        for n=1:E_N(m)
            e=1+sqrt(1)*randn(1,D); %��˹�������������Ϊ1����ֵҲΪ1��1*D�������
            E_Spark(n,:,m)=E_Spark(n,:,m).*e;
%             for k=1:D   %Խ����
%                 if E_Spark(n,k,m)>UB(k)||E_Spark(n,k,m)<LB(k)  %��i���̻�����ά�����iҳ�������ĵ�j���𻨣���ά�����j�У��ĵ�k����������ά�����k�У�
%                     E_Spark(n,k,m)=LB(k)+rand*(UB(k)-LB(k));   %ӳ�����
%                 end
                if E_Spark(n,1,m)>500||E_Spark(n,1,m)<100
             E_Spark(n,1,m)=unifrnd(100,500,1,1); 
                end
                if E_Spark(n,2,m)>79||E_Spark(n,2,m)<69
             E_Spark(n,2,m)= unifrnd(69,79,1,1);
                end
%             end
        end
    end
    
    %ѡ����������̻�����ը�𻨡�����������������ά�����У�ѡȡN������������Ϊ��һ�����Ƚ����Ÿ������£�Ȼ��ʣ�µ�N-1�������̶�ԭ��ѡȡ��
    n=sum(E_N)+N; %�̻������ܸ���
    q=1;
    Fitness = zeros(1,1);
    E_Sum = zeros(1,D);
    for i=1:N  % ��άת��ά
        for j=1:(E_N(i)+1)  % ��ά����ÿһҳ����������ÿ���̻���������Ļ���֮�ͣ�
            E_Sum(q,:)=E_Spark(j,:,i); % �̻��������
            Fitness(q)=fun_1(E_Sum(q,:)); % ���������̻����𻨵���Ӧ�ȣ�����ѡȡ���Ÿ���
            q=q+1;
        end
    end
    [Fitness,X]=sort(Fitness);  % ��Ӧ����������
    x(1,:)=E_Sum(X(1),:);    % ���Ÿ���
    dist=pdist(E_Sum);       % �������������ŷʽ����
    S=squareform(dist);      % �������������ų�n*n���飬��i��֮�ͼ�Ϊ��i���𻨵������𻨵ľ���֮��
    P = zeros(1,n);
    for i=1:n                % �ֱ������֮��
        P(i)=sum(S(i,:));
    end
    [P,Ix]=sort(P,'descend');% �����밴�������У�ѡȡǰN-1����ָ������������ܶȽϸߣ����ø�����Χ�кܶ�������ѡ�߸���ʱ���ø��屻ѡ��ĸ��ʻή��
    for i=1:(N-1)
        x(i+1,:)=E_Sum(Ix(i),:);
    end
end

% for i=1:N
%     Fit(i)=fitness(x(i,:));
% end


%�����ֵ���
[F(T),Y]=min(Fit);
fmin=min(F);
xm=x(Y,:);
fprintf('xm=%f  %f\n',xm(1),xm(2));
fprintf('fmin=%f\n',fmin);
figure(1);
t=1:T;
plot(t,F)
xlabel('��������')
ylabel('Ŀ�꺯��ֵ')
title('FWA�㷨��������');
