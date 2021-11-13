%��ʼ��
clc
close all
clear
format long
tic
%==============================================================
%%��������
data=xlsread('F:\����\d8ef833925dece32ae6630ba9b27564f\PSO_lssvm_prediction\1.xlsx');
[row,col]=size(data);
x=data(:,1:col-1);
y=data(:,col);
set=50; %���ò���������
row1=row-set;%
train_x=x(1:row1,:);
train_y=y(1:row1,:);
test_x=x(row1+1:row,:);%Ԥ������
test_y=y(row1+1:row,:);%Ԥ�����
train_x=train_x';
train_y=train_y';
test_x=test_x';
test_y=test_y';
%%���ݹ�һ��
[train_x,minx,maxx, train_yy,miny,maxy] =premnmx(train_x,train_y);
test_x=tramnmx(test_x,minx,maxx);
train_x=train_x';
train_yy=train_yy';
train_y=train_y';
test_x=test_x';
test_y=test_y';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%����lssvm��ز���
type='f';
kernel = 'RBF_kernel';
proprecess='proprecess';
%% ������ʼ��
%����Ⱥ�㷨�е���������
c1 = 1.5;%; % c1 belongs to [0,2] c1:��ʼΪ1.5,pso�����ֲ������������������弫ֵ�Ե�ǰ���Ӱ��
c2 = 1.7;%; % c2 belongs to [0,2] c2:��ʼΪ1.7,pso����ȫ����������������ȫ�ּ�ֵ�Ե�ǰ���Ӱ��
maxgen=100; % �������� 300
sizepop=20; % ��Ⱥ��ģ30
popcmax=10^(3);% popcmax:��ʼΪ1000,SVM ����c�ı仯�����ֵ.
popcmin=10^(-1);% popcmin:��ʼΪ0.1,SVM ����c�ı仯����Сֵ.
popgmax=10^(2);% popgmax:��ʼΪ100,SVM ����g�ı仯�����ֵ
popgmin=10^(-2);% popgmin:��ʼΪ0.01,SVM ����g�ı仯����Сֵ.
k = 0.5; % k belongs to [0.1,1.0];
Vcmax = k*popcmax;%���� c �����ٶ����ֵ
Vcmin = -Vcmax ;
Vgmax = k*popgmax;%���� g �����ٶ����ֵ
Vgmin = -Vgmax ; 
eps = 10^(-6);
wVmax=1;
wVmin=0.01;
%% ������ʼ���Ӻ��ٶ�
pop=ones(sizepop,2);
for i=1:sizepop
% ���������Ⱥ
pop(i,1) = (popcmax-popcmin)*rand(1,1)+popcmin ; % ��ʼ��Ⱥ
pop(i,2) = (popgmax-popgmin)*rand(1,1)+popgmin;
V(i,1)=Vcmax*rands(1,1); % ��ʼ���ٶ�
V(i,2)=Vgmax*rands(1,1);
end
fitness=ones(sizepop, 1);
for i=1:sizepop
    fitness(i)=fun1(pop(i,:),train_x,train_yy,type,kernel,proprecess,miny,maxy,train_y);
end
% �Ҽ�ֵ�ͼ�ֵ��
[global_fitness,bestindex]=min(fitness); % ȫ�ּ�ֵ
local_fitness=fitness; % ���弫ֵ��ʼ�� 
global_x=pop(bestindex,:); % ȫ�ּ�ֵ��
local_x=pop; % ���弫ֵ���ʼ��
% ÿһ����Ⱥ��ƽ����Ӧ��
avgfitness_gen = zeros(1,maxgen);
%% ����Ѱ��
for i=1:maxgen
    i
for j=1:sizepop
%�ٶȸ���
%wV =wVmax-(wVmax-wVmin)/maxgen/maxgen*i*i   ���εݼ�����
%wV =wVmax-(wVmax-wVmin)/maxgen*i   %���Եݼ�����
wV = 1; % wV best belongs to [0.8,1.2]Ϊ���ʸ��¹�ʽ���ٶ�ǰ��ĵ���ϵ������һ���ٶ�/λ�öԵ�ǰ�ٶ�/λ�õ�Ӱ��
V(j,:) = wV*V(j,:) + c1*rand*(local_x(j,:) - pop(j,:)) + c2*rand*(global_x - pop(j,:));
if V(j,1) > Vcmax %���¼�������ʽ��Ϊ���޶��ٶ��������С֮��
V(j,1) = Vcmax;
end
if V(j,1) < Vcmin
V(j,1) = Vcmin;
end
if V(j,2) > Vgmax
V(j,2) = Vgmax;
end
if V(j,2) < Vgmin
V(j,2) = Vgmin; %���ϼ�������ʽ��Ϊ���޶��ٶ��������С֮��
end
%��Ⱥ����
wP = 1; % wP:��ʼΪ1,��Ⱥ���¹�ʽ���ٶ�ǰ��ĵ���ϵ��
pop(j,:)=pop(j,:)+wP*V(j,:);
if pop(j,1) > popcmax %���¼�������ʽ��Ϊ���޶� c �������С֮��
pop(j,1) = popcmax;
end
if pop(j,1) < popcmin
pop(j,1) = popcmin;
end
if pop(j,2) > popgmax %���¼�������ʽ��Ϊ���޶� g �������С֮��
pop(j,2) = popgmax;
end
if pop(j,2) < popgmin
pop(j,2) = popgmin;
end

% ����Ӧ���ӱ���      
if rand>0.5
k=ceil(2*rand);%ceil ������������Ĵ�����Բ��
if k == 1
pop(j,k) = (20-1)*rand+1;
end
if k == 2
pop(j,k) = (popgmax-popgmin)*rand+popgmin;
end 
fitness(j)=fun1(pop(j,:),train_x,train_yy,type,kernel,proprecess,miny,maxy,train_y);
%�������Ÿ���
if fitness(j) < local_fitness(j)
local_x(j,:) = pop(j,:);
local_fitness(j) = fitness(j);
end

if fitness(j) == local_fitness(j) && pop(j,1) < local_x(j,1)
local_x(j,:) = pop(j,:);
local_fitness(j) = fitness(j);
end 

%Ⱥ�����Ÿ���
if fitness(j) < global_fitness
global_x = pop(j,:);
global_fitness = fitness(j);
end

if abs( fitness(j)-global_fitness )<=eps && pop(j,1) < global_x(1)
global_x = pop(j,:);
global_fitness = fitness(j);
end
end
fit_gen(i)=global_fitness; 
avgfitness_gen(i) = sum(fitness)/sizepop;

%if global_fitness<0.00005%�趨��ֹ�����������������ѵ����Ӱ���ƹ�������
%break;
%end

end
end
%% �������
plot(fit_gen,'LineWidth',2);
title(['��Ӧ������','(����c1=',num2str(c1),',c2=',num2str(c2),',��ֹ����=',num2str(maxgen),')'],'FontSize',13);
xlabel('��������');ylabel('��Ӧ��');

bestc = global_x(1);
bestg = global_x(2);

gam=bestc;
sig2=bestg;
model=initlssvm(train_x,train_yy,type,gam,sig2,kernel,proprecess);%ԭ������ʾ
model=trainlssvm(model);%ԭ������ʾ
%���ѵ�����Ͳ��Լ���Ԥ��ֵ
[train_predict_y,zt,model]=simlssvm(model,train_x);
[test_predict_y,zt,model]=simlssvm(model,test_x);

%Ԥ�����ݷ���һ��
train_predict=postmnmx(train_predict_y,miny,maxy);%Ԥ�����
test_predict=postmnmx(test_predict_y,miny,maxy);


%���������
trainmse=sum((train_predict-train_y).^2)/length(train_y);
%testmse=sum((test_predict-test_y).^2)/length(test_y) 

for i=1:set
RD(i)=(test_predict(i)-test_y(i))/test_y(i)*100;
end
for i=1:set
D(i)=test_predict(i)-test_y(i);
end
RD=RD';
D=D';

figure
plot(test_predict,':og')
hold on
plot(test_y,'- *')
legend('Ԥ�����','�������')
title('����Ԥ�����','fontsize',12)
ylabel('�������','fontsize',12)
xlabel('����','fontsize',12)
 figure
plot(train_predict,':og')
hold on
plot(train_y,'- *')
legend('Ԥ�����','�������')
title('����Ԥ�����','fontsize',12)
ylabel('�������','fontsize',12)
xlabel('����','fontsize',12)

toc   %����ʱ��
