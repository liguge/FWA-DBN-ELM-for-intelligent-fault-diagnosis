%初始化
clc
close all
clear
format long
tic
%==============================================================
%%导入数据
data=xlsread('F:\下载\d8ef833925dece32ae6630ba9b27564f\PSO_lssvm_prediction\1.xlsx');
[row,col]=size(data);
x=data(:,1:col-1);
y=data(:,col);
set=50; %设置测量样本数
row1=row-set;%
train_x=x(1:row1,:);
train_y=y(1:row1,:);
test_x=x(row1+1:row,:);%预测输入
test_y=y(row1+1:row,:);%预测输出
train_x=train_x';
train_y=train_y';
test_x=test_x';
test_y=test_y';
%%数据归一化
[train_x,minx,maxx, train_yy,miny,maxy] =premnmx(train_x,train_y);
test_x=tramnmx(test_x,minx,maxx);
train_x=train_x';
train_yy=train_yy';
train_y=train_y';
test_x=test_x';
test_y=test_y';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%定义lssvm相关参数
type='f';
kernel = 'RBF_kernel';
proprecess='proprecess';
%% 参数初始化
%粒子群算法中的两个参数
c1 = 1.5;%; % c1 belongs to [0,2] c1:初始为1.5,pso参数局部搜索能力，表征个体极值对当前解得影响
c2 = 1.7;%; % c2 belongs to [0,2] c2:初始为1.7,pso参数全局搜索能力，表征全局极值对当前解得影响
maxgen=100; % 进化次数 300
sizepop=20; % 种群规模30
popcmax=10^(3);% popcmax:初始为1000,SVM 参数c的变化的最大值.
popcmin=10^(-1);% popcmin:初始为0.1,SVM 参数c的变化的最小值.
popgmax=10^(2);% popgmax:初始为100,SVM 参数g的变化的最大值
popgmin=10^(-2);% popgmin:初始为0.01,SVM 参数g的变化的最小值.
k = 0.5; % k belongs to [0.1,1.0];
Vcmax = k*popcmax;%参数 c 迭代速度最大值
Vcmin = -Vcmax ;
Vgmax = k*popgmax;%参数 g 迭代速度最大值
Vgmin = -Vgmax ; 
eps = 10^(-6);
wVmax=1;
wVmin=0.01;
%% 产生初始粒子和速度
pop=ones(sizepop,2);
for i=1:sizepop
% 随机产生种群
pop(i,1) = (popcmax-popcmin)*rand(1,1)+popcmin ; % 初始种群
pop(i,2) = (popgmax-popgmin)*rand(1,1)+popgmin;
V(i,1)=Vcmax*rands(1,1); % 初始化速度
V(i,2)=Vgmax*rands(1,1);
end
fitness=ones(sizepop, 1);
for i=1:sizepop
    fitness(i)=fun1(pop(i,:),train_x,train_yy,type,kernel,proprecess,miny,maxy,train_y);
end
% 找极值和极值点
[global_fitness,bestindex]=min(fitness); % 全局极值
local_fitness=fitness; % 个体极值初始化 
global_x=pop(bestindex,:); % 全局极值点
local_x=pop; % 个体极值点初始化
% 每一代种群的平均适应度
avgfitness_gen = zeros(1,maxgen);
%% 迭代寻优
for i=1:maxgen
    i
for j=1:sizepop
%速度更新
%wV =wVmax-(wVmax-wVmin)/maxgen/maxgen*i*i   二次递减策略
%wV =wVmax-(wVmax-wVmin)/maxgen*i   %线性递减策略
wV = 1; % wV best belongs to [0.8,1.2]为速率更新公式中速度前面的弹性系数，上一个速度/位置对当前速度/位置的影响
V(j,:) = wV*V(j,:) + c1*rand*(local_x(j,:) - pop(j,:)) + c2*rand*(global_x - pop(j,:));
if V(j,1) > Vcmax %以下几个不等式是为了限定速度在最大最小之间
V(j,1) = Vcmax;
end
if V(j,1) < Vcmin
V(j,1) = Vcmin;
end
if V(j,2) > Vgmax
V(j,2) = Vgmax;
end
if V(j,2) < Vgmin
V(j,2) = Vgmin; %以上几个不等式是为了限定速度在最大最小之间
end
%种群更新
wP = 1; % wP:初始为1,种群更新公式中速度前面的弹性系数
pop(j,:)=pop(j,:)+wP*V(j,:);
if pop(j,1) > popcmax %以下几个不等式是为了限定 c 在最大最小之间
pop(j,1) = popcmax;
end
if pop(j,1) < popcmin
pop(j,1) = popcmin;
end
if pop(j,2) > popgmax %以下几个不等式是为了限定 g 在最大最小之间
pop(j,2) = popgmax;
end
if pop(j,2) < popgmin
pop(j,2) = popgmin;
end

% 自适应粒子变异      
if rand>0.5
k=ceil(2*rand);%ceil 是向离它最近的大整数圆整
if k == 1
pop(j,k) = (20-1)*rand+1;
end
if k == 2
pop(j,k) = (popgmax-popgmin)*rand+popgmin;
end 
fitness(j)=fun1(pop(j,:),train_x,train_yy,type,kernel,proprecess,miny,maxy,train_y);
%个体最优更新
if fitness(j) < local_fitness(j)
local_x(j,:) = pop(j,:);
local_fitness(j) = fitness(j);
end

if fitness(j) == local_fitness(j) && pop(j,1) < local_x(j,1)
local_x(j,:) = pop(j,:);
local_fitness(j) = fitness(j);
end 

%群体最优更新
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

%if global_fitness<0.00005%设定终止条件，避免网络过度训练，影响推广能力。
%break;
%end

end
end
%% 结果分析
plot(fit_gen,'LineWidth',2);
title(['适应度曲线','(参数c1=',num2str(c1),',c2=',num2str(c2),',终止代数=',num2str(maxgen),')'],'FontSize',13);
xlabel('进化代数');ylabel('适应度');

bestc = global_x(1);
bestg = global_x(2);

gam=bestc;
sig2=bestg;
model=initlssvm(train_x,train_yy,type,gam,sig2,kernel,proprecess);%原来是显示
model=trainlssvm(model);%原来是显示
%求出训练集和测试集的预测值
[train_predict_y,zt,model]=simlssvm(model,train_x);
[test_predict_y,zt,model]=simlssvm(model,test_x);

%预测数据反归一化
train_predict=postmnmx(train_predict_y,miny,maxy);%预测输出
test_predict=postmnmx(test_predict_y,miny,maxy);


%计算均方差
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
legend('预测输出','期望输出')
title('网络预测输出','fontsize',12)
ylabel('函数输出','fontsize',12)
xlabel('样本','fontsize',12)
 figure
plot(train_predict,':og')
hold on
plot(train_y,'- *')
legend('预测输出','期望输出')
title('网络预测输出','fontsize',12)
ylabel('函数输出','fontsize',12)
xlabel('样本','fontsize',12)

toc   %计算时间
