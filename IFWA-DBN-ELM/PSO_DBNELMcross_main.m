clear 
close all
format compact 
format long
%% 1.数据加载
fprintf(1,'加载数据 \n');
load('001');%其中1-173为1类，174-343为2类 344-510为3类 511-600为4类，各选择20%作为测试集
%第一类173组
[i1 i2]=sort(rand(15,1)); 
train(1:12,:)=input(i2(1:12),:);     train_label(1:12,1)=output(i2(1:12),1);
test(1:3,:)=input(i2(13:15),:);     test_label(1:3,1)=output(i2(13:15),1);
%第二类有170组,16-30为2类
[i1 i2]=sort(rand(15,1));
train(13:24,:)=input(15+i2(1:12),:);    train_label(13:24,1)=output(15+i2(1:12),1);
test(4:6,:)=input(15+i2(13:15),:);     test_label(4:6,1)=output(15+i2(13:15),1);
%第三类有167,31-45为3类
[i1 i2]=sort(rand(15,1));
train(25:36,:)=input(30+i2(1:12),:);    train_label(25:36,1)=output(30+i2(1:12),1);
test(7:9,:)=input(30+i2(13:15),:);     test_label(7:9,1)=output(30+i2(13:15),1);
%第4类有90,46-60为4类
[i1 i2]=sort(rand(15,1));
train(37:48,:)=input(45+i2(1:12),:);    train_label(37:48,1)=output(45+i2(1:12),1);
test(10:12,:)=input(45+i2(13:15),:);     test_label(10:12,1)=output(45+i2(13:15),1); 
clear i1 i2 input output
% %%打乱顺序
k=rand(48,1);[m n]=sort(k);
train=train(n(1:48),:);train_label=train_label(n(1:48),:);
k=rand(12,1);[m n]=sort(k);
test=test(n(1:12),:);test_label=test_label(n(1:12),:);
% clear k m n
%no_dims = round(intrinsic_dim(train, 'MLE')); %round四舍五入
%disp(['MLE estimate of intrinsic dimensionality: ' num2str(no_dims)]);
numbatches=4;%数据分块数
numcases=12;%每块数据集的样本个数（不能太小）块数不能超过样本数
numdims=size(train,2);%单个样本的维数
% 训练数据
x=train;%将数据转换成DBN的数据格式
for i=1:numbatches
    train1=x((i-1)*numcases+1:i*numcases,:);
    batchdata(:,:,i)=train1;
end%将分好的10组数据都放在batchdata中
%% rbm参数
maxepoch=20;%训练rbm的次数
hid=4; %隐含层数
hmax=500;hmin=100; %各隐含层节点数取值区间
tic;
%%
%h=PSO_dbnelm_cross(hid,hmax,hmin,batchdata,train,train_label); %PSO优化隐含层节点数
%%
t1=toc
tic;
%h=round(h);
numpen0=250; numpen1=250; numpen2=250;numpen3=250; %dbn最终隐含层的节点数
disp('构建一个num2str(H)层的置信网络');
clear i 
%% 训练第1层RBM
fprintf(1,'Pretraining Layer 1 with RBM: %d-%d \n',numdims,numpen0);%6400-500
numhid=numpen0;
restart=1;
rbm1;%使用cd-k训练rbm，注意此rbm的可视层不是二值的，而隐含层是二值的
vishid1=vishid;hidrecbiases=hidbiases;
%% 训练第2层RBM
fprintf(1,'\nPretraining Layer 2 with RBM: %d-%d \n',numpen0,numpen1);%500-200
batchdata=batchposhidprobs;%将第一个RBM的隐含层的输出作为第二个RBM 的输入
numhid=numpen1;%将numpen的值赋给numhid，作为第二个rbm隐含层的节点数
restart=1;
rbm1;
hidpen=vishid; penrecbiases=hidbiases; hidgenbiases=visbiases;
%% 训练第3层RBM
fprintf(1,'\nPretraining Layer 3 with RBM: %d-%d \n',numpen1,numpen2);%200-100
batchdata=batchposhidprobs;%显然，将第二哥RBM的输出作为第三个RBM的输入
numhid=numpen2;%第三个隐含层的节点数
restart=1;
rbm1;
hidpen2=vishid; penrecbiases2=hidbiases; hidgenbiases2=visbiases;
%% 训练第4层RBM
fprintf(1,'\nPretraining Layer 4 with RBM: %d-%d \n',numpen2,numpen3);%200-100
batchdata=batchposhidprobs;%显然，将第二哥RBM的输出作为第三个RBM的输入
numhid=numpen3;%第三个隐含层的节点数
restart=1;
rbm1;
hidpen3=vishid; penrecbiases3=hidbiases; hidgenbiases3=visbiases;

 %% 训练极限学习机
 % 训练集特征输出
w1=[vishid1; hidrecbiases]; 
w2=[hidpen; penrecbiases]; 
w3=[hidpen2; penrecbiases2];
w4=[hidpen3; penrecbiases3];
digitdata = [x ones(size(x,1),1)];%x表示train数据集
w1probs = 1./(1 + exp(-digitdata*w1));%
  w1probs = [w1probs  ones(size(x,1),1)];%
w2probs = 1./(1 + exp(-w1probs*w2));%
  w2probs = [w2probs ones(size(x,1),1)];%
w3probs = 1./(1 + exp(-w2probs*w3)); %
  w3probs = [w3probs ones(size(x,1),1)];%
w4probs = 1./(1 + exp(-w3probs*w4)); % 
H = w4probs';  %%第三个rbm的实际输出值，也是elm的输入值H
lamda=0.001;  %% 正则化系数在0.0007-0.00037之间时  测试精度最大81.667%
H=H+1/lamda;  %加入regularization factor
T =train_label';            %训练集标签
T1=ind2vec(T);              %做分类需要先将T转换成向量索引
OutputWeight=pinv(H') *T1'; 
Y=(H' * OutputWeight)';
temp_Y=zeros(1,size(Y,2));
for n=1:size(Y,2)
    [max_Y,index]=max(Y(:,n));
    temp_Y(n)=index;
end
Y_train=temp_Y;
%Y_train=vec2ind(temp_Y1);
% 训练集准确率
train_accuracy=sum(Y_train==T)/length(T)
% 训练集实际分类与预测分类对比
figure(1)
plot(Y_train,'bo');hold on 
plot(T,'r*');

%% 测试极限学习机
N2 = size(test,1);
% 测试集特征输出
w1=[vishid1; hidrecbiases]; %(784+1*500)
w2=[hidpen; penrecbiases]; %(500+1*500)
w3=[hidpen2; penrecbiases2];%(500+1*2000)
w4=[hidpen3; penrecbiases3];
test1 = [test ones(N2,1)];
w1probs = 1./(1 + exp(-test1*w1));
  w1probs = [w1probs  ones(N2,1)];
w2probs = 1./(1 + exp(-w1probs*w2)); 
  w2probs = [w2probs ones(N2,1)];
w3probs = 1./(1 + exp(-w2probs*w3)); 
  w3probs = [w3probs ones(N2,1)];
w4probs = 1./(1 + exp(-w3probs*w4));  
H1=w4probs';
%加入正则化系数
H1=H1+1/lamda;
TY=(H1' * OutputWeight)';     %   TY: the actual output of the testing data
temp_Y=zeros(1,size(TY,2));
for n=1:size(TY,2)
    [max_Y,index]=max(TY(:,n));
    temp_Y(n)=index;
end
TY1=temp_Y;
% 加载输出
TV=test_label';
% 测试集分类准确率
test_accuracy = sum(TV==TY1) / length(TV)
% 测试集实际分类与预测分类对比
figure(2)
plot(TV,'r*');
hold on
plot(TY1,'bo');
xlabel('测试集样本数')
ylabel('标签种类')
title('测试阶段：实际输出与理想输出的差');
legend('真实值','预测值')
% 程序运行时间
t2=toc