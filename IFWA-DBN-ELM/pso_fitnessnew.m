function per_accuracy_crossvalindation= pso_fitnessnew(X,batchdata,train,train_label)

maxepoch=20;%训练rbm的次数
X=round(X);
%% 训练第1层RBM
numhid=X(1,1);
restart=1;
rbm1;%使用cd-k训练rbm，注意此rbm的可视层不是二值的，而隐含层是二值的
vishid1=vishid;hidrecbiases=hidbiases;
%% 训练第2层RBM

batchdata=batchposhidprobs;%将第一个RBM的隐含层的输出作为第二个RBM 的输入
numhid=X(1,2);%将numpen的值赋给numhid，作为第二个rbm隐含层的节点数
restart=1;
rbm1;
hidpen=vishid; penrecbiases=hidbiases; hidgenbiases=visbiases;
%% 训练第3层RBM
batchdata=batchposhidprobs;%显然，将第二个RBM的输出作为第三个RBM的输入
numhid=X(1,3);%第三个隐含层的节点数
restart=1;
rbm1;
hidpen2=vishid; penrecbiases2=hidbiases; hidgenbiases2=visbiases;
%% 训练第4层RBM

batchdata=batchposhidprobs;%显然，将第二哥RBM的输出作为第三个RBM的输入
numhid=X(1,4);%第三个隐含层的节点数
restart=1;
rbm1;
hidpen3=vishid; penrecbiases3=hidbiases; hidgenbiases3=visbiases;

 %% 训练极限学习机
 % 训练集特征输出
w1=[vishid1; hidrecbiases]; 
w2=[hidpen; penrecbiases]; 
w3=[hidpen2; penrecbiases2];
w4=[hidpen3; penrecbiases3];
digitdata = [train ones(size(train,1),1)];%x表示train数据集
w1probs = 1./(1 + exp(-digitdata*w1));%
  w1probs = [w1probs  ones(size(train,1),1)];%
w2probs = 1./(1 + exp(-w1probs*w2));%
  w2probs = [w2probs ones(size(train,1),1)];%
w3probs = 1./(1 + exp(-w2probs*w3)); %
  w3probs = [w3probs ones(size(train,1),1)];%
w4probs = 1./(1 + exp(-w3probs*w4)); % 

H_dbn = w4probs;  %%第4个rbm的实际输出值，也是elm的隐含层输出值H

%% 交叉验证
indices = crossvalind('Kfold',size(H_dbn,1),10);%对训练数据进行10折编码
%[Train, Test] = crossvalind('HoldOut', N, P) % 将原始数据随机分为两组,一组做为训练集,一组做为验证集
%[Train, Test] = crossvalind('LeaveMOut', N, M) %留M法交叉验证，默认M为1，留一法交叉验证
sum_accuracy = 0;
for i = 1:10
    %%
    cross_test = (indices == i); %每次循选取一个fold作为测试集
    cross_train = ~cross_test;   %取corss_test的补集作为训练集，即剩下9个fold
    %%
    P_train = H_dbn(cross_train,:)';
    P_test= H_dbn(cross_test,:)';
    T_train= train_label(cross_train,:)';
    T_test=train_label(cross_test,:)';
% 训练ELM
lamda=0.001;  %% 正则化系数在0.0007-0.00037之间时，一个一个试出来的
H1=P_train+1/lamda;% 加入regularization factor

T =T_train;            %训练集标签
T1=ind2vec(T);              %做分类需要先将T转换成向量索引
OutputWeight=pinv(H1') *T1'; 
Y=(H1' * OutputWeight)';

temp_Y=zeros(1,size(Y,2));
for n=1:size(Y,2)
    [max_Y,index]=max(Y(:,n));
    temp_Y(n)=index;
end
Y_train=temp_Y;
%Y_train=vec2ind(temp_Y1);
H2=P_test+1/lamda;
T_cross=(H2' * OutputWeight)';                       %   TY: the actual output of the testing data
temp_Y=zeros(1,size(T_cross,2));
for n=1:size(T_cross,2)
    [max_Y,index]=max(T_cross(:,n));
    temp_Y(n)=index;
end
TY1=temp_Y;
% 加载输出
TV=T_test;
sum_accuracy=sum_accuracy+sum(TV==TY1) / length(TV);
end


per_accuracy_crossvalindation=sum_accuracy/10;%利用交叉验证的平均精度做适应度函数
%========================================================
%===================交叉验证结束==========================
end