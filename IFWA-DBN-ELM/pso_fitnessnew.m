function per_accuracy_crossvalindation= pso_fitnessnew(X,batchdata,train,train_label)

maxepoch=20;%ѵ��rbm�Ĵ���
X=round(X);
%% ѵ����1��RBM
numhid=X(1,1);
restart=1;
rbm1;%ʹ��cd-kѵ��rbm��ע���rbm�Ŀ��Ӳ㲻�Ƕ�ֵ�ģ����������Ƕ�ֵ��
vishid1=vishid;hidrecbiases=hidbiases;
%% ѵ����2��RBM

batchdata=batchposhidprobs;%����һ��RBM��������������Ϊ�ڶ���RBM ������
numhid=X(1,2);%��numpen��ֵ����numhid����Ϊ�ڶ���rbm������Ľڵ���
restart=1;
rbm1;
hidpen=vishid; penrecbiases=hidbiases; hidgenbiases=visbiases;
%% ѵ����3��RBM
batchdata=batchposhidprobs;%��Ȼ�����ڶ���RBM�������Ϊ������RBM������
numhid=X(1,3);%������������Ľڵ���
restart=1;
rbm1;
hidpen2=vishid; penrecbiases2=hidbiases; hidgenbiases2=visbiases;
%% ѵ����4��RBM

batchdata=batchposhidprobs;%��Ȼ�����ڶ���RBM�������Ϊ������RBM������
numhid=X(1,4);%������������Ľڵ���
restart=1;
rbm1;
hidpen3=vishid; penrecbiases3=hidbiases; hidgenbiases3=visbiases;

 %% ѵ������ѧϰ��
 % ѵ�����������
w1=[vishid1; hidrecbiases]; 
w2=[hidpen; penrecbiases]; 
w3=[hidpen2; penrecbiases2];
w4=[hidpen3; penrecbiases3];
digitdata = [train ones(size(train,1),1)];%x��ʾtrain���ݼ�
w1probs = 1./(1 + exp(-digitdata*w1));%
  w1probs = [w1probs  ones(size(train,1),1)];%
w2probs = 1./(1 + exp(-w1probs*w2));%
  w2probs = [w2probs ones(size(train,1),1)];%
w3probs = 1./(1 + exp(-w2probs*w3)); %
  w3probs = [w3probs ones(size(train,1),1)];%
w4probs = 1./(1 + exp(-w3probs*w4)); % 

H_dbn = w4probs;  %%��4��rbm��ʵ�����ֵ��Ҳ��elm�����������ֵH

%% ������֤
indices = crossvalind('Kfold',size(H_dbn,1),10);%��ѵ�����ݽ���10�۱���
%[Train, Test] = crossvalind('HoldOut', N, P) % ��ԭʼ���������Ϊ����,һ����Ϊѵ����,һ����Ϊ��֤��
%[Train, Test] = crossvalind('LeaveMOut', N, M) %��M��������֤��Ĭ��MΪ1����һ��������֤
sum_accuracy = 0;
for i = 1:10
    %%
    cross_test = (indices == i); %ÿ��ѭѡȡһ��fold��Ϊ���Լ�
    cross_train = ~cross_test;   %ȡcorss_test�Ĳ�����Ϊѵ��������ʣ��9��fold
    %%
    P_train = H_dbn(cross_train,:)';
    P_test= H_dbn(cross_test,:)';
    T_train= train_label(cross_train,:)';
    T_test=train_label(cross_test,:)';
% ѵ��ELM
lamda=0.001;  %% ����ϵ����0.0007-0.00037֮��ʱ��һ��һ���Գ�����
H1=P_train+1/lamda;% ����regularization factor

T =T_train;            %ѵ������ǩ
T1=ind2vec(T);              %��������Ҫ�Ƚ�Tת������������
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
% �������
TV=T_test;
sum_accuracy=sum_accuracy+sum(TV==TY1) / length(TV);
end


per_accuracy_crossvalindation=sum_accuracy/10;%���ý�����֤��ƽ����������Ӧ�Ⱥ���
%========================================================
%===================������֤����==========================
end