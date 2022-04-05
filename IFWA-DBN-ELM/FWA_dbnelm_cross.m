function h = PSO_dbnelm_cross(hid,hmin,hmax,batchdata,train,train_label)
%% 参数设定  
N=15;   % N烟花数
    % D变量维数
D=hid;    %将变量D赋值给hid
M=5;     % M变异火花数
En=6;    % En爆炸数目
Er=5;    % Er爆炸半径
a=0.3;   % a,b为爆炸数目限制因子
b=0.6;
T=20;   % T为迭代次数

%求最大值变量上下界
LB=[hmin,hmin,hmin,hmin];
UB=[hmax,hmax,hmax,hmax];
%随机在解空间初始化N个烟花位置
 x = zeros(N,D);
for i=1:N
    x(i,:)=LB+rand(1,D).*(UB-LB);
end
%循环迭代
E_Spark=zeros(T,D,N);
Fit = zeros(1,N);
F = zeros(1,T);
for t=1:T     %计算每个烟花适应度值
    x=round(x);
    for i=1:N
        Fit(i)=pso_fitnessnew(x(i,:),batchdata,train,train_label);
    end
    [F(t),~]=min(Fit);
    Fmin=min(Fit);
    Fmax=max(Fit);
    %计算每个烟花的爆炸半径E_R和爆炸数目E_N以及产生的爆炸火花
    E_R = zeros(1,N);
    E_N = zeros(1,N);
    for i=1:N
        E_R(i)=Er*((Fit(i)-Fmin+eps)/(sum(Fit)-N*Fmin+eps));  %爆炸半径
        E_N(i)=En*((Fmax-Fit(i)+eps)/(N*Fmax-sum(Fit)+eps));  %爆炸数目
        if E_N(i)<a*En    % 爆炸数目限制
            E_N(i)=round(a*En);
        elseif E_N(i)>b*En
            E_N(i)=round(b*En);
        else
            E_N(i)=round(E_N(i));
        end
        %产生爆炸火花 E_Spark
        for j=2:(E_N(i)+1)              % 第i个烟花共产生E_N(i)个火花
            E_Spark(1,:,i)=x(i,:);      % 将第i个烟花保存为第i个火花序列中的第一个，爆炸产生的火花从序列中的第二个开始存储（即烟花为三维数组每一页的第一行）
            h=E_R(i)*(-1+2*rand(1,D));  % 位置偏移
            E_Spark(j,:,i)=x(i,:)+h;    % 第i个烟花（三维数组的i页）产生的第j（三维数组的j行）个火花
            for k=1:D   %越界检测
                if E_Spark(j,k,i)>UB(k)||E_Spark(j,k,i)<LB(k)  %第i个烟花（三维数组的i页）产生的第j个火花（三维数组的j行）的第k个变量（三维数组的k列）
                    E_Spark(j,k,i)=LB(k)+rand*(UB(k)-LB(k));   %映射规则
                end
            end
        end
    end
    %产生高斯变异火花Mut_Spark,随机选择M个烟花进行变异
    Mut=randperm(N);          % 随机产生1-N内的N个数
    for m1=1:M                % M个变异烟花
        m=Mut(m1);            % 随机选取烟花
        for n=1:E_N(m)
            e=1+sqrt(1)*randn(1,D); %高斯变异参数，方差为1，均值也为1的1*D随机矩阵
            E_Spark(n,:,m)=E_Spark(n,:,m).*e;
            for k=1:D   %越界检测
                if E_Spark(n,k,m)>UB(k)||E_Spark(n,k,m)<LB(k)  %第i个烟花（三维数组的i页）产生的第j个火花（三维数组的j行）的第k个变量（三维数组的k列）
                    E_Spark(n,k,m)=LB(k)+rand*(UB(k)-LB(k));   %映射规则
                end
            end
        end
    end
    
    %选择操作，从烟花、爆炸火花、变异火花里（都包含在三维数组中）选取N个优良个体作为下一代（先将最优个体留下，然后剩下的N-1个按轮盘赌原则选取）
    n=sum(E_N)+N; %烟花、火花总个数
    q=1;
    Fitness = zeros(1,1);
    E_Sum = zeros(1,D);
    for i=1:N  % 三维转二维
        for j=1:(E_N(i)+1)  % 三维数组每一页的行数（即每个烟花及其产生的火花数之和）
            E_Sum(q,:)=E_Spark(j,:,i); % 烟花与火花总量
%             Fitness(q)=fitness(E_Sum(q,:)); % 计算所有烟花、火花的适应度，用于选取最优个体
           Fitness(q)=pso_fitnessnew(E_Sum(q,:),batchdata,train,train_label);
            q=q+1;
        end
    end
    [Fitness,X]=sort(Fitness);  % 适应度升序排列
    x(1,:)=E_Sum(X(1),:);    % 最优个体
    dist=pdist(E_Sum);       % 求解各火花两两间的欧式距离
    S=squareform(dist);      % 将距离向量重排成n*n数组，第i行之和即为第i个火花到其他火花的距离之和
    P = zeros(1,n);
    for i=1:n                % 分别求各行之和
        P(i)=sum(S(i,:));
    end
    [P,Ix]=sort(P,'descend');% 将距离按降序排列，选取前N-1个，指的是如果个体密度较高，即该个体周围有很多其他候选者个体时，该个体被选择的概率会降低
    for i=1:(N-1)
        x(i+1,:)=E_Sum(Ix(i),:);
    end
end

% for i=1:N
%     Fit(i)=pso_fitnessnew(x(i,:),batchdata,train,train_label);
% end
toc

%求最大值输出
[F(T),Y]=min(Fit);
fmin=min(F);
xm=round(x(Y,:));
h=round(x(Y,:));
X=round(xm);
disp(['最优解为:',num2str(xm)]); 
disp(['最优值为:',num2str(fmin)]);
figure(1);
t=1:T;
plot(t,F)
xlabel('迭代次数')
ylabel('目标函数值')
title('FWA算法迭代曲线')
end
