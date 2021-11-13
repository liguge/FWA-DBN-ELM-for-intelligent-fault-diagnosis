% The Whale Optimization Algorithm
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%IWOA������ʼ��%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    SearchAgents_no=30;
    Max_iter=10;
    dim=2;
    lb1=10^(-1);
    ub1=10^(3);
    lb2=10^(-2);
    ub2=10^(2);
    % initialize position vector and score for the leader����ʼ��λ���������쵼�ߵ÷�
    Leader_pos=zeros(1,dim);
    Leader_score=inf; %change this to -inf for maximization problems�����˸���Ϊ-inf�Ի��������⣬Inf�����
    %Initialize the positions of search agents��ʼ�����������λ��
    %Positions=initialization(SearchAgents_no,dim,ub,lb);%Positions�������������Ķ�άλ�á�
    %% ������ʼ���Ӻ��ٶ�
Positions=ones(SearchAgents_no,2);
for i=1:SearchAgents_no
% ���������Ⱥ
Positions(i,1) = (ub1-lb1)*rand(1,1)+lb1; % ��ʼ��Ⱥ
Positions(i,2) = (ub2-lb2)*rand(1,1)+lb2;
end
    Convergence_curve=zeros(1,Max_iter);%Convergence_curve��������
    t=0;% Loop counter
    % Main loop
    while t<Max_iter
        t
        for i=1:size(Positions,1)%��ÿ������һ��һ������Ƿ�Խ��        
            % Return back the search agents that go beyond the boundaries of
            % the search space�����س��������ռ�߽����������
            %Flag4ub=Positions(i,:)>ub;
            %Flag4lb=Positions(i,:)<lb;
            %�������ֵ�����ó����ֵ��������Сֵ�����ó���Сֵ
            %Positions(i,:)=(Positions(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;        
            % Calculate objective function for each search agent��Ŀ�꺯��ֵ�ļ���
            if Positions(i,1)>ub1||Positions(i,1)<lb1
                Positions(i,1) = (ub1-lb1)*rand(1,1)+lb1;
           end
           if Positions(i,2)>ub2||Positions(i,2)<lb2
            Positions(i,2) = (ub2-lb2)*rand(1,1)+lb2;
           end
            fitness=fun1(Positions(i,:),train_x,train_yy,type,kernel,proprecess,miny,maxy,train_y);       
            % Update the leader
            if fitness<Leader_score % Change this to > for maximization problem
                Leader_score=fitness; % Update alpha
                Leader_pos=Positions(i,:);
            end        
        end    
        %a=2-t*((2)/Max_iter); % a decreases linearly fron 2 to 0 in Eq. (2.3) 
        a=2*exp(0.15*(-log10((10*t/Max_iter)^4))-1);
        % a2 linearly dicreases from -1 to -2 to calculate t in Eq. (3.12)�������ʣ�
        a2=-1+t*((-1)/Max_iter);    
        % Update the Position of search agents����������
        for i=1:size(Positions,1)
            r1=rand(); % r1 is a random number in [0,1]
            r2=rand(); % r2 is a random number in [0,1]        
            A=2*a*r1-a;  % Eq. (2.3) in the paper
            C=2*r2;      % Eq. (2.4) in the paper  
            b=1;               %  parameters in Eq. (2.5)
            l=(a2-1)*rand+1;   %  parameters in Eq. (2.5)
            
            p = rand();        % p in Eq. (2.6)
            
            for j=1:size(Positions,2)%��ÿһ������ض�ά�Ƚ���ѭ������            
                if p<0.5%������Χ����
                    if abs(A)>=1
                        rand_leader_index = floor(SearchAgents_no*rand()+1);%floor�� X ��ÿ��Ԫ���������뵽С�ڻ���ڸ�Ԫ�ص���ӽ�����
                        X_rand = Positions(rand_leader_index, :);
                        D_X_rand=abs(C*X_rand(j)-Positions(i,j)); % Eq. (2.7)
                        Positions(i,j)=X_rand(j)-A*D_X_rand;      % Eq. (2.8)
                        
                    elseif abs(A)<1
                        D_Leader=abs(C*Leader_pos(j)-Positions(i,j)); % Eq. (2.1)
                        z=0.6*(Leader_pos(j)-Positions(i,j));
                        Positions(i,j)=Leader_pos(j)-A*D_Leader+z;      % Eq. (2.2)
                    end                
                elseif p>=0.5%��������λ��                
                    distance2Leader=abs(Leader_pos(j)-Positions(i,j));
                    % Eq. (2.5)
                    %Positions(i,j)=distance2Leader*exp(b.*l).*cos(l.*2*pi)+Leader_pos(j);
                    Positions(i,j)=distance2Leader*b*l*cos(l.*2*pi)+Leader_pos(j);
                end            
            end
        end
        t=t+1;
        Convergence_curve(t)=Leader_score;
    end
    plot(Convergence_curve,'LineWidth',2);
title(['��Ӧ������','(��ֹ����=',num2str(Max_iter),')'],'FontSize',13);
xlabel('��������');ylabel('��Ӧ��');
bestc = Leader_pos(1);
bestg = Leader_pos(2);
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


