clear;clc
N = 30; % ��Ⱥ����
c1 = 2; % ����Ⱥ����
c2 = 2; % ����Ⱥ����
 w = 0.5;                         % ����Ȩ�����ֵ
%  Wmin = 0.2;                         % ����Ȩ����Сֵ
T = 20;    % ѭ����������
D = 2;      % ��Ⱥ�и������,2��δ֪��
Vmax = 1000;                            % �ٶ����ֵ
Vmin = -1000;                           % �ٶ���Сֵ
%------��ʼ����Ⱥ�ĸ���------------
x=ones(N,D);
% v=ones(N,D);x=��1,2,3,4����
v = rand(N, D)*(Vmax - Vmin) + Vmin;
for i=1:N
%      for j=1:D
        x(i,1)=unifrnd(100,500,1,1);  %�����ʼ��λ��
        x(i,2)=unifrnd(107,117,1,1); 
        
end
 
% randint(1,1,[low high]);

%% ------�ȼ���������ӵ���Ӧ��----------------------
% p = round(x);
p=x;
pbest = ones(N, 1);
for i = 1:N
   pbest(i) = fun_1(x(i, :));
end
%%%%%%%%%%  ��ʼ��ȫ������λ�ú�����ֵ %%%%%%%%%
g = ones(1, D);
gbest = inf;
for i = 1:N
   if pbest(i) < gbest
      g = p(i, :);
      gbest = pbest(i);
   end
end
gb = ones(1, T);
%% ------������Ҫѭ��------------
for i=1:T
    for j = 1:N
        %%%%%%%%%%  ���¸�������λ�ú�����ֵ %%%%%%%%%
       if fun_1(x(j, :)) < pbest(j)
         p(j, :) = x(j, :);
         pbest(j) = fun_1(x(j, :));
      end
      %%%%%%%%%%  ����ȫ������λ�ú�����ֵ %%%%%%%%%
      if pbest(j) < gbest
         g = p(j, :);
         gbest = pbest(j);
      end
           %%%%%%%%%%  ��̬�������Ȩ��ֵ %%%%%%%%%
%         w = Wmax -(Wmax - Wmin)*i/T;

        %%%%%%%%%%  ����λ�ú��ٶ�ֵ %%%%%%%%%
        v(j, :) =w*v(j, :) + c1*rand*(p(j, :) - x(j, :)) + c2*rand*(g - x(j, :));
        x(j, :) = x(j, :) + v(j, :);
        %%%%%%%%%%  �߽��������� %%%%%%%%%
        if x(j,1)>500||x(j,1)<100
             x(j,1)=unifrnd(100,500,1,1); 
        end
        if x(j,2)>117||x(j,2)<107
           x(j,2)= unifrnd(107,117,1,1);
        end
        for ii = 1:D
         if (v(j, ii) > Vmax) || (v(j, ii) < Vmin)
            v(j, ii) = rand*(Vmax - Vmin) + Vmin;
         end
        end
    end
   %%%%%%%%%%  ��¼����ȫ������ֵ %%%%%%%%%
      gb(i) = gbest;
end
disp(['���Ÿ��壺' num2str(g)]);
disp(['����ֵ��' num2str(gb(end))]);
disp(['����ֵ��' num2str(gb)]);
plot(gb, 'LineWidth', 2);
xlabel('��������');
ylabel('��Ӧ��ֵ');
title('��Ӧ�Ƚ�������');