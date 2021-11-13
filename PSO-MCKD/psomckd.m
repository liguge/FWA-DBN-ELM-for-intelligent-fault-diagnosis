clear;clc
N = 30; % 种群个数
c1 = 2; % 粒子群参数
c2 = 2; % 粒子群参数
 w = 0.5;                         % 惯性权重最大值
%  Wmin = 0.2;                         % 惯性权重最小值
T = 20;    % 循环迭代步数
D = 2;      % 种群中个体个数,2个未知数
Vmax = 1000;                            % 速度最大值
Vmin = -1000;                           % 速度最小值
%------初始化种群的个体------------
x=ones(N,D);
% v=ones(N,D);x=（1,2,3,4）；
v = rand(N, D)*(Vmax - Vmin) + Vmin;
for i=1:N
%      for j=1:D
        x(i,1)=unifrnd(100,500,1,1);  %随机初始化位置
        x(i,2)=unifrnd(107,117,1,1); 
        
end
 
% randint(1,1,[low high]);

%% ------先计算各个粒子的适应度----------------------
% p = round(x);
p=x;
pbest = ones(N, 1);
for i = 1:N
   pbest(i) = fun_1(x(i, :));
end
%%%%%%%%%%  初始化全局最优位置和最优值 %%%%%%%%%
g = ones(1, D);
gbest = inf;
for i = 1:N
   if pbest(i) < gbest
      g = p(i, :);
      gbest = pbest(i);
   end
end
gb = ones(1, T);
%% ------进入主要循环------------
for i=1:T
    for j = 1:N
        %%%%%%%%%%  更新个体最优位置和最优值 %%%%%%%%%
       if fun_1(x(j, :)) < pbest(j)
         p(j, :) = x(j, :);
         pbest(j) = fun_1(x(j, :));
      end
      %%%%%%%%%%  更新全局最优位置和最优值 %%%%%%%%%
      if pbest(j) < gbest
         g = p(j, :);
         gbest = pbest(j);
      end
           %%%%%%%%%%  动态计算惯性权重值 %%%%%%%%%
%         w = Wmax -(Wmax - Wmin)*i/T;

        %%%%%%%%%%  更新位置和速度值 %%%%%%%%%
        v(j, :) =w*v(j, :) + c1*rand*(p(j, :) - x(j, :)) + c2*rand*(g - x(j, :));
        x(j, :) = x(j, :) + v(j, :);
        %%%%%%%%%%  边界条件处理 %%%%%%%%%
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
   %%%%%%%%%%  记录历代全局最优值 %%%%%%%%%
      gb(i) = gbest;
end
disp(['最优个体：' num2str(g)]);
disp(['最优值：' num2str(gb(end))]);
disp(['最优值：' num2str(gb)]);
plot(gb, 'LineWidth', 2);
xlabel('迭代次数');
ylabel('适应度值');
title('适应度进化曲线');