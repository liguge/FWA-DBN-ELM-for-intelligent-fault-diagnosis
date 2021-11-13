% 计算初始适应度
function F =fun1(X,train_x,train_yy,type,kernel,proprecess,miny,maxy,train_y)
gam=X(:,1);
sig2=X(:,2);
model=initlssvm(train_x,train_yy,type,gam,sig2,kernel,proprecess);
model=trainlssvm(model);
%求出训练集和测试集的预测值
[train_predict_y,zt,model]=simlssvm(model,train_x);
% [test_predict_y,zt,model]=simlssvm(model,test_x);
%预测数据反归一化
train_predict=postmnmx(train_predict_y ,miny,maxy);%预测输出
%  test_predict=postmnmx(test_predict_y ,miny,maxy); %测试集预测值
%计算均方差

trainmse=sum((train_predict-train_y).^2)/length(train_y);
% testmse=sum((test_predict-test_y).^2)/length(test_y); 
F=trainmse; %以测试集的预测值计算的均方差为适应度值
end
