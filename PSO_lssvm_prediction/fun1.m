% �����ʼ��Ӧ��
function F =fun1(X,train_x,train_yy,type,kernel,proprecess,miny,maxy,train_y)
gam=X(:,1);
sig2=X(:,2);
model=initlssvm(train_x,train_yy,type,gam,sig2,kernel,proprecess);
model=trainlssvm(model);
%���ѵ�����Ͳ��Լ���Ԥ��ֵ
[train_predict_y,zt,model]=simlssvm(model,train_x);
% [test_predict_y,zt,model]=simlssvm(model,test_x);
%Ԥ�����ݷ���һ��
train_predict=postmnmx(train_predict_y ,miny,maxy);%Ԥ�����
%  test_predict=postmnmx(test_predict_y ,miny,maxy); %���Լ�Ԥ��ֵ
%���������

trainmse=sum((train_predict-train_y).^2)/length(train_y);
% testmse=sum((test_predict-test_y).^2)/length(test_y); 
F=trainmse; %�Բ��Լ���Ԥ��ֵ����ľ�����Ϊ��Ӧ��ֵ
end
