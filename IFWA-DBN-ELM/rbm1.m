
% This program trains Restricted Boltzmann Machine in which
% visible, binary, stochastic pixels are connected to
% hidden, binary, stochastic feature detectors using symmetrically
% weighted connections. Learning is done with 1-step Contrastive Divergence.   
% The program assumes that the following variables are set externally:
% maxepoch  -- maximum number of epochs
% numhid    -- number of hidden units 
% batchdata -- the data that is divided into batches (numcases numdims numbatches)
% restart   -- set to 1 if learning starts from beginning 

epsilonw      = 0.01;   % Learning rate for weights Ȩֵѧϰ��
epsilonvb     = 0.01;   % Learning rate for biases of visible units���ӽڵ��ƫ��ѧϰ�� 
epsilonhb     = 0.01;   % Learning rate for biases of hidden units �����ڵ��ƫ��ѧϰ��
weightcost  = 0.0008;   %Ȩ��˥��ϵ��
initialmomentum  = 0.5; %��ʼ������
finalmomentum    = 0.9; %ȷ��������

[numcases,numdims ,numbatches]=size(batchdata);

if restart ==1
  restart=0;
  epoch=1;
%   numhid=round(numhid);
% Initializing symmetric weights and biases. 
  vishid     = 0.1*(randn(numdims,numhid));%���ӽڵ㵽�����ڵ�֮���Ȩֵ��ʼ��
  hidbiases  = zeros(1,numhid);%�����ڵ�ĳ�ʼ��Ϊ0
  visbiases  = zeros(1,numdims);%���ӽڵ�ƫ�ó�ʼ��Ϊ0

  poshidprobs = zeros(numcases,numhid);%��ʼ��������������򴫲�ʱ��������������
  neghidprobs = zeros(numcases,numhid);
  posprods    = zeros(numdims,numhid);
  negprods    = zeros(numdims,numhid);
  vishidinc  = zeros(numdims,numhid);
  hidbiasinc = zeros(1,numhid);
  visbiasinc = zeros(1,numdims);
                                       %�������ݼ����򴫲�ʱ��������������
  batchposhidprobs=zeros(numcases,numhid,numbatches);
end

for epoch = epoch:maxepoch%���е�������
 fprintf(1,'epoch %d\r',epoch); 
 errsum=0;
 for batch = 1:numbatches%ÿ�ε������������е����ݿ�
 %fprintf(1,'epoch %d batch %d\r',epoch,batch); 

%%%%%%%%% ��ʼ����׶εļ���%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  data = batchdata(:,:,batch);      %ÿ�ε���ѡ��һ�����������ݣ���ÿһ�д���һ������ֵ
                                    %��������ݲ��Ƕ�ֵ�ģ��ϸ���˵��Ӧ�ý�����ж�ֵ��
  poshidprobs = 1./(1 + exp(-data*vishid - repmat(hidbiases,numcases,1)));   
                                     %����������ڵ��������ʣ����õ���sigmoid����
  %%%��������׶εĲ���ͳ����%%%%%%%%%%%%%%%%%%%%
  batchposhidprobs(:,:,batch)=poshidprobs;
  posprods    = data' * poshidprobs;%�ÿ��ӽڵ�������������ڵ������ĳ˻���������ɢ��ͳ����
  poshidact   = sum(poshidprobs);   %�������ֵ������ͣ����ڼ��������ڵ��ƫ��
  posvisact = sum(data);             %�����ݽ�����ͣ����ڼ�����ӽڵ��ƫ�ã���������������ĸ���Ϊ1ʱ��
                                      % ��õ�ƫ�������е��ֻ�����ͬ����ʱ��Ӱ��Ԥѵ���Ľ��

%%%%%%%%% ����׶ν���  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  poshidstates = poshidprobs > rand(numcases,numhid);
  %��������ĸ��ʼ���ֵposhidprobs����0.1��ֵ�������ո���ֵ��С���ж���rand(m,n)����
  %m*n��С�ľ��󣬽�poshidprobs�е�ֵ��rand�����ıȽϣ����������������Ӧλ�õ�ֵ��������Ӧλ��Ϊ1������Ϊ0
    
%%%%%%%%% ��ʼ����׶εļ���  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  negdata = 1./(1 + exp(-poshidstates*vishid' - repmat(visbiases,numcases,1)));
  %����׶μ�����ӽڵ��ֵ
  neghidprobs = 1./(1 + exp(-negdata*vishid - repmat(hidbiases,numcases,1)));    
  %����������ڵ�ĸ���ֵ
  negprods  = negdata'*neghidprobs;
  %���㷴��ɢ��ͳ����
  neghidact = sum(neghidprobs);
  negvisact = sum(negdata); 

%%%%%%%%% ����׶ν��� %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  err= sum(sum( (data-negdata).^2 ));%����ѵ������ԭʼ���ݺ��ع�����֮����ع����
  errsum = err + errsum;

   if epoch>5
     momentum=finalmomentum;         %�ڵ������²��������У�ǰ4��ʹ�ó�ʼ�����֮��ʹ��ȷ��������
   else
     momentum=initialmomentum;
   end

%%%%%%%%%һ�´������ڸ���Ȩֵ��ƫ��%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    vishidinc = momentum*vishidinc + ...%Ȩֵ����ʱ������
                epsilonw*( (posprods-negprods)/numcases - weightcost*vishid);
    visbiasinc = momentum*visbiasinc + (epsilonvb/numcases)*(posvisact-negvisact);
    %���Ӳ�ƫ�ø���ʱ������
    hidbiasinc = momentum*hidbiasinc + (epsilonhb/numcases)*(poshidact-neghidact);
    %������ƫ�ø���ʱ������
    vishid = vishid + vishidinc;%����Ȩֵ
    visbiases = visbiases + visbiasinc;%���¿��Ӳ�ƫ��
    hidbiases = hidbiases + hidbiasinc;%����������ƫ��

%%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

  end
  fprintf(1, 'epoch %4i error %6.1f  \n', epoch, errsum); 
end
%%%ÿ�ε�����������ʾѵ�������ع����