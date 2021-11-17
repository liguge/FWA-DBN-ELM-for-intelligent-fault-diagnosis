
% This program trains Restricted Boltzmann Machine in which
% visible, binary, stochastic pixels are connected to
% hidden, binary, stochastic feature detectors using symmetrically
% weighted connections. Learning is done with 1-step Contrastive Divergence.   
% The program assumes that the following variables are set externally:
% maxepoch  -- maximum number of epochs
% numhid    -- number of hidden units 
% batchdata -- the data that is divided into batches (numcases numdims numbatches)
% restart   -- set to 1 if learning starts from beginning 

epsilonw      = 0.01;   % Learning rate for weights 权值学习率
epsilonvb     = 0.01;   % Learning rate for biases of visible units可视节点的偏置学习率 
epsilonhb     = 0.01;   % Learning rate for biases of hidden units 隐含节点的偏置学习率
weightcost  = 0.0008;   %权重衰减系数
initialmomentum  = 0.5; %初始动量项
finalmomentum    = 0.9; %确定动量项

[numcases,numdims ,numbatches]=size(batchdata);

if restart ==1
  restart=0;
  epoch=1;
%   numhid=round(numhid);
% Initializing symmetric weights and biases. 
  vishid     = 0.1*(randn(numdims,numhid));%可视节点到隐含节点之间的权值初始化
  hidbiases  = zeros(1,numhid);%隐含节点的初始化为0
  visbiases  = zeros(1,numdims);%可视节点偏置初始化为0

  poshidprobs = zeros(numcases,numhid);%初始化单个迷你块正向传播时隐含层的输出概率
  neghidprobs = zeros(numcases,numhid);
  posprods    = zeros(numdims,numhid);
  negprods    = zeros(numdims,numhid);
  vishidinc  = zeros(numdims,numhid);
  hidbiasinc = zeros(1,numhid);
  visbiasinc = zeros(1,numdims);
                                       %整个数据集正向传播时隐含层的输出概率
  batchposhidprobs=zeros(numcases,numhid,numbatches);
end

for epoch = epoch:maxepoch%所有迭代次数
 fprintf(1,'epoch %d\r',epoch); 
 errsum=0;
 for batch = 1:numbatches%每次迭代都遍历所有的数据块
 %fprintf(1,'epoch %d batch %d\r',epoch,batch); 

%%%%%%%%% 开始正向阶段的计算%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  data = batchdata(:,:,batch);      %每次迭代选择一个迷你块的数据，，每一行代表一个样本值
                                    %这里的数据并非二值的，严格来说，应该将其进行二值化
  poshidprobs = 1./(1 + exp(-data*vishid - repmat(hidbiases,numcases,1)));   
                                     %计算隐含层节点的输出概率，所用的是sigmoid函数
  %%%计算正向阶段的参数统计量%%%%%%%%%%%%%%%%%%%%
  batchposhidprobs(:,:,batch)=poshidprobs;
  posprods    = data' * poshidprobs;%用可视节点向量和隐含层节点向量的乘积计算正向散度统计量
  poshidact   = sum(poshidprobs);   %针对样本值进行求和，用于计算隐含节点的偏置
  posvisact = sum(data);             %对数据进行求和，用于计算可视节点的偏置，当迷你块中样本的个数为1时，
                                      % 求得的偏置向量中的又换宿相同，此时会影响预训练的结果

%%%%%%%%% 正向阶段结束  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  poshidstates = poshidprobs > rand(numcases,numhid);
  %将隐含层的概率激活值poshidprobs进行0.1二值化，按照概率值大小来判定。rand(m,n)产生
  %m*n大小的矩阵，将poshidprobs中的值和rand产生的比较，如果大于随机矩阵对应位置的值，则将其相应位置为1，否则为0
    
%%%%%%%%% 开始反向阶段的计算  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  negdata = 1./(1 + exp(-poshidstates*vishid' - repmat(visbiases,numcases,1)));
  %反向阶段计算可视节点的值
  neghidprobs = 1./(1 + exp(-negdata*vishid - repmat(hidbiases,numcases,1)));    
  %计算隐含层节点的概率值
  negprods  = negdata'*neghidprobs;
  %计算反向散度统计量
  neghidact = sum(neghidprobs);
  negvisact = sum(negdata); 

%%%%%%%%% 反向阶段结束 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  err= sum(sum( (data-negdata).^2 ));%计算训练集中原始数据和重构数据之间的重构误差
  errsum = err + errsum;

   if epoch>5
     momentum=finalmomentum;         %在迭代更新参数过程中，前4次使用初始动量项，之后使用确定动量项
   else
     momentum=initialmomentum;
   end

%%%%%%%%%一下代码用于更新权值和偏置%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    vishidinc = momentum*vishidinc + ...%权值更新时的增量
                epsilonw*( (posprods-negprods)/numcases - weightcost*vishid);
    visbiasinc = momentum*visbiasinc + (epsilonvb/numcases)*(posvisact-negvisact);
    %可视层偏置更新时的增量
    hidbiasinc = momentum*hidbiasinc + (epsilonhb/numcases)*(poshidact-neghidact);
    %隐含层偏置更新时的增量
    vishid = vishid + vishidinc;%更新权值
    visbiases = visbiases + visbiasinc;%更新可视层偏置
    hidbiases = hidbiases + hidbiasinc;%更新隐含层偏置

%%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

  end
  fprintf(1, 'epoch %4i error %6.1f  \n', epoch, errsum); 
end
%%%每次迭代结束后，显示训练集的重构误差