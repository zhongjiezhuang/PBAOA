clc;clear;
dataset_num = 10;
times_count = 30; 
result_all = zeros(times_count,dataset_num*3); % 1,2,3 cloumns is the error,num,fitness of the first dataset
result_all2 = zeros(times_count,dataset_num*3); % 1,2,3 cloumns is the error,num,fitness of the first dataset
result_conclude = zeros(dataset_num*2,4); %The first row is the mean value of the first dataset, the second row is the std of the first dataset
result_conclude2 = zeros(dataset_num*2,4);%The first row is the mean value of the first dataset, the second row is the std of the first dataset

datanames = {'arrhythmia' 'gastroenterology' 'LSVT_voice_rehabilitation' 'PersonGait' 'SCADI' 'Urban_land_cover' 'ORL' 'warpAR10P' 'warpPIE10P' 'Yale' 'CNAE_9' 'COIL20' 'lung' 'TOX_171'};
for id = 1:14
    data= load_data( datanames(id) );
    N = 2; % N-fold cross-validation
    [~, dim] = size(data);
    Solution_no=8; %Number of search solutions
    M_Iter=100;    %Maximum number of iterations
    tic
    for times_id = 1 : times_count 
        [iristrain,iristest] = alg_CrossValidation_Fast(data,N,1);
         global train;global trainlabel;global test;global testlabel;
        train = iristrain(:,1:dim-1,:);
        trainlabel = iristrain(:,dim,:);
        test = iristest(:,1:dim-1,:);
        testlabel = iristest(:,dim,:);  
        
        disp(['id：',num2str(id),'  times：', num2str(times_id)]);      
        [Best_FF,Best_P,Conv_curve,accuracy_curve]=BAOA(Solution_no,M_Iter,dim-1); 
        [Best_FF2,Best_P2,Conv_curve2,accuracy_curve2,iter2]=PBAOA(Solution_no,M_Iter,dim-1,4,5); 
       
        result_all(times_id,(3*id-2):(3*id-1)) = accuracy_curve(:,M_Iter);
        result_all(times_id,3*id) = Conv_curve(M_Iter);
        disp(['x[',num2str(Best_P),']=',num2str(result_all(times_id,3*id))]);   
        
        result_all2(times_id,(3*id-2):(3*id-1)) = accuracy_curve2(:,iter2-1);
        result_all2(times_id,3*id) = Conv_curve2(iter2-1);
        disp(['x[',num2str(Best_P2),']=',num2str(result_all2(times_id,3*id))]);
    end
    result_mean = mean(result_all,1);    
    result_std = std(result_all,1); 
    result_time = toc/times_count; 
    result_conclude(2*id-1,1:3) = result_mean(1,(3*id-2):3*id); %error,number of selected features , fitness value
    result_conclude(2*id,1:3) = result_std(1,(3*id-2):3*id);
    result_conclude(2*id-1,4) = result_time;
    result_conclude(2*id,4) = result_time; 
    
    result_mean = mean(result_all2,1);    
    result_std = std(result_all2,1); 
    result_time = toc/times_count; 
    result_conclude2(2*id-1,1:3) = result_mean(1,(3*id-2):3*id); %error,number of selected features , fitness value
    result_conclude2(2*id,1:3) = result_std(1,(3*id-2):3*id);
    result_conclude2(2*id-1,4) = result_time;
    result_conclude2(2*id,4) = result_time;
end

