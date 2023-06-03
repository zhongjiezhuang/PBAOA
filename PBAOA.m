function [Best_FF_All,Best_P_All,Conv_curve,accuracy_curve,C_Iter]=PBAOA(N,M_Iter,Dim,groups,T)
global train;global trainlabel;global test;global testlabel;
iter_all = N*M_Iter;
iter=0;
N = N/groups;

k =5;
UB = 1; LB=0;
UB_Contiuous = 100; LB_Contiuous = -100;
disp('PBAOA Working');
%Two variables to keep the positions and the fitness value of the best-obtained solution

Best_P_All=zeros(1,Dim);
Best_FF_All=inf;
accuracy_FF_All=zeros(2,1);
Conv_curve=zeros(1,M_Iter);
accuracy_curve=zeros(2,M_Iter);
%Initialize the positions of solution
for g = 1:groups
    group(g).Best_P=zeros(1,Dim);
    group(g).Best_FF=inf;
    group(g).accuracy_FF=inf;
    group(g).X=initialization_binary(N,Dim);
    group(g).Xnew=group(g).X;
    group(g).Ffun=zeros(1,size(group(g).X,1));% (fitness values)
    group(g).Ffun_new=zeros(1,size(group(g).Xnew,1));% (fitness values)
    
    group(g).Commu_P=zeros(1,Dim);
end

MOP_Max=1;
MOP_Min=0.2;
C_Iter=1;
Alpha=5;
Mu=0.499;

for g = 1:groups
    for i=1:size(group(g).X,1)
        for tt = 1:2
            train_knn = train(:,group(g).X(i,:)==1,tt);
            test_knn = test(:,group(g).X(i,:)==1,tt);
            trainlabel_knn = trainlabel(:,1,tt);
            testlabel_knn = testlabel(:,1,tt);
            testpredict = alg_KNN(train_knn,trainlabel_knn,test_knn,k);
            group(g).Accuracy(1,tt) = alg_Accuracy(testpredict,testlabel_knn);
            group(g).Accuracy(2,tt) = size(train_knn,2)/Dim;
        end
        group(g).Ffun(1,i) = 0.99*(1-mean(group(g).Accuracy(1,:)))+ 0.01*(mean(group(g).Accuracy(2,:)));
        group(g).accuracy_all(1,i) = 1-mean(group(g).Accuracy(1,:));
        group(g).accuracy_all(2,i) = mean(group(g).Accuracy(2,:));
        iter=iter+1;
        
        if group(g).Ffun(1,i)<group(g).Best_FF
            group(g).Best_FF=group(g).Ffun(1,i);
            group(g).Best_P=group(g).X(i,:);
            group(g).accuracy_FF=group(g).accuracy_all(:,i);
        end
    end
end

while C_Iter<M_Iter+1 && iter< iter_all+1 %Main loop
    if rem(C_Iter,T) == 0
        for g = 1:groups
            [group(g).Ffun, I] = sort(group(g).Ffun);
            group(g).X = group(g).X(I,:);
        end
        
        for g = 1:groups
            for dd = 1:Dim
                if group(g).Ffun(1) == Best_FF_All
                    if rand()<0.5
                        gg = ceil(rand()*4); 
                        if gg ==g
                            gg = mod(gg,4)+1;
                        end
                        group(g).Commu_P(1,dd) = group(gg).X(1,dd);
                    else
                        group(g).Commu_P(1,dd) = Best_P_All(1,dd);
                    end
                else              
                    if rand()<0.5
                        group(g).Commu_P(1,dd) = group(g).X(1,dd);
                    else
                        group(g).Commu_P(1,dd) = Best_P_All(1,dd);
                    end
                end
            end
            
            for tt = 1:2
                train_knn = train(:,group(g).Commu_P(1,:)==1,tt);
                test_knn = test(:,group(g).Commu_P(1,:)==1,tt);
                trainlabel_knn = trainlabel(:,1,tt);
                testlabel_knn = testlabel(:,1,tt);
                testpredict = alg_KNN(train_knn,trainlabel_knn,test_knn,k);
                group(g).Accuracy(1,tt) = alg_Accuracy(testpredict,testlabel_knn);
                group(g).Accuracy(2,tt) = size(train_knn,2)/Dim;
            end
            Commu_fit = 0.99*(1-mean(group(g).Accuracy(1,:)))+ 0.01*(mean(group(g).Accuracy(2,:)));
            iter = iter+1;
            
            if Commu_fit < group(g).Best_FF
                group(g).Best_FF  = Commu_fit;
                group(g).Best_P = group(g).Commu_P;
            end
            if Commu_fit < Best_FF_All
                Best_FF_All  = Commu_fit;
                Best_P_All = group(g).Commu_P;
            end
        end
    end
    
    MOP=1-((C_Iter)^(1/Alpha)/(M_Iter)^(1/Alpha));   % Probability Ratio
    MOA=MOP_Min+C_Iter*((MOP_Max-MOP_Min)/M_Iter); %Accelerated function
    
    for g = 1:groups
        group(g).Xnew_copy = group(g).Xnew;
        %Update the Position of solutions
        for i=1:size(group(g).X,1)   % if each of the UB and LB has a just value
            for j=1:size(group(g).X,2)
                r1=rand();
                if r1>MOA
                    r2=rand();
                    if r2>0.5
                        flag = transfer(((UB_Contiuous-LB_Contiuous)*Mu+LB_Contiuous)/(MOP+eps));
                    else
                        flag = transfer(-((UB_Contiuous-LB_Contiuous)*Mu+LB_Contiuous)/(MOP+eps));
                    end                    
                    if flag ==0                         
                        Xtemp=~group(g).X(i,j);
                    else
                        Xtemp=group(g).X(i,j);
                    end                                   
                    if rand()>MOA
                        group(g).Xnew(i,j)=group(g).Best_P(1,j)*Xtemp;
                    else
                        group(g).Xnew(i,j)=Best_P_All(1,j)*Xtemp;
                    end
                else
                    r3=rand();
                    if r3>0.5
                        flag = transfer(-MOP*((UB_Contiuous-LB_Contiuous)*Mu+LB_Contiuous));
                    else
                        flag = transfer(MOP*((UB_Contiuous-LB_Contiuous)*Mu+LB_Contiuous));
                    end
                    
                    if flag ==0                         
                        Xtemp=~group(g).X(i,j);
                    else
                        Xtemp=group(g).X(i,j);
                    end
                    
                    if rand()>MOA
                        group(g).Xnew(i,j)=group(g).Best_P(1,j)+Xtemp;
                    else
                        group(g).Xnew(i,j)=Best_P_All(1,j)+Xtemp;
                    end
                end
            end
        
            
            Flag_UB=group(g).Xnew(i,:)>UB; % check if they exceed (up) the boundaries
            Flag_LB=group(g).Xnew(i,:)<LB; % check if they exceed (down) the boundaries
            group(g).Xnew(i,:)=(group(g).Xnew(i,:).*(~(Flag_UB+Flag_LB)))+UB.*Flag_UB+LB.*Flag_LB;
            
            for tt = 1:2
                train_knn = train(:,group(g).Xnew(i,:)==1,tt);
                test_knn = test(:,group(g).Xnew(i,:)==1,tt);
                trainlabel_knn = trainlabel(:,1,tt);
                testlabel_knn = testlabel(:,1,tt);
                testpredict = alg_KNN(train_knn,trainlabel_knn,test_knn,k);
                group(g).Accuracy(1,tt) = alg_Accuracy(testpredict,testlabel_knn);
                group(g).Accuracy(2,tt) = size(train_knn,2)/Dim;
            end
            group(g).Ffun_new(1,i) = 0.99*(1-mean(group(g).Accuracy(1,:)))+ 0.01*(mean(group(g).Accuracy(2,:)));
            group(g).accuracy_all(1,i) = 1-mean(group(g).Accuracy(1,:));
            group(g).accuracy_all(2,i) = mean(group(g).Accuracy(2,:));
            iter = iter+1;
            
            if group(g).Ffun_new(1,i)<group(g).Ffun(1,i)
                group(g).X(i,:)=group(g).Xnew(i,:);
                group(g).Ffun(1,i)=group(g).Ffun_new(1,i);
            end
            if group(g).Ffun(1,i)<group(g).Best_FF
                group(g).Best_FF=group(g).Ffun(1,i);
                group(g).Best_P=group(g).X(i,:);
                group(g).accuracy_FF=group(g).accuracy_all(:,i);
            end
        end
        
        if group(g).Best_FF<Best_FF_All
            Best_FF_All = group(g).Best_FF;
            Best_P_All = group(g).Best_P;
            accuracy_FF_All=group(g).accuracy_FF;
        end
        Conv_curve(C_Iter)=Best_FF_All;
        accuracy_curve(:,C_Iter)=accuracy_FF_All;
    end
    
    % Print the best solution details after every 50 iterations
    if mod(C_Iter,10)==0
        display(['At iteration ', num2str(C_Iter), ' the best solution fitness is ', num2str(Best_FF_All)]);
    end
    
    C_Iter=C_Iter+1;  % incremental iteration
end
end

function bin_x=transfer(x)
s=abs(tanh(x));% V2 transfer function
if s >= rand()
    bin_x=0;
else
    bin_x=1;
end
end





