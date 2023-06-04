
function [Best_FF,Best_P,Conv_curve,accuracy_curve]=BAOA(N,M_Iter,Dim)
global train;global trainlabel;global test;global testlabel;
k =5;
UB = 1; LB=0;
UB_Contiuous = 100; LB_Contiuous = -100;
disp('BAOA Working');
%Two variables to keep the positions and the fitness value of the best-obtained solution

Best_P=zeros(1,Dim);
Best_FF=inf;
Conv_curve=zeros(1,M_Iter);
accuracy_FF=inf;
accuracy_curve=zeros(2,M_Iter);

%Initialize the positions of solution
X=initialization_binary(N,Dim);
Xnew=X;
Ffun=zeros(1,size(X,1));% (fitness values)
Ffun_new=zeros(1,size(Xnew,1));% (fitness values)

MOP_Max=1;
MOP_Min=0.2;
C_Iter=1;
Alpha=5;
Mu=0.499;

Accuracy = zeros(2,2);
accuracy_all = zeros(2,M_Iter);

for i=1:size(X,1)
    for tt = 1:2
        train_knn = train(:,X(i,:)==1,tt);
        test_knn = test(:,X(i,:)==1,tt);
        trainlabel_knn = trainlabel(:,1,tt);
        testlabel_knn = testlabel(:,1,tt);
        testpredict = alg_KNN(train_knn,trainlabel_knn,test_knn,k);
        Accuracy(1,tt) = alg_Accuracy(testpredict,testlabel_knn);
        Accuracy(2,tt) = size(train_knn,2)/Dim;
    end
    Ffun(1,i) = 0.99*(1-mean(Accuracy(1,:)))+ 0.01*(mean(Accuracy(2,:)));
    accuracy_all(1,i) = 1-mean(Accuracy(1,:));
    accuracy_all(2,i) = mean(Accuracy(2,:));
    
    if Ffun(1,i)<Best_FF
        Best_FF=Ffun(1,i);
        Best_P=X(i,:);
        accuracy_FF=accuracy_all(:,i);
    end
end

while C_Iter<M_Iter+1  %Main loop
    MOP=1-((C_Iter)^(1/Alpha)/(M_Iter)^(1/Alpha));   % Probability Ratio
    MOA=MOP_Min+C_Iter*((MOP_Max-MOP_Min)/M_Iter); %Accelerated function
    
    %Update the Position of solutions
    for i=1:size(X,1)   % if each of the UB and LB has a just value
        for j=1:size(X,2)
            r1=rand();
            if r1>MOA
                r2=rand();
                if r2>0.5
                    flag = transfer(((UB_Contiuous-LB_Contiuous)*Mu+LB_Contiuous)/(MOP+eps));
                else
                    flag = transfer(-((UB_Contiuous-LB_Contiuous)*Mu+LB_Contiuous)/(MOP+eps));
                end
                
                if flag ==0
                    Xnew(i,j)=Best_P(1,j)*~X(i,j);
                else
                    Xnew(i,j)=Best_P(1,j)*X(i,j);
                end
            else
                r3=rand();
                if r3>0.5
                    flag = transfer(-MOP*((UB_Contiuous-LB_Contiuous)*Mu+LB_Contiuous));                  
                else
                    flag = transfer(MOP*((UB_Contiuous-LB_Contiuous)*Mu+LB_Contiuous));
                end
                
                if flag ==0
                    Xnew(i,j)=Best_P(1,j)+~X(i,j);
                else
                     Xnew(i,j)=Best_P(1,j)+X(i,j);
                end
            end
        end
        
        Flag_UB=Xnew(i,:)>UB; % check if they exceed (up) the boundaries
        Flag_LB=Xnew(i,:)<LB; % check if they exceed (down) the boundaries
        Xnew(i,:)=(Xnew(i,:).*(~(Flag_UB+Flag_LB)))+UB.*Flag_UB+LB.*Flag_LB;
        
        for tt = 1:2
            train_knn = train(:,Xnew(i,:)==1,tt);
            test_knn = test(:,Xnew(i,:)==1,tt);
            trainlabel_knn = trainlabel(:,1,tt);
            testlabel_knn = testlabel(:,1,tt);
            testpredict = alg_KNN(train_knn,trainlabel_knn,test_knn,k);
            Accuracy(1,tt) = alg_Accuracy(testpredict,testlabel_knn);
            Accuracy(2,tt) = size(train_knn,2)/Dim;
        end
        Ffun_new(1,i) = 0.99*(1-mean(Accuracy(1,:)))+ 0.01*(mean(Accuracy(2,:)));
        accuracy_all(1,i) = 1-mean(Accuracy(1,:));
        accuracy_all(2,i) = mean(Accuracy(2,:));
        
        if Ffun_new(1,i)<Ffun(1,i)
            X(i,:)=Xnew(i,:);
            Ffun(1,i)=Ffun_new(1,i);
        end
        if Ffun(1,i)<Best_FF
            Best_FF=Ffun(1,i);
            Best_P=X(i,:);
            accuracy_FF=accuracy_all(:,i);
        end
    end
    
    %Update the convergence curve
    Conv_curve(C_Iter)=Best_FF;
    accuracy_curve(:,C_Iter)=accuracy_FF;
%     Print the best solution details after every 50 iterations
    if mod(C_Iter,10)==0
        display(['At iteration ', num2str(C_Iter), ' the best solution fitness is ', num2str(Best_FF)]);
    end
    
    C_Iter=C_Iter+1;  % incremental iteration
end
end

function bin_x=transfer(x)
s=abs(tanh(x));% V2 transfer function
if s > rand()
    bin_x=0;
else
    bin_x=1;
end
end





