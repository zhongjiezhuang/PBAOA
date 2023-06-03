function [train,test] = alg_CrossValidation_Fast(data,K,ratio)
%alg_CrossValidation generates K-fold cross-validation data
%   [TRAIN,TEST] = alg_CrossValidation(DATA,K)
%   returns matrices with K pages for training and testing in a K-fold
%   cross-validation.
%   
%   The row number of DATA must be divisible by K.
%   This algorithm extracts samples from DATA every K rows.
%   You can use one page of TRAIN/TEST for each validation.
%   Rows represent samples.
%   Columns represent the dimensions of samples.
%   November 2, 2016, by HanzheTeng

% data_copy = data;
[row,col] = size(data);
Train_label = data(:,1);
U  = unique(Train_label);
nclasses = length(U);%number of classes
data_d = [];
for k =1:nclasses
    index = find(Train_label==U(k));
    length_index = length(index);
    temp1 = randperm(length_index);
    temp2 = temp1(1:floor(length_index*ratio));
    data_d = [data_d; data(index(temp2),:)];
end
data = data_d;

[row,col] = size(data);
testrow = row/K;
testrow = floor(testrow);
% if testrow~=fix(testrow)
%     error('The row number of DATA must be divisible by K.');
% end
test = zeros(testrow,col,K);
train = zeros(row-testrow,col,K);

for i=1:K
    flag = ones(row,1); % flag matrix
    trainrow = 1; 
    for j=1:testrow
        test(j,:,i) = data(K*j-K+i,:);
        flag(K*j-K+i,1) = 0;
    end
    for j=1:row
        if(flag(j)==1)
            train(trainrow,:,i) = data(j,:);
            trainrow = trainrow+1;
        end
    end
end

end % End of function alg_CrossValidation
