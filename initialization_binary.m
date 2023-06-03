
function X=initialization_binary(N,Dim)
X = zeros(N,Dim);
X(rand(N,Dim)>0.5)=1; 
end