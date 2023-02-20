function [W, obj] = DGMFS(X_train, Y_train, para )

% Calucate some statitics about the data
Y_train(find(Y_train==-1))=0;
[num_feature, num_train] = size(X_train); num_label = size(Y_train, 1);

H=eye(num_train)-ones(num_train,num_train)./num_train;

% calculate graph Laplacian
options1 = [];
options1.NeighborMode = 'KNN';
options1.k = 100;
options1.WeightMode = 'HeatKernel';
options1.t = 1;
%Ls
S = constructW(X_train',options1); 
nRowS = size(S,1);
for i = 1:nRowS
    sum_row = sum(S(i,:));
    S(i,:) = S(i,:)/(sum_row+eps);
end
diag_ele_arr = sum(S+S',2);
D = diag(diag_ele_arr);
Ls = D-S-S';

%Lf
S=[]; D=[];
S = constructW(X_train,options1); 
nRowS = size(S,1);
for i = 1:nRowS
    sum_row = sum(S(i,:));
    S(i,:) = S(i,:)/(sum_row+eps);
end
diag_ele_arr = sum(S+S',2);
D = diag(diag_ele_arr);
Lf = D-S-S';

%Initialize F
eY = eigY(Ls,num_label);
label = litekmeans(eY,num_label,'Replicates',20);
F = zeros(num_label, num_train);
for i = 1:num_train
    F(label(i), i) = 1;
end

% Initialize W
W = zeros(num_feature, num_label); 

iter = 1;
Maxiter = 100;

obj = [];
obj1 = norm((W'*X_train-F)*H,'fro')^2 + norm(F-Y_train,'fro')^2 + para.alpha*trace(F*Ls*F')...
    +para.beta*trace(W'*Lf*W) + para.lambda*(sum(sqrt(sum(W.*W,2)))-sqrt(sum(sum(W.*W,2))));
obj = [obj,obj1];
while iter <= Maxiter
    %Update W
    X1 = X_train * H; F1 = F * H;
    u = 1./sqrt(sum(W.*W, 2) + eps);    
    U = diag(u);
    A = X1*F1';
    B = X1*X1' + para.beta*Lf + para.lambda*U-0.5*para.lambda*(1/(sqrt(trace(W'*W))+eps));
    W = B\A;
    
    %Update F--------------------------------------------------------------
    P = H + eye(num_train) + para.alpha*Ls;
    Q = W'*X_train*H + Y_train;
    F = F.* Q ./ (F*P+eps);
    
    obj1 = norm((W'*X_train-F)*H,'fro')^2 + norm(F-Y_train,'fro')^2 + para.alpha*trace(F*Ls*F')...
    +para.beta*trace(W'*Lf*W) + para.lambda*(sum(sqrt(sum(W.*W,2)))-sqrt(sum(sum(W.*W,2))));

    cver = abs((obj(end) - obj1)/obj1);
    obj = [obj,obj1];
    iter = iter + 1;
    if (cver < 10^-5 && iter > 5) 
        break
    end
    
end

end


