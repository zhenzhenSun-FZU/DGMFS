% This is an example file on how the DGMFS [1] program could be used.

% [1] Z. Sun, H. Xie, J. Liu, et al
% Dual-Graph with Non-Convex Sparse Regularization for Multi-Label Feature Selection, Applied Intelligence, 2023.

clc; clear; 
addpath(genpath('.\'))
DataName={'education'};
for i=1:size(DataName,1)
    dataset = DataName{i};
    load(dataset);

    % Calucate some statitics about the data
    [num_train, num_label] = size(Y_train); [num_test, num_feature] = size(X_test);
    pca_remained = round(num_feature*0.95);

    % Performing PCA
    all = [X_train; X_test]; 
    ave = mean(all);
    all = (all'-concur(ave', num_train + num_test))';
    covar = cov(all); covar = full(covar);
    [u,s,v] = svd(covar);
    t_matrix = u(:, 1:pca_remained)';   
    all = (t_matrix * all')';
    X_train = all(1:num_train,:); X_test = all((num_train + 1):(num_train + num_test),:);
    
    if num_feature <= 100
        FeaNumCandi = 5:5:50;
    else
        FeaNumCandi = 10:10:100;
    end
    
    para.sigma = 1;
    para.alpha =10;para.beta = 1; para.lambda = 0.01;
    
   % Running the DGMFS procedure for feature selection
                
    t0 = clock;
    [ W, obj ] = DGMFS( X_train', Y_train', para );
    time = etime(clock, t0);

    [dumb,idx] = sort(sum(W.*W,2),'descend'); 
    feature_idx = idx(1:num_feature);

    % The default setting of MLKNN
      Num = 10;Smooth = 1;  
                 
     HL=[]; RL=[];CV=[];AP=[];MI=[];MA=[];

     % Train and test
     % If you use MLKNN as the classifier, please cite the literature [2]
     % [2] M.-L. Zhang, Z.-H. Zhou:
     % ML-KNN: A lazy learning approach to multi-label learning. Pattern Recognition 2007, 40(7): 2038-2048.
      for feaIdx = 1:length(FeaNumCandi)
             feaNum = FeaNumCandi(feaIdx);
             fprintf('Running the program with the selected features - %d/ \n',feaNum);

             f=feature_idx(1:feaNum);
             [Prior,PriorN,Cond,CondN]=MLKNN_train(X_train(:,f),Y_train',Num,Smooth);
             [HammingLoss,RankingLoss,Coverage,Average_Precision,macrof1,microf1,Outputs,Pre_Labels]=...
                  MLKNN_test(X_train(:,f),Y_train',X_test(:,f),Y_test',Num,Prior,PriorN,Cond,CondN);

              HL(feaIdx)=HammingLoss;
              RL(feaIdx)=RankingLoss;
              CV(feaIdx)=Coverage;
              AP(feaIdx)=Average_Precision;
              MA(feaIdx)=macrof1;
              MI(feaIdx)=microf1;
       end
       result_path = strcat(dataset,'\','alpha_',num2str(alpha),'_beta_',num2str(beta),'_lambda_',num2str(lambda),'_result.mat');
       save(result_path,'HL','RL','CV','AP','MA','MI','time');        
end

