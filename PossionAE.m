function PAE = PossionAE(X,nHid,beta)
%% All vector should be column-wise
% HYPERPARAMETERS
PAE.learningrate = 0.001;
PAE.moment = 0.1;
PAE.maxIter = 200;
PAE.batchSize = 100;                                                          % TO CHECK SGD, SET BATCHSIZE = 1
PAE.beta = beta;
PAE.checkGradient = 0;
% TO DO LOG OPERATION
X = X+1;    

[nObj,nVis] = size(X);
numBatches = nObj / PAE.batchSize;
% WEIGHT INITIALIZATION
PAE.We = (rand(nHid,nVis)-0.5) * 2 * sqrt(6/(nVis + nHid));                 % encoder matrix
PAE.bvis = zeros(nHid,1);                                                   % encoder bias

PAE.Wd = (rand(nVis,nHid)-0.5) * 2 * sqrt(6/(nVis + nHid));                 % decoder matrix
PAE.bhid = zeros(nVis,1);                                                   % DECODER BIAS

errorLog = zeros(1,numBatches*PAE.maxIter);
loss = zeros(1,PAE.maxIter);
%% FORWORD PASS
for iter = 1:PAE.maxIter
    kk = randperm(nObj);                                                    % SHUFFLE THE DATA
    trainingdata = X(kk,:);
    fprintf('#Epoch %d: ',iter);
    for j=1:numBatches
        batchData = trainingdata((j-1)*PAE.batchSize+1:j*PAE.batchSize,:);  % BATCHED DATA
        
        PAE.z = bsxfun(@plus,PAE.We * log(batchData)',PAE.bvis);            % CALCULATE Z
        PAE.aHid = sigmoid(PAE.z);                                          % CALCULATE ACTIVATIONS

        %z2 = bsxfun(@plus,PAE.Wd*PAE.aHid,PAE.bhid);
        z2 = bsxfun(@plus,PAE.Wd*PAE.z,PAE.bhid);
        Lambda = beta * exp(z2);
        % max(max(Lambda))
        %reconstruction error
        %Fac = factorial(batchData');
        lastterm = batchData'.*(batchData-1)'/2;
        E = Lambda - batchData'.*z2+ lastterm + log(PAE.beta) * batchData';

        PAE.L = sum(sum((E)))/PAE.batchSize; %loss function
        errorLog((iter-1)*numBatches+j) = PAE.L;                            % recorded every epoch
        
%% BACKWORD PASS (compute the gradients)
%       dLambda = PAE.beta*exp(PAE.Wd.*repmat(PAE.z,1,nVis));
        PAE.dz2 = Lambda - batchData';
        
        PAE.dWd = ((Lambda - batchData')*PAE.aHid');                        % dE/dWd = dE/dz2*z'
        PAE.dbhid = sum(PAE.dz2,2);

        PAE.daHid = PAE.Wd' * PAE.dz2;
        PAE.dz =  PAE.daHid .* invsigmoid(PAE.z);
        PAE.dWe = PAE.dz * log(batchData);
        PAE.dbvis = sum(PAE.dz,2);

%% UPDATE GRADIENTS
        PAE.We = PAE.We - PAE.learningrate*(PAE.dWe/PAE.batchSize+ PAE.moment*PAE.We);
        PAE.bvis = PAE.bvis - PAE.learningrate*(PAE.dbvis/PAE.batchSize);
        PAE.Wd = PAE.Wd - PAE.learningrate*(PAE.dWd/PAE.batchSize+ PAE.moment*PAE.Wd);
        PAE.bhid = PAE.bhid - PAE.learningrate*(PAE.dbhid/PAE.batchSize);
        
        % CHECK GRADIENT
        if(PAE.checkGradient == 1)
%            PAEchecknumgrad(PAE,batchData,[]);
        end
    end
    loss(iter) = mean(errorLog((iter-1)*numBatches+1:iter*numBatches));
    fprintf('neg-likelihood : %f\n',loss(iter));
end
disp('Training process done...');
figure;plot(loss);
end
%% INVERSE OF SIGMOID FUNCTION
function res = invsigmoid(z)
    res = sigmoid(z).*(1-sigmoid(z));
end