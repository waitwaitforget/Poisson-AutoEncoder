% THIS SCRIPT IMPLEMENT A POSSION AUTOENCODER (A SPECIAL KIND OF AE)
% PAPER REFERENCED TO "Ranzato M A, Szummer M. Semi-supervised learning of compact document representations with deep networks[C]"
%                   //Proceedings of the 25th international conference on Machine learning. ACM, 2008: 792-799.
% CODED BY KAI TIAN
% CONTACT : tiank311@gmail.com
% IF YOU HAVE ANY QUESTION, PLEASE LET ME KNOW.
% FEEL FREE FOR ANY USE OF THIS SCRIPT.

function PAE = PossionAE(X,nHid,beta)
%% All vector should be column-wise
% HYPERPARAMETERS
PAE.learningrate = .001;
PAE.moment = .0;
PAE.maxIter = 200;
PAE.batchSize = 100;                                                        % TO CHECK SGD, SET BATCHSIZE = 1
PAE.beta = beta;
PAE.checkGradient = 0;                                                      % SET TO 1,IF YOU WANT TO CHECKGRADIENT
% TO DO LOG OPERATIONs
X = X + 1;    

[nObj,nVis] = size(X);
numBatches = nObj / PAE.batchSize;
% WEIGHT INITIALIZATION
PAE.We = (rand(nHid,nVis)-0.5) * 2 * sqrt(6/(nVis + nHid));                 % encoder matrix
PAE.bvis = zeros(nHid,1);                                                   % encoder bias
PAE.pWe = zeros(size(PAE.We));
PAE.pbvis = zeros(nHid,1);

PAE.Wd = (rand(nVis,nHid)-0.5) * 2 * sqrt(6/(nVis + nHid));                 % decoder matrix
PAE.bhid = zeros(nVis,1);                                                   % DECODER BIAS
PAE.pWd = zeros(size(PAE.Wd));
PAE.pbhid = zeros(nVis,1);

errorLog = zeros(1,numBatches*PAE.maxIter);
loss = zeros(1,PAE.maxIter);
%% FORWORD PASS
for iter = 1:PAE.maxIter
    kk = randperm(nObj);                                                    % SHUFFLE THE DATA
    trainingdata = X(kk,:);
    cprintf('Keywords','#Epoch %d: ',iter);
    for j=1:numBatches
        batchData = trainingdata((j-1)*PAE.batchSize+1:j*PAE.batchSize,:);  % BATCHED DATA
        
        PAE.z = bsxfun(@plus,PAE.We * log(batchData)',PAE.bvis);            % CALCULATE Z
        PAE.aHid = sigmoid(PAE.z);                                          % CALCULATE ACTIVATIONS

        PAE.z2 = bsxfun(@plus,PAE.Wd*PAE.aHid,PAE.bhid);
     
        Lambda = beta * exp(PAE.z2);
        
        lastterm = logadd(batchData');

        E = Lambda - batchData'.*log(Lambda)+ lastterm;     
    
        PAE.L = sum(sum((E)))/PAE.batchSize;                                % loss function
        errorLog((iter-1)*numBatches+j) = PAE.L;                            % recorded every epoch
        
%% BACKWORD PASS (compute the gradients)
        PAE.dz2 = (Lambda - batchData')/PAE.batchSize;
        
        PAE.dWd = ((Lambda - batchData')*PAE.aHid')/PAE.batchSize;                        % dE/dWd = dE/dz2*z'
        PAE.dbhid = sum(PAE.dz2,2)/PAE.batchSize ;

        PAE.daHid = PAE.Wd' * PAE.dz2  ;
        PAE.dz =  PAE.daHid .* invsigmoid(PAE.z);
        PAE.dWe = PAE.dz * log(batchData);
        PAE.dbvis = sum(PAE.dz,2);
        % CHECK GRADIENT
        if(PAE.checkGradient == 1)
            PAEchecknumgrad(PAE,batchData,[]);
        end
%% UPDATE GRADIENTS
        PAE.pWe = PAE.moment * PAE.pWe + PAE.learningrate * (PAE.dWe);
        PAE.pbvis  = PAE.moment * PAE.pbvis + PAE.learningrate * (PAE.dbvis);
        
        PAE.We = PAE.We - PAE.pWe;
        PAE.bvis = PAE.bvis - PAE.pbvis;
       
        PAE.pWd = PAE.moment * PAE.pWd + PAE.learningrate * (PAE.dWd);
        PAE.pbhid  = PAE.moment * PAE.pbhid + PAE.learningrate * (PAE.dbhid);
        PAE.Wd = PAE.Wd - PAE.pWd;
        PAE.bhid = PAE.bhid - PAE.pbhid;
        
       
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
function y = logadd(x)
y = zeros(size(x));
    for e = 1:numel(x)
        for i=2:x(e)
           y(e) = y(e) + log(i);
        end
    end
end