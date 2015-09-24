function pae = PAEff(pae,x,y)
        
        pae.z = bsxfun(@plus,pae.We * log(x)',pae.bvis);
        pae.aHid = sigmoid(pae.z);

        pae.z2 = bsxfun(@plus,pae.Wd*pae.aHid,pae.bhid);
        
        Lambda = pae.beta * exp(pae.z2);
        lastterm = logadd(x');
        E = Lambda - x'.*log(Lambda)+ lastterm;

        pae.L = sum(sum((E)))/pae.batchSize; %loss function
end

function y = logadd(x)
y = zeros(size(x));
    for e = 1:numel(x)
        for i=2:x(e)
           y(e) = y(e) + log(i);
        end
    end
end