function PAEchecknumgrad(pae, x, y)
    epsilon = 1e-6;
    er = 1e-5;
    for i = 1:length(pae.bvis)
            pae_m = pae;
            pae_p = pae;
            pae_m.bvis(i) = pae_m.bvis(i) - epsilon;
            pae_p.bvis(i) = pae_p.bvis(i) + epsilon;
            pae_m = PAEff(pae_m,x,y);
            pae_p = PAEff(pae_p,x,y);
            dW = (pae_p.L-pae_m.L)/(2*epsilon);
            e = abs(dW - pae.dbvis(i));
            assert(e<er,'numberical gradient checking failed');
    end
    
%     for i = 1:size(pae.We,1)
%         for j = 1:size(pae.We,2)
%             pae_m = pae;
%             pae_p = pae;
%             pae_m.We(i,j) = pae.We(i,j) - epsilon;
%             pae_p.We(i,j) = pae.We(i,j) + epsilon;
%             
%             pae_m = PAEff(pae_m,x,y);
%             pae_p = PAEff(pae_p,x,y);
%             dW = (pae_p.L-pae_m.L)/(2*epsilon);
%             e = abs(dW - pae.dWe(i,j));
%             assert(e<er,'numberical gradient checking failed');
%         end
%     end
%     for i =1 : length(pae.bhid)
%             pae_m = pae;
%             pae_p = pae;
%             pae_m.bhid(i) = pae_m.bhid(i) - epsilon;
%             pae_p.bhid(i) = pae_p.bhid(i) + epsilon;
%              pae_m = PAEff(pae_m,x,y);
%             pae_p = PAEff(pae_p,x,y);
%             dW = (pae_p.L-pae_m.L)/(2*epsilon);
%             e = abs(dW - pae.dbhid(i));
%             assert(e<er,'numberical gradient checking failed');
%     end
%     for i = 1:size(pae.Wd,1)
%         for j = 1:size(pae.Wd,2)
%             pae_m = pae;
%             pae_p = pae;
%             pae_m.Wd(i,j) = pae.Wd(i,j) - epsilon;
%             pae_p.Wd(i,j) = pae.Wd(i,j) + epsilon;
%             
%             pae_m = PAEff(pae_m,x,y);
%             pae_p = PAEff(pae_p,x,y);
%             dW = (pae_p.L-pae_m.L)/(2*epsilon);
%             e = abs(dW - pae.dWd(i,j));
%             assert(e<er,'numberical gradient checking failed');
%         end
%     end
%     xx;
end
