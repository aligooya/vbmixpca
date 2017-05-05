function [P,Mu,Sig]=myIsoGMMFit(X,J)
[N,dim]=size(X);
[~,Mu,sumd,~] = kmeans(X,J,'MaxIter',1000,'Display','iter','OnlinePhase','off');
Sig = sum(sumd)/N;
XX = single(repmat(X,[1,J])); %N x (dim x J)
XX_MU = bsxfun(@minus,XX,single(reshape(Mu',[dim*J,1])')); %N x (dim x J)
D = squeeze(-sum(reshape(XX_MU.^2,[N,dim,J]),2)/(2*Sig));
D = bsxfun(@minus,D,max(D,[],2));
for i=1:10
    %E-Step
    D = exp(bsxfun(@minus,D,max(D,[],2)));
    P = bsxfun(@rdivide, D, sum(D,2));
    %M-Step
    Mu = bsxfun(@times,P'*X,1./sum(P,1)');
    XX_MU = bsxfun(@minus,XX,single(reshape(Mu',[dim*J,1])')); %N x (dim x J)
    D = squeeze(sum(reshape(XX_MU.^2,[N,dim,J]),2));
    tmpSig = sum(sum(D.*P))/sum(P(:))/dim;
    if (abs(Sig-tmpSig) < 1e-3*Sig)
        break
    else
        Sig = tmpSig;
    end
    D = -D/(2*Sig);
    fprintf('\t\titeration: %d Sig: %f...\n',i,Sig);
   
end
end