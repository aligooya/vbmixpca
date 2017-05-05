% This class computes a Mixture of Probabilistic Principal Component Analysers 
% from spatial point clouds with no point-to-point correspondances. If the
% point clouds represent shapes, the model generates clusters of shape
% atlases, computing mean shape and modes of variations in each cluster.

% The computed Lower Bound (LB) on the data likelihood can be used for automatic 
% model selection (i.e. unmber of clusters, modes of varations, etc).
%
% Usage:
%
% >> vB = vBMixPCA( 'pointSetsList', 'gmmFileName', M, L, J); 
% >> vB = vB.solve(0, nIter);
% 
% where:
% 
% pointSetsList:        is the name of a file listing the absolute paths to point sets. Each point
%                       set file should define N_k x D (dimensional) points in rows: 
% x1 y1 z1
% x2 y2 z2
% .
% .
% .
% gmmFileName:          is an (optional) GMM file defining initialization. This can
%                       be empty (i.e. []). In that case, the algorithm will fit a GMM (with M
%                       centroids) to the available points (usually slow). You can save your GMM file using
%                       writeGmm method provided by the class for a later use.
% 
% M:                    is the number of Gaussian centroids which define the GMM for each point
%                       set. M*dim(points) determines the latent dimension of the MixPPCA sapce.

% L:                    is the number of varation modes in each PPCA cluster
% J:                    is the number of PPCA clusters
% nIter:                is the numer of iterations to solve the mixture. 

% After solve() step,  the vB object will store the variation modes, means of PPCA cluster,
% point correspondances in its properties: E_W, muBar, E_T, E_Z, etc
% where E_{.} stands for expectation. 

% You can access the E_Mu's (the projections to PPCA spaces), muBar (PPCA cluster means) using vis_E_Mu
% method. For instance:

% >> figure, vB.vis_E_Mu(1:vB.J,1:vB.K,[],1,[],0)

% You can also visualize the point sets and their clustrs using

% >> figure, vB.vis_X([],[])

% The 'model evidence' for a particular setting of M, L, and J can be
% accessed by vB.LB values, thus faciliating the model selection. LB can
% also be used to monitor the convergence. In practice, 

% It is recommended that you have access to RAM > 16 GByte, to process a
% collection of 50 point sets with 4000 points each. 

% This code uses a vairant of Fast and Efficient Spectral Clustering for
% initial clustering of shapes (thanks and credit to Ingo University of
% Stuttgart)

% The variables follow their original naming in the following paper:
% 
% "Mixture of Probabilistic Principal Component Analyzers for Shapes from Point Sets" 
% DOI 10.1109/TPAMI.2017.2700276, 
% IEEE Transactions on Pattern Analysis and Machine Intelligence (in press)
% 
% 
% Please cite the above paper in case you find the model useful in your own
% research. I might be able to help you with your quesries on how to run
% the model, if you send me (Ali Gooya) an email to: a.gooya@sheffield.ac.uk
%
% This software follows the MIT license prototype.


classdef vBMixPCA
   properties
        X % A cell array keeping Xk
        projX % A cell array keeping projected Xk
        E_T % An array keeping <tk>  K x J (shape cluster variables)   
        E_V % An array to keep v_k L x K  (loading vectors)
        Cov_V % L x L x K 3D matrix 
        E_Mu % MD x J x K 3D array  (projections to PPCA spaces)
        Diag_Cov_Mu % MD x J x K 
        E_W % MD x L x J 3D array (Modes of variations)
        initW % MD x L x J 3D array 
        Cov_W % MD x L x J 3D array 
        E_Z % A cell array of 2D arrays keeping posterior values K x (Nk x M) points
        E_Bet % A double %needs a carefull removal
        E_lnBet
        M %dimension of Mu_k
        L %dimention of V_k
        dim %dimention of points
        J %number of classes
        K %number of point sets
        N %total number of points
        E_MuBar % An array  MD x J  (shape cluster means)
        Cov_MuBar % An array MD x J
        a,b, aBar, bBar % doubles
        XBar % An array keeping XBar_k vectors  MD x K
        R % MD x K 2D array <-- A cell array keeping R_k K x (MD x MD)
        E_Omega % A matrix keeping precisions of modes L x J
        E_lnOmega % A matrix keeping precisions of modes L x J
        E_Pi % A vector J x 1
        E_lnPi % J x 1
        kmeansIdx
        lnRho % A cell array K x (Nk x M)
        E_VVt % L x L x K 3D matrix <--- A cell array K x (L x L)
        Lambda % Hypers for p(pi), Jx1
        Lambda0 % Hyper for p(pi), double
        LB % Lower bound
        E_PiP % Mixing coef over Z, M x 1
        E_lnPiP % A vector M x 1
        LambdaP % Hypers for p(pip), M x 1 
        Lambda0P  % Hyper for p(pip), double
        EPS
        gmm
        listFileName
        gmmFileName
        epsBar % An L x J array to keep scale hyperparameter of Omega
        etaBar % A  L x J array to keep shape hyperparameters of Omega
        eps % A double
        eta % A double
    end
    
    methods

        function vB = vBMixPCA(listFileName,gmmFileName,M,L,J)
            if (nargin == 5)
                vB.L = L;
                vB.M = M;
                vB.J = J;
                vB = vB.update_X(listFileName,J);
                
                
                if ( ~isempty(gmmFileName) )
                    [~,struc] = fileattrib(gmmFileName);
                    vB.gmmFileName = struc.Name;
                    data = dlmread(gmmFileName,' ');
                    [d1,d2]=size(data);
                    vB.M = d1-1;
                    vB.dim = d2 -1;
                    sigma = data(1,1:vB.dim);
                    mu = data(2:end,1:vB.dim);
                    pI = data(2:end,vB.dim + 1)';
                    vB.gmm = gmdistribution(mu,sigma,pI);
                    
                end
                 
                           
                for j =1:vB.J
                    vB.E_W(:,:,j) = zeros(vB.M*vB.dim, vB.L,'single');
                    vB.Cov_W(:,:,j) = zeros(vB.M*vB.dim, vB.L,'single');
                end
                
                for k =1:vB.K
                    for j =1:vB.J
                         vB.E_Mu(:,j,k) = zeros(vB.dim*vB.M,1,'single');
                        vB.Diag_Cov_Mu(:,j,k) = zeros(vB.dim*vB.M,1,'single');
                     end
                    vB.E_V(:,k) = zeros(vB.L,1);
                    vB.Cov_V(:,:,k) = zeros(vB.L);
                    vB.E_VVt(:,:,k) = zeros(vB.L);
                end
                
                vB.E_Omega = ones(vB.L, vB.J)/vB.N;
                vB.E_lnOmega = log(vB.E_Omega);
                
                vB.a = 0;
                vB.b = 0;
                vB.aBar = vB.a;
                vB.bBar = vB.b;
                vB.E_Bet = vB.a/vB.b;
                vB.E_lnBet = log(vB.E_Bet);
                vB.Lambda0 = 1; 
                vB.Lambda = vB.Lambda0 *ones(vB.J,1);
                vB.E_Pi = ones(1,vB.J)./vB.J;
                vB.E_lnPi = log(vB.E_Pi);
                
                vB.Lambda0P = 1; 
                vB.LambdaP = vB.Lambda0P *ones(1,vB.M);
                vB.E_PiP = ones(1,vB.M)./vB.M;
                vB.E_lnPiP = log(vB.E_PiP);
                vB.EPS = 1e-12;

                vB.eps = 0; 
                vB.eta = 0;
                vB.epsBar = vB.eps*ones(vB.L,vB.J);
                vB.etaBar = vB.eta*ones(vB.L,vB.J);
            end 
        end % constructor 
       function obj = update_X(obj,listFileName,J)
                %retrive the names and read the data files
                [~,struc] = fileattrib(listFileName);
                obj.listFileName = struc.Name;
                
                listFile = fopen(listFileName,'r');
                dataFileName = fgetl(listFile);
                K = 0;
                N = 0;
                while  ischar(dataFileName)
                    K = K + 1;
                    obj.X{K}= single(dlmread(dataFileName,' '));
                    [Nk ,dim]=size(obj.X{K});
                    obj.E_Z{K}=zeros(Nk,obj.M);
                    obj.lnRho{K}=zeros(Nk,obj.M);
                    N = N + Nk;
                    dataFileName = fgetl(listFile);
                end
                fclose(listFile);
                obj.N = N;
                %%% resize the variables if needed.
                for k=K+1:obj.K
                    obj.E_Z{k}=[];
                    obj.lnRho{k}=[];
                end
                obj.K = K;
                obj.dim = dim;
                
                %%%% revised from submission %%%%
                %resizing the relevant variables
                obj.E_Mu = zeros(obj.dim*obj.M, obj.J, obj.K);
                obj.Diag_Cov_Mu = zeros(obj.dim*obj.M, obj.J, obj.K);
                obj.E_V = zeros(obj.L, obj.K); %no loading initially
                obj.Cov_V = repmat(eye(obj.L),[1,1,obj.K]);
                obj.E_VVt = obj.Cov_V;
                obj.XBar = zeros(obj.dim*obj.M,obj.K);
                obj.R = zeros(obj.dim*obj.M,obj.K);
                obj.E_T = ones(obj.K,obj.J)/obj.J;
                %%%%%%%%%%%%%%%%%
        end
        
       function obj = test(obj,nIter)
            obj.E_T = ones(obj.K,obj.J)/obj.J;
            obj.LB = [];
            %%%% revised from submission %%%%
            obj = obj.update_E_Mu();
            %%%%%%%%%%%%%%%%%
            for n=1:nIter
                fprintf('\t\tupdating lnRho ...\n');
                obj = obj.update_lnRho();
                fprintf('\t\tupdating E_Z ...\n');
                obj = obj.update_E_Z();
                fprintf('\t\tupdating XBar ...\n');
                obj = obj.update_XBar();
                fprintf('\t\tupdating V ...\n');
                obj = obj.update_E_V();
                obj = obj.update_E_Mu();
                obj = obj.update_abBar();
                
                %%% This can be set on/off
                %fprintf('\t\tupdating Beta ...\n');
                %obj = obj.update_E_Bet(); 
                %%%%%%%%%%%%%%
                
                fprintf('\t\tupdating E_T ...\n');
                obj = obj.update_E_T();
                obj = obj.update_LB(n);
                
                fprintf('\t\titeration: %d |eT|: %f\n',n,trace(obj.E_T*obj.E_T'))                
                
            end  
       end
       
       function writeGmm(obj,filename)
           gmmData(1,:)=[diag(obj.gmm.Sigma)',0];
           gmmData = [gmmData;[obj.gmm.mu,obj.gmm.PComponents']];
           dlmwrite(filename,gmmData,' ');
       end
       
       function obj = initialize(obj,Mu)
           sumT = floor(sum(obj.E_T,1));
           max_eig_no = obj.M*obj.dim;
           for j=1:obj.J
               obj.E_W(:,:,j) = zeros(obj.dim*obj.M,obj.L);
               obj.initW(:,:,j) = zeros(obj.dim*obj.M,obj.L);
               obj.E_MuBar(:,j) = Mu*obj.E_T(:,j)./sum(obj.E_T(:,j));
               S = bsxfun(@minus,Mu,obj.E_MuBar(:,j));
               S = bsxfun(@times,S,obj.E_T(:,j)'./sum(obj.E_T(:,j)))*S';
               effK = min(sumT(j),max_eig_no);
               effL = min(obj.L,effK-1);
               [U, S0, flag] = eigs(double(S),effK);
               
                   
               dS0=diag(S0);
               s(j) = mean(dS0(effL+1:end));
               obj.initW(:,1:effL,j) = single(U(:,1:effL)*diag(sqrt(abs(dS0(1:effL) - s(j)) )));
               obj.E_W = obj.initW;
           end
           for k=1:obj.K
                j = obj.kmeansIdx(k);
                W = obj.E_W(:,:,j);
                xB = obj.E_MuBar(:,j);
                obj.E_V(:,k) = (W'*W + eye(obj.L))\W'*(Mu(:,k) - xB);
                obj.E_VVt(:,:,k) = eye(obj.L);
            end
            for k=1:obj.K
                for j =1:obj.J
                   obj.E_Mu(:,j,k) = single(obj.E_W(:,:,j)*obj.E_V(:,k) + obj.E_MuBar(:,j));
                   obj.Diag_Cov_Mu(:,j,k) = diag(obj.E_W(:,:,j)*obj.E_W(:,:,j)');
                end
            end            
            for k=1:obj.K
                j = obj.kmeansIdx(k);
                D(:,k) = obj.E_Mu(:,j,k) - Mu(:,k);
            end
                      
           obj.E_Bet = obj.M*obj.dim*obj.K/sum(D(:).*D(:));
           obj.E_lnBet = log(obj.E_Bet);
           obj.Cov_MuBar = zeros(obj.M*obj.dim,obj.J);
       end
       
       function obj = clusterUsingGraphLalacian(obj,F) %F is the normalized posteriors from initialization
           W = mySimGraph_NearestNeighbors(F',5,1,.1);
           [C, ~, ~,D] = mySpectralClustering(W, obj.J, 3);
           obj.kmeansIdx =C*(1:obj.J)';
           vaR=sum(sum(D.*full(C),1))/obj.K/obj.J;
           D = -D/(2*vaR);
           obj.E_T = exp(bsxfun(@minus, D, max(D, [], 2)));
           obj.E_T = bsxfun(@rdivide, obj.E_T, sum(obj.E_T,2));
       end
       
       function obj = solve(obj,nStart, nIter)
            if ( nStart == 0 )
                if ( isempty(obj.gmm) )
                    %fitting a mixture model to all data points
                    %options = statset('MaxIter',1000,'Display','iter','TolX',10);
                    %obj.gmm = gmdistribution.fit(cat(1,obj.X{1:obj.K}),obj.M,'SharedCov',true,'Options',options,'Replicate',1,'CovType','diagonal','Regularize',1e-12);
                    
                    % this is faster
                    [~,gmmMean,gmmBet]=myIsoGMMFit(cat(1,obj.X{1:obj.K}),obj.M);
                    obj.gmm = gmdistribution(gmmMean,gmmBet*eye(obj.dim),ones(obj.M,1)/obj.M);
                end
                %finding "corresponding points" and posteriors
                F = zeros(obj.K,obj.M);
                Mu = zeros(obj.M*obj.dim,obj.K);
                for k=1:obj.K
                    P=posterior(obj.gmm,obj.X{k});
                    [~,I] = max(P,[],1);
                    Mu(:,k) = reshape(obj.X{k}(I,:)',[obj.M*obj.dim,1])';
                    f = sum(P,1);
                    F(k,:) = f./sum(f);
                end
               
                [U,S,V]=svd(F);
                nC = round(min(obj.M/2,obj.K/2)); %this can be refined 
                
                F = U(:,1:nC)*S(1:nC,1:nC)*V(:,1:nC)';
               
                if (obj.J > 1)
                    obj = obj.clusterUsingGraphLalacian( F);
                    
                else
                    obj.E_T = ones(obj.K,obj.J);
                    obj.kmeansIdx = ones(obj.K,1);
                end
                obj.E_Pi = mean(obj.E_T,1);
                
                obj = obj.initialize(Mu);

                fprintf('\t\tinitialization done ...\n');
            end

                        
            for n=(nStart+1):(nStart+nIter+1)
                fprintf('\t\tupdating lnRho ...\n');
                obj = obj.update_lnRho();
                fprintf('\t\tupdating E_Z ...\n');
                obj = obj.update_E_Z();
                obj = obj.prune();  
                fprintf('\t\tupdating XBar ...\n');
                obj = obj.update_XBar(); 
                obj = obj.update_abBar();
                fprintf('\t\tupdating Beta ...\n');
                obj = obj.update_E_Bet();
                fprintf('\t\tupdating W ...\n');
                obj = obj.update_E_W();
                fprintf('\t\tupdating V ...\n');
                obj = obj.update_E_V();
                obj = obj.update_E_Mu(); 
                fprintf('\t\tupdating E_T ...\n');
                obj = obj.update_E_T();
                if find(sum(obj.E_T) == 0)
                    obj = obj.prune();
                    fprintf('\t\tEffective number of clusters is reduced to: %d\n',obj.J);
                end               

                obj = obj.update_epsEtaBar();
                fprintf('\t\tupdating E_Omega ...\n');
                obj = obj.update_E_Omega();             
                              
                fprintf('\t\tupdating Pi ...\n');
                obj = obj.update_E_Pi();
               
                fprintf('\t\tupdating PiP ...\n');
                obj = obj.update_E_PiP();
                fprintf('\t\tupdating E_MuBar ...\n');
                obj = obj.update_E_MuBar();
             
                obj = obj.update_LB(n);
                sumW = 0;
                

                for j=1:obj.J
                    sumW = sumW + sqrt(sum(sum(obj.E_W(:,:,j)'*obj.E_W(:,:,j))));
                end
                fprintf('\t\titeration: %d E_Bet: %f |E_W|: %f |eT|: %f LB: %f\n',...
                    n,obj.E_Bet,sumW, trace(obj.E_T*obj.E_T'), obj.LB(n))
            end       
           
        end
        
       function obj = saveToStruct(obj, filename)
              varname = inputname(1);
              props = properties(obj);
              for p = 1:numel(props)
                  s.(props{p})=obj.(props{p});
              end
              eval([varname ' = s'])
              save(filename, varname)
       end      
       function obj = update_E_PiP(obj)
            ez = zeros(1,obj.M);
            for k=1:obj.K
                ez = ez + sum(obj.E_Z{k},1);
            end
            obj.LambdaP = obj.Lambda0P + ez;
            obj.E_PiP = obj.LambdaP./sum(obj.LambdaP);
            obj.E_lnPiP = psi(obj.LambdaP) - psi(sum(obj.LambdaP)*ones(size(obj.LambdaP)));
       end

       
       
       function [obj,L] = update_LB(obj,iter)

            L = -obj.aBar*log(obj.bBar);

            for j=1:obj.J
                for l = 1:obj.L
                    L = L - obj.epsBar(l,j)*log(obj.etaBar(l,j)) + 0.5*sum(log(obj.Cov_W(:,l,j)));
                end
            end
            for k=1:obj.K
               eZ = obj.E_Z{k}(obj.E_Z{k} > obj.EPS);
               L = L - sum(sum(eZ.*log(eZ)));
            end
            
            eT = obj.E_T((obj.E_T > 0));
            L = L - sum(sum(eT.*log(eT)));

            for k=1:obj.K
                L =  L + obj.L/2;
                L =  L - 0.5*(trace(obj.E_VVt(:,:,k)) - log(det(obj.Cov_V(:,:,k))));
            end

            L = L + gammaln(obj.J*obj.Lambda0) - obj.J*gammaln(obj.Lambda0);
            L = L - gammaln(sum(obj.Lambda)) + sum(gammaln(obj.Lambda));
            
            L = L + gammaln(obj.M*obj.Lambda0P) - obj.M*gammaln(obj.Lambda0P);
            L = L - gammaln(sum(obj.LambdaP)) + sum(gammaln(obj.LambdaP));
            
            L = L + sum(log(1:obj.L));
            L = L + sum(log(1:obj.J));
            L = L + sum(log(1:obj.M));
            obj.LB(iter) = L;
        end
       
       function obj = prune(obj)
            %prunning redundant classes
            jInd = find(sum(obj.E_T,1) == 0);
            obj.E_Pi(jInd) = [];
            obj.E_lnPi(jInd) = [];
            obj.Lambda(jInd) = [];
            obj.E_MuBar(:,jInd) = [];
            obj.E_W(:,:,jInd) = [];
            obj.Cov_W(:,:,jInd) = [];
            obj.E_Mu(:,jInd,:) = [];
            obj.Diag_Cov_Mu(:,jInd,:) = [];
            obj.E_T(:,jInd) = [];
            obj.E_Omega(:,jInd) = [];
            
            obj.J = length(obj.E_Pi);
            obj.E_Pi = obj.E_Pi./sum(obj.E_Pi);
            obj.E_T = obj.E_T./(repmat(sum(obj.E_T,2),[1,obj.J]));
            %prunning redundant model points
            mInd = find(sum(cat(1,obj.E_Z{1:obj.K}),1) == 0);
            obj.E_PiP(mInd) = [];
            obj.E_lnPiP(mInd) = [];
            obj.LambdaP(mInd) = [];
            for k=1:obj.K
                obj.lnRho{k}(:,mInd) = [];
                obj.E_Z{k}(:,mInd) = [];
            end
            ind = obj.dim*(mInd-1)+1;
            for d=1:obj.dim
                obj.E_Mu(ind,:,:) =[];
                obj.Diag_Cov_Mu(ind,:,:) = [];
                obj.R(ind,:) = [];
                obj.XBar(ind,:) = [];
                obj.E_W(ind,:,:) = [];
                obj.Cov_W(ind,:,:) = [];
                obj.E_MuBar(ind,:) = [];
            end
            obj.M = length(obj.E_PiP);
            obj.E_PiP = obj.E_PiP./sum(obj.E_PiP);
        end
        
 
       function obj = update_E_V(obj)
            for k = 1:obj.K
                Dk = zeros(obj.L);
                dk = zeros(obj.L,1);
                for j = 1:obj.J
                    Tr =bsxfun(@times,obj.R(:,k),obj.Cov_W(:,:,j)); %MD x L
                    Tr =diag(sum(Tr,1));
                    Dk = Dk + obj.E_T(k,j)*(obj.E_W(:,:,j)'*diag(obj.R(:,k))*obj.E_W(:,:,j)  + Tr);
                    dk = dk + obj.E_T(k,j)*(obj.E_W(:,:,j)'*(obj.XBar(:,k) - obj.R(:,k).*obj.E_MuBar(:,j)));
                end
                obj.Cov_V(:,:,k) = inv(eye(obj.L) + obj.E_Bet*Dk);
                obj.E_V(:,k) = obj.E_Bet * obj.Cov_V(:,:,k)*dk;
                obj.E_VVt(:,:,k) = obj.Cov_V(:,:,k) + obj.E_V(:,k) * (obj.E_V(:,k))';
            end
       end
       
      
       function obj = update_epsEtaBar(obj)
           for j = 1:obj.J
               for l = 1:obj.L
                  obj.epsBar(l,j) = obj.eps + (obj.dim * obj.M)/2;
                  obj.etaBar(l,j) = obj.eta +  (sum(obj.Cov_W(:,l,j)) + obj.E_W(:,l,j)'*obj.E_W(:,l,j))/2;
               end
           end
       end
       
       function obj = update_E_Omega(obj)
            for j = 1:obj.J
                for l = 1:obj.L
                    obj.E_Omega(l,j) = obj.epsBar(l,j)/obj.etaBar(l,j);
                    obj.E_lnOmega(l,j) = psi(obj.epsBar(l,j)) - log(obj.etaBar(l,j));
                end
            end
        end        
        
        function obj = update_E_W(obj)
            for j =1:obj.J
                Qj = zeros(obj.dim*obj.M,obj.L,'single'); %MD x L
                for k=1:obj.K
                    Gk = obj.E_VVt(:,:,k) - diag(diag(obj.E_VVt(:,:,k)));
                    Qj = Qj + obj.E_T(k,j)*((obj.XBar(:,k) - obj.R(:,k).*obj.E_MuBar(:,j))*obj.E_V(:,k)' - diag(obj.R(:,k))*obj.E_W(:,:,j)*Gk);
                end
                
                for l=1:obj.L
                    ebtv = obj.E_Bet * obj.E_T(:,j).*squeeze(obj.E_VVt(l,l,:)); %K x 1
                    ebtv = sum(bsxfun(@times,obj.R,ebtv'),2); %MD x 1
                    Rlj = obj.E_Omega(l,j) + ebtv; %MD x 1
                    obj.Cov_W(:,l,j) = single(1./Rlj); %MD x 1
                    obj.E_W(:,l,j) = single(obj.E_Bet * obj.Cov_W(:,l,j).*Qj(:,l));
                    if ( sum(isnan( obj.E_W(:,l,j)) ) )
                        sprintf('E_W has nan!\n');
                        return;
                    end
                    if ( sum(isnan( obj.Cov_W(:,l,j)) ) )
                        sprintf('Cov_W has nan!\n');
                        return;
                    end
                end
            end
        end
        
        function obj = update_projX(obj)
            for k=1:obj.K
                [Nk ~] = size(obj.X{k});
                E_P = zeros(Nk,obj.dim);
                for j =1:obj.J
                    E_P = E_P + obj.E_T(k,j)*obj.E_Z{k}*obj.v2P(obj.E_Mu(:,j,k));
                end 
                obj.projX{k}= E_P;
            end
        end
        
        function vis_projX(obj,k, clr)
            E_P = obj.projX{k};
            [Nk ~] = size(obj.X{k});
            cmap=linspecer(obj.J);
            if isempty(clr)
                eT = (obj.E_T(k,:) == max(obj.E_T(k,:)));
                mclr =  (1:obj.J) *  eT' ;
                if ( obj.dim == 3)
                   scatter3(E_P(:,1),E_P(:,2),E_P(:,3),100,'Cdata',repmat(cmap(mclr,:),[Nk , 1]),'Marker','x'), hold on;
                else
                   scatter(E_P(:,1),E_P(:,2),100,'Cdata',repmat(cmap(mclr,:),[Nk , 1]),'Marker','x'), hold on;
                end
            else
                eT = (obj.E_T(k,:) == max(obj.E_T(k,:)));
                mclr =  (1:obj.J) *  eT' ;
                if (isnumeric(clr))
                   clr2=repmat(clr,[Nk,1]);
                end
                if ( obj.dim == 3)
                   scatter3(E_P(:,1),E_P(:,2),E_P(:,3),100,'CData',clr2,'Marker','x'), hold on;
                else
                   scatter(E_P(:,1),E_P(:,2),100,'CData',clr2,'Marker','x'), hold on;
                end
           end
            
        end
        
        function vis_X(obj,ind,clr)
            cmap=linspecer(obj.J);
            
            
            
            if isempty(ind)
                ind = 1:obj.K;
            end
            for k = ind
                [Nk ~] = size(obj.X{k});
                if isempty(clr)
                    eT = (obj.E_T(k,:) == max(obj.E_T(k,:)));
                    mclr =  (1:obj.J) *  eT' ;
                    if ( obj.dim == 3)
                        scatter3(obj.X{k}(:,1),obj.X{k}(:,2),obj.X{k}(:,3),20,'Cdata',repmat(cmap(mclr,:),[Nk , 1]),'Marker','x'), hold on;
                    else
                        scatter(obj.X{k}(:,1),obj.X{k}(:,2),100,'Cdata',repmat(cmap(mclr,:),[Nk , 1]),'Marker','x','linewidth',2), hold on;
                    end
                else
                    %eT = (obj.E_T(k,:) == max(obj.E_T(k,:)));
                    %mclr =  (1:obj.J) *  eT' ;
                    if (isnumeric(clr))
                        clr2=repmat(clr,[Nk,1]);
                    end
                    if ( obj.dim == 3)
                        scatter3(obj.X{k}(:,1),obj.X{k}(:,2),obj.X{k}(:,3),20,'CData',clr2,'Marker','x'), hold on;
                    else
                        scatter(obj.X{k}(:,1),obj.X{k}(:,2),100,'CData',clr2,'Marker','x','linewidth',2), hold on;
                    end
                end
                   
            end
        end
        
        function obj = update_E_Pi(obj)
            obj.Lambda = (obj.Lambda0 + sum(obj.E_T,1));
            obj.E_Pi = (obj.Lambda)./sum(obj.Lambda);
            obj.E_lnPi = psi(obj.Lambda) - psi(sum(obj.Lambda)*ones(size(obj.Lambda)));
        end
       
        function obj = update_ab(obj)
            %obj.a = obj.aBar;   
            obj.b = obj.a*(obj.bBar - obj.b)/(obj.aBar - obj.a);
        end
        
     
        
        function obj = update_E_MuBar(obj)
            for j =1:obj.J
                Rj = zeros(obj.dim * obj.M,1);
                xj = zeros(obj.dim * obj.M,1);
                for k=1:obj.K 
                    Rj = Rj + obj.E_T(k,j)*obj.R(:,k);
                    xj = xj + obj.E_T(k,j)*(obj.XBar(:,k) - bsxfun(@times, obj.R(:,k),obj.E_W(:,:,j))*obj.E_V(:,k)); %MD x 1
                end
			       obj.Cov_MuBar(:,j) = 1./(obj.E_Bet*Rj);
			       obj.E_MuBar(:,j) = obj.Cov_MuBar(:,j).*xj*obj.E_Bet;
                if ( sum ( isnan(obj.E_MuBar(:,j))))
                    sprintf('E_MuBar has nan!\n')
                    return;
                end
            end
        end

        function obj = update_E_Mu(obj)
            for k=1:obj.K
                for j =1:obj.J
                   obj.E_Mu(:,j,k) = single(obj.E_W(:,:,j)*obj.E_V(:,k) + obj.E_MuBar(:,j));
                   obj.Diag_Cov_Mu(:,j,k) = sum( obj.E_W(:,:,j).*(obj.E_W(:,:,j)*(obj.E_VVt(:,:,k) - obj.E_V(:,k)*obj.E_V(:,k)'))...
                       +bsxfun(@times,diag(obj.E_VVt(:,:,k))', obj.Cov_W(:,:,j)),2);% + obj.Cov_MuBar(:,j);
                end
            end
        end
        function obj = update_E_T(obj)
            lam = zeros(obj.K,obj.J,'single');
            for j =1:obj.J
                for k=1:obj.K
                    lam(k,j) = single((-0.5*obj.E_Bet*(sum(obj.R(:,k).*(obj.Diag_Cov_Mu(:,j,k) + obj.E_Mu(:,j,k).^2))) ...
                               +obj.E_Bet*obj.E_Mu(:,j,k)'*obj.XBar(:,k) + obj.E_lnPi(j)));   
                end
            end
            obj.E_T = exp(bsxfun(@minus, lam, max(lam, [], 2)));
            obj.E_T = bsxfun(@rdivide, obj.E_T, sum(obj.E_T,2));
            if ( sum(sum(isnan(obj.E_T))) )
                sprintf('E_T has nan!\n')
                    return;
            end
        end
                
        function obj = update_E_Z(obj)
            for k=1:obj.K
                rho = bsxfun(@plus,-.5*obj.E_Bet*obj.lnRho{k}, obj.E_lnPiP);
                rho = exp(bsxfun(@minus, rho, max(rho, [], 2)));
                rho = bsxfun(@rdivide, rho, sum(rho,2))+obj.EPS;
                obj.E_Z{k} = rho;
                obj.R(:,k) = reshape(repmat(sum(rho,1),obj.dim,1),obj.dim*obj.M,1);
            end
            R = obj.R;
            if sum(isnan(R(:)))
                fprintf('Matrix R has nan!\n');
                return;
            end
            clear R;
        end
        
        function obj = update_E_Bet(obj)
                obj.E_Bet = obj.aBar/obj.bBar;
                obj.E_lnBet = psi(obj.aBar) - log(obj.bBar);
        end        
        
        
        function obj = update_abBar(obj)
            obj.aBar = obj.a + .5*obj.dim*obj.N;
            obj.bBar = obj.b;
            for k=1:obj.K
                obj.bBar = obj.bBar + 0.5*sum(sum(obj.E_Z{k}.*obj.lnRho{k}));
            end
        end
 
        function obj = update_lnRho(obj)
            for k=1:obj.K
                [Nk, dim] = size(obj.X{k});
                lam = zeros([Nk obj.M]);
                for j = 1:obj.J
                    TR = sum(reshape(obj.Diag_Cov_Mu(:,j,k),[dim,obj.M]),1); %1 x M
                    MU = repmat(obj.v2P(obj.E_Mu(:,j,k))',[Nk , 1]); %(DxNk) x M
                    X = obj.X{k};
                    X_MU = bsxfun(@minus,obj.p2V(X),MU);
                    lam = lam + obj.E_T(k,j)*bsxfun(@plus,reshape(sum(reshape(X_MU.^2,dim,Nk*obj.M),1),[Nk,obj.M]),TR);
                end
                obj.lnRho{k} = lam;
            end
        end
                
        function obj = update_XBar(obj)
            for k=1:obj.K
                obj.XBar(:,k) = obj.p2V( (obj.E_Z{k})'*obj.X{k} );
            end
        end
 
        function p = v2P(obj, v)
            p = reshape(v,[obj.dim obj.M])';
        end
        
        function samples = sample(obj, nS, samplesListFileName, genericName)
            samplesListFile = fopen(samplesListFileName,'w');
            %samples = zeros(obj.M, obj.dim, nS);
            for n = 1:nS
                %smapling a type
                t = rand();
                for j=1:obj.J
                    if (t<sum(obj.E_Pi(1:j)))
                        t = j;
                        break;
                    end
                end
                v = mvnrnd(zeros(obj.L,1),eye(obj.L));
                %v = zeros(1,obj.L);
                means = obj.v2P( obj.E_MuBar(:,t) + obj.E_W(:,:,t)*v');
                meansIdx = mnrnd(1,obj.E_PiP,obj.M)*[1:obj.M]';

                samples(:,:,n) = means(meansIdx,:);
                %samples(:,:,n) = means;
                sampleFileName = sprintf('%s/%s_%03d.txt',pwd,genericName,n);
                dlmwrite(sampleFileName,samples(:,:,n),'delimiter',' ');
                fprintf(samplesListFile,'%s\n',sampleFileName);
                %dlmwrite(sampleFileName,samples(:,:,n),'delimiter',' ');
            end
            fclose('all');
        end
       
        function vis_E_Mu(obj,jInd,kInd,clr,nCol,axs,thr)
            cmap2=linspecer(obj.J);
                
            cls = (obj.E_T == repmat(max(obj.E_T,[],2),[1, obj.J]));
            if isempty(jInd)
                jInd = 1:obj.J;
            end
            pInd = 1;
            
            N =length(jInd);
            if isempty(nCol)
                nCol = 10;
            end
            nR = floor(N/nCol);
            for j=jInd
                cl = j;
                ind_J = find(cls(:,j) == 1);
                eZ = cat(1,obj.E_Z{(ind_J)});
                nu = sum(eZ,1)/max(sum(eZ,1));
                ALP = nu*obj.E_Pi(j)/max(obj.E_Pi);
                thALP = (ALP>thr)';
                MuB = obj.v2P(obj.E_MuBar(:,j));
                MuB = MuB(thALP==1,:);
                [effM, ~] = size(MuB);
                CLR = rgb2hsv(repmat(cmap2(cl,:),[effM , 1]));
% comment out next 2 lines for transparency                
%                CLR(:,2) = CLR(:,2,:).*ALP(:,thALP==1)';
%                CLR(:,3) = ones(effM,1);
                CLR = hsv2rgb(CLR);
                if ( ~isempty(kInd) )
                    for k=kInd
                        if sum(k == ind_J)
                            Mu_k = obj.v2P(obj.E_Mu(:,j,k));
                            Mu_k = Mu_k(thALP==1,:);
                            if (nCol > 1)
                                subplot(nR+1,nCol,pInd)
                            end
                            if ( isempty(clr) || isnumeric(clr) )
                                if (obj.dim  == 2)
                                    scatter(Mu_k(:,1),Mu_k(:,2),40, 'Cdata',CLR,'Marker','*'), hold on;
                                elseif (obj.dim == 3)
                                    scatter3(Mu_k(:,1),Mu_k(:,2),Mu_k(:,3),40, 'Cdata',CLR,'Marker','*'), hold on;
                                    %plt=plot3(Mu_k(:,1),Mu_k(:,2),Mu_k(:,3),'o'); hold on;
                                    %set(plt.MarkerHandle,'EdgeColorBinding','interpolated','EdgeColorData',CLR);
                                end
                            else
                                if (obj.dim  == 2)                                
                                    scatter(Mu_k(:,1),Mu_k(:,2),40, clr,'Marker','*'), hold on;
                                elseif (obj.dim == 3)
                                    scatter3(Mu_k(:,1),Mu_k(:,2),Mu_k(:,3), 40,clr,'Marker','*'), hold on;
                                end
                            end%if
                            if ~isempty(axs)
                                axis(axs);
                            end
                        end
                    end%for k
                end%if
                
                if ( isempty(clr) || isnumeric(clr) )
                    if (nCol > 1)
                      subplot(nR+1,nCol,pInd)
                    end
                    %subplot(nR+1,nCol,pInd)
                    if (obj.dim  == 2)
                        scatter(MuB(:,1),MuB(:,2),40,'Cdata',CLR,'Marker','.','LineWidth',2 ), hold on;
                    elseif ( obj.dim == 3)
                        scatter3(MuB(:,1),MuB(:,2),MuB(:,3),40, 'Cdata',CLR,'Marker','o','LineWidth',2 ), hold on;
                        %plt=plot3(MuB(:,1),MuB(:,2),MuB(:,3),'.'); hold on;
                        %set(plt.MarkerHandle,'EdgeColorBinding','interpolated','EdgeColorData',CLR);
                    end
                    
                else
                    if (nCol > 1)
                      subplot(nR+1,nCol,pInd)
                    end
                    %subplot(nR+1,nCol,pInd)
                    if (obj.dim  == 2)
                        scatter(MuB(:,1),MuB(:,2), 40, clr,'Marker','.','LineWidth',2), hold on;
                    elseif ( obj.dim == 3)
                        scatter3(MuB(:,1),MuB(:,2),MuB(:,3), 40, clr,'Marker','o','LineWidth',2), hold on;
                        %plt=plot3(MuB(:,1),MuB(:,2),MuB(:,3),'*'); hold on;
                        %set(plt.MarkerHandle,'EdgeColorBinding','interpolated','EdgeColorData',CLR);
                    end
                end 
                if ~isempty(axs)
                   axis(axs);
                end
                pInd = pInd +1;
            end%bug for
        end%function
    
    end %methods
  
    methods(Static)
            function v = p2V(p)
            [m n] = size(p);
            v =  reshape(p',[m*n,1]);
            end
            
            function r = drchrnd(a,n)
            % take a sample from a dirichlet distribution
            p = length(a);
            r = gamrnd(repmat(a,n,1),1,n,p);
            r = r ./ repmat(sum(r,2),1,p);                      
            end
    end
  
end
