# vbmixpca
Implements mixture of PPCA's for point clouds
% This class computes a mixture of PPCA from spatially point clouds with no correspondances.
% The LB can be used for automatic model selection (i.e. unmber of clusters, modes of varations, etc.
%
% Usage:
%
% >> vB = vBMixPCA( 'pointSetFileNames', 'gmmFileName', M, L, J); 
% >> vB = vB.solve(0, nIter);
% 
% where:
% 
% pointSetFileName is the name of a file listing the absolute paths to point sets. Each point
% set file should define D-dimenstional points in rows: 
% x1 y1 z1
% x2 y2 z2
% .
% .
% .
% gmmFileName is an (optional) GMM file defining initialization. This can
% be empty (i.e. []). In that case, the algorithm will fit a GMM
% to the available points (usually slow). 
% 
% M is the number of Gaussian centroids which define the GMM for each point
% set.
% L is the number of varation modes in each PPCA cluster
% J is the number of PPCA clusters
% nIter is the numer of iterations to solve the mixture. After this step,
% the vB object will store the variation modes, means of PPCA cluster,
% point correspondances in its properties: E_W, muBar, E_T, E_Z, etc. 

% The variables follow their original naming in the following paper:
% 
% "Mixture of Probabilistic Principal Component Analyzers for Shapes from Point Sets" 
% DOI 10.1109/TPAMI.2017.2700276, 
% IEEE Transactions on Pattern Analysis and Machine Intelligence (in press)
% 
% Please cite the above paper in case you find the model useful in your own
% application. I might be able to help you with your quesries on how to run
% the model, if you send me an email to: a.gooya@sheffield.ac.uk
%
% This software follows the MIT license prototype.
