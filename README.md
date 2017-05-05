Variational Bayes Mixture of PCA
This class computes a Mixture of Probabilistic Principal Component Analysers 
from spatial point clouds with no point-to-point correspondances. If the
point clouds represent shapes, the model generates clusters of shape
atlases, computing mean shape and modes of variations in each cluster using 
a variatonal Bayesian inference approach in MATLAB.

The computed Lower Bound (LB) on the data likelihood can be used for automatic 
model selection (i.e. unmber of clusters, modes of varations, etc).

The variables follow their original naming in the following paper:

"Mixture of Probabilistic Principal Component Analyzers for Shapes from Point Sets" 
DOI 10.1109/TPAMI.2017.2700276, 
IEEE Transactions on Pattern Analysis and Machine Intelligence (in press)
 
 
Please cite the above paper in case you find the model useful in your own
research. I might be able to help you with your quesries on how to run
the model, if you send me (Ali Gooya) an email to: a.gooya@sheffield.ac.uk



Usage:

>> vB = vBMixPCA( 'pointSetsList', 'gmmFileName', M, L, J);

>> vB = vB.solve(0, nIter);
 
where:
 
pointSetsList:        is the name of a file listing the absolute paths to point sets. Each point
                       set file should define N_k x D (dimensional) points in rows:
                       
x1 y1 z1

x2 y2 z2

.

.

.

gmmFileName:          is an (optional) GMM file defining initialization. This can
                      be empty (i.e. []). In that case, the algorithm will fit a GMM (with M
                      centroids) to the available points (usually slow). You can save your GMM file using
                      writeGmm method provided by the class for a later use.
 
M:                    is the number of Gaussian centroids which define the GMM for each point
                      set. M*dim(points) determines the latent dimension of the MixPPCA sapce.

L:                    is the number of varation modes in each PPCA cluster

J:                    is the number of PPCA clusters

nIter:                is the numer of iterations to solve the mixture. 

After solve() step,  the vB object will store the variation modes, means of PPCA cluster,
point correspondances in its properties: E_W, muBar, E_T, E_Z, etc
where E_{.} stands for expectation. 

You can access the E_Mu's (the projections to PPCA spaces), muBar (PPCA cluster means) using vis_E_Mu
method. For instance:

>> figure, vB.vis_E_Mu(1:vB.J,1:vB.K,[],1,[],0)

You can also visualize the point sets and their clustrs using

>> figure, vB.vis_X([],[])

The 'model evidence' for a particular setting of M, L, and J can be
accessed by vB.LB values, thus faciliating the model selection. LB can
also be used to monitor the convergence. In practice, 

It is recommended that you have access to RAM > 16 GByte, to process a
collection of 50 point sets with 4000 points each. 

This software follows the MIT license prototype.
