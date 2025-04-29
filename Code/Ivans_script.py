# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 13:46:12 2024

@author: ivandep
"""

import pandas as pd
import numpy as np
import openturns as ot
import openturns.viewer as viewer
from matplotlib import pylab as plt
from scipy.stats import norm
from sklearn.cluster import KMeans

#ot.Log.Show(ot.Log.NONE)

np.random.seed(1234)

# Define input variables-------------------------------------------------------

# If Uniform, just use the lower and upper bounds as inputs

# If Normal, 99.7% probability interval, +/- 3*sigma
# mean=(Low+Up)/2
# std=(Up-Low)/2/3

# If Lognormal, 99.7% probability interval, +/- 3*sigma
# Only if both values are greater then 0
# meanLn=[ln(Low)+ln(Up)]/2
# stdLn=[ln(Up)-ln(Low)]/2/3

# Lower and upper bounds for random variables
# Block volume-----------------------------------------------------------------
blockVolumeLow=0.03
blockVolumeUp=3.0

# Normal
muBV=(blockVolumeLow+blockVolumeUp)/2
stdBV=(blockVolumeUp-blockVolumeLow)/2/3

# Logormal
muLnBV=(np.log(blockVolumeLow)+np.log(blockVolumeUp))/2
stdLnBV=(np.log(blockVolumeUp)-np.log(blockVolumeLow))/2/3

# Translational velocity-------------------------------------------------------
tranVelLow=5.0
tranVelUp=40.0

# Normal
muTV=(tranVelLow+tranVelUp)/2
stdTV=(tranVelUp-tranVelLow)/2/3

# Logormal
muLnTV=(np.log(tranVelLow)+np.log(tranVelUp))/2
stdLnTV=(np.log(tranVelUp)-np.log(tranVelLow))/2/3

# Rotational velocity----------------------------------------------------------
rotVelLow=0.001
rotVelUp=35.0

# Normal
muRV=(rotVelLow+rotVelUp)/2
stdRV=(rotVelUp-rotVelLow)/2/3

# Logormal
muLnRV=(np.log(rotVelLow)+np.log(rotVelUp))/2
stdLnRV=(np.log(rotVelUp)-np.log(rotVelLow))/2/3

# Impact Angle-----------------------------------------------------------------
impAngLow=-80.0
impAngUp=45.0

# Normal
muIA=(impAngLow+impAngUp)/2
stdIA=(impAngUp-impAngLow)/2/3

# Logormal
#muLnIA=(np.log(impAngLow)+np.log(impAngUp))/2
#stdLnIA=(np.log(impAngUp)-np.log(impAngLow))/2/3

# Impact position Y------------------------------------------------------------
impPosYLow=-14.18
impPosYUp=14.18

# Normal
muPY=(impPosYLow+impPosYUp)/2
stdPY=(impPosYUp-impPosYLow)/2/3

# Logormal
#muLnPY=(np.log(impPosYLow)+np.log(impPosYUp))/2
#stdLnPY=(np.log(impPosYUp)-np.log(impPosYLow))/2/3

# Impact position Z------------------------------------------------------------
impPosZLow=0.82
impPosZUp=4.5

# Normal
muPZ=(impPosZLow+impPosZUp)/2
stdPZ=(impPosZUp-impPosZLow)/2/3

# Logormal
muLnPZ=(np.log(impPosZLow)+np.log(impPosZUp))/2
stdLnPZ=(np.log(impPosZUp)-np.log(impPosZLow))/2/3

# Define the correlation matrix for the 6 random variables---------------------
R = ot.CorrelationMatrix(6)
R[0,1]=0
R[0,2]=0
R[0,3]=0
R[0,4]=0
R[0,5]=0

R[1,2]=0
R[1,3]=0
R[1,4]=0
R[1,5]=0

R[2,3]=0
R[2,4]=0
R[2,5]=0

R[3,4]=0
R[3,5]=0

R[4,5]=0

# Build normal copula based on the correlation matrix
copula = ot.NormalCopula(R)


# Define the marginal distributions--------------------------------------------
marginals = [ot.LogNormal(muLnBV, stdLnBV, 0.0), 
             ot.LogNormal(muLnTV, stdLnTV, 0.0), 
             ot.Uniform(rotVelLow, rotVelUp),
             ot.Uniform(impAngLow, impAngUp),
             ot.Normal(muPY, stdPY),
             ot.Normal(muPZ, stdPZ)]

# Build the joint distribution
distribution = ot.ComposedDistribution(marginals, copula)
 
# Transformation
T=distribution.getIsoProbabilisticTransformation()
T1=distribution.getInverseIsoProbabilisticTransformation()

# size = 1000
# sample = distribution.getSample(size)

# sample_u=T(sample)

# sample_x=T1(sample_u)

# Reliability analysis---------------------------------------------------------
# Number of random variables
dimension=6

# Failure limit
disLim=0.2

# Refinement-------------------------------------------------------------------
# Maximum number of refinements
nRef=3

# Allocate pf vector
pf=np.zeros(nRef)
pfe=np.zeros(nRef)
alphaCor=np.zeros(nRef)

deltaPf=np.zeros(nRef)
deltaPfe=np.zeros(nRef)
deltaAlphaCor=np.zeros(nRef)

# Number of samples for calculating pfe, 10^6
nSamp=1000000

# Number of ABAQUS analyses in each refinement
nABA=30

# Allocate matrix for correction coefficient
samplesCorrection=np.zeros((nABA,dimension,nRef))
gCorrection=np.zeros((nABA,nRef))

for j in range(nRef):
    # Fit the kriging metamodel
    # Read by default 1st sheet of an excel file and convert to numpy
    analyses = pd.read_excel('Analyses.xlsx').to_numpy()

    # Columns to analyze
    cols=np.array([1,2,3,4,5,6,10])

    # Remove no values data
    vals=np.array(analyses[analyses[:,-1]!='NO VALUE%',:][:,[1,2,3,4,5,6,10]],dtype=float)

    # Number of datapoints
    nData=np.shape(vals)[0]

    # All data are used for trainig
    # Training
    x_train=ot.Sample(vals[:,0:6])
    # Convert to standard normal
    x_train_u=T(x_train)
    # Output
    y_train=ot.Sample(np.reshape(vals[:,6],(nData,1)))
     
    # Dimensions
    dimension=x_train.getDimension()

    # Create a kriging metamodel
    # Constant trend
    basis = ot.ConstantBasisFactory(dimension).build()
    # Covariance model
    covarianceModel = ot.SquaredExponential(dimension)
    # Build the model
    algo = ot.KrigingAlgorithm(x_train_u, y_train, covarianceModel, basis)
    # Fit the model
    algo.run()

    result = algo.getResult()
    
    # Select seeds for Metropolis-Hastings sampling
    # Number of seeds
    nSeed=10
    
    # Split the data into nSeed clusters
    kmeans = KMeans(n_clusters=nSeed).fit(np.array(x_train_u))
    
    # Cluster centers
    centers=kmeans.cluster_centers_
    
    # Number of trial samples per seed
    nTrial=1000

    
    # Allocate matrix
    sampleRef=np.zeros((nSeed*nTrial,dimension))
    gRef=np.zeros(nSeed*nTrial)
    
    # Counter
    countSeed=0
    
    for i in range(nSeed*nTrial):
        if(np.mod(i,nTrial)==0):
            # Copy center value
            sampleRef[i,:]=centers[countSeed,:]
            
            # Calculate pdf in the standard normal space
            pdfCurr=norm.pdf(sampleRef[i,0])*norm.pdf(sampleRef[i,1])*\
                norm.pdf(sampleRef[i,2])*norm.pdf(sampleRef[i,3])*\
                    norm.pdf(sampleRef[i,4])*norm.pdf(sampleRef[i,5])
            
            # Calculate performance function
            gRef[i]=np.array(result.getConditionalMean(sampleRef[i,:]))-disLim
    
            # Calculate standard deviation
            gStdCurr=np.sqrt(result.getConditionalMarginalVariance(sampleRef[i,:]))
    
            # Calculate probability of being in the failure zone
            piCurr=norm.cdf(-gRef[i]/gStdCurr)
    
            # Calculate the product
            hCurr=pdfCurr*piCurr
            
            # Update counter
            countSeed+=1
            
        else:
            # Propose sample
            sampleProp=sampleRef[i-1,:]+np.random.normal(scale=0.1,size=(1,dimension))
            
            # Calculate pdf in the standard normal space
            pdfProp=norm.pdf(sampleProp[0,0])*norm.pdf(sampleProp[0,1])*\
                norm.pdf(sampleProp[0,2])*norm.pdf(sampleProp[0,3])*\
                    norm.pdf(sampleProp[0,4])*norm.pdf(sampleProp[0,5])
                    
            # Calculate performance function
            gProp=np.array(result.getConditionalMean(sampleProp))-disLim
    
            # Calculate standard deviation
            gStdProp=np.sqrt(result.getConditionalMarginalVariance(sampleProp))
    
            # Calculate probability of being in the failure zone
            piProp=norm.cdf(-gProp/gStdProp)
    
            # Calculate the product
            hProp=pdfProp*piProp
            
            # Calculate acceptance probability
            alpha=hProp/hCurr       
            
            # Check acceptance
            if(np.random.uniform()<=alpha):
                # Accept
                sampleRef[i,:]=sampleProp
                # Copy likelihood
                hCurr=hProp
                # Copy performance function
                gRef[i]=gProp
            else:
                # Reject
                sampleRef[i,:]=sampleRef[i-1,:]
                # Copy performance function
                gRef[i]=gRef[i-1]
    
    
    
    # Generate samples from the original pdf
    samplePf = distribution.getSample(nSamp)
    
    # Make metamodel predictions
    g=np.array(result.getConditionalMean(samplePf))-disLim
    
    # Get standard deviation
    gStd=np.sqrt(np.array(result.getConditionalMarginalVariance(samplePf)))
    
    # Calculate augmented failure probability
    pfe[j]=np.sum(norm.cdf(-g/gStd))/nSamp
    
    
    # Perform clustering to prepare inputs for ABAQUS
    # Sort the data close to zero
    idSort=np.argsort(np.abs(gRef))
    
    # Select a subset of the samples for clustering
    sampleRefCluster=sampleRef[idSort[0:np.min([10*nABA,nTrial])],:]
    
    # Split the data into nABA clusters
    kmeans = KMeans(n_clusters=nABA).fit(np.array(sampleRefCluster))
    
    # Cluster centers
    centersCor=kmeans.cluster_centers_
    
    # Convert to the design space
    centersCorDesign=np.array(T1(ot.Sample(centersCor)))
    
    # Perform ABAQUS analyses at "centersCorDesign" and update the Excel file
    
    # Also store the results in the following way
    samplesCorrection[:,:,j]=centersCor
    
    # # The dissipation rates are to be saved in 
    # #gCorrection[:,j]=Abaqus results....
    
    # # Abaqus samples in a vector
    # samplesABA=ot.Sample(np.reshape(samplesCorrection[:,:,0:j+1],(nABA*(j+1),dimension)))
    
    # # Get metamodel predictions at these points
    # gMod=np.array(result.getConditionalMean(samplesABA))-disLim
    
    # # Get standard deviation
    # gModStd=np.sqrt(result.getConditionalMarginalVariance(samplesABA))
    
    # # Reshape g vector
    # ga=np.reshape(gCorrection[:,0:j+1],(nABA*(j+1),1))
    
    # # Calculate correction factor
    # alphaCor[j]=1/((j+1)*nABA)*np.sum((ga<=0)/(norm.cdf(-gMod/gModStd)))
    
    # # Calculate failure probability
    # pf[j]=pfe[j]*alphaCor[j]
    
    
    # Calculate variance of pfe
    VarPfe=1/(nSamp-1)*(1/nSamp*np.sum(norm.cdf(-g/gStd)**2)-pfe[j]**2)
    # Coefficient of variation of Pfe
    deltaPfe[j]=np.sqrt(VarPfe)/pfe[j]
    
    nA='number of refinement samples'
    # Calculate variance of alphaCorr, nA- is the total number of refinement samples
    VarAlphaCorr=1/(nA-1)*(1/nA*np.sum(((ga<=0)/(norm.cdf(-gMod/gModStd)))**2)-alphaCorr[j]**2)
    # Coefficient of variation of the correction factor
    deltaAlphaCor[j]=np.sqrt(VarAlphaCorr)/alphaCorr[j]
    
    # Total coefficient of variation
    deltaPf[j]=np.sqrt(deltaPfe[j]**2+deltaAlphaCor[j]**2)
    
    
    
    
    
    







