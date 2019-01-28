# -*- coding: utf-8 -*-

import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import scipy.ndimage.filters
from scipy.optimize import leastsq
import matplotlib.pyplot as plt


#------------------------------------------------------------------------------
# DAISY correction program.
# Date: January 16, 2019.
# Python version: 2.7.10 (compatible with 32 bits and 64 bits)
# Authors: Clément Cabriel [1], Jean Commère [1], Caroline Schou [2]. Affiliations: [1] Institut des Sciences Moléculaires d'Orsay, CNRS UMR8214, Université Paris-Sud, France; [2] Abbelight, Paris, France.
# Corresponding authors: Clément Cabriel (clement.cabriel@u-psud.fr), Sandrine Lévêque-Fort (sandrine.leveque-fort@u-psud.fr).
# This code is provided as supporting material for the manuscript [Cabriel et al., 'Combining 3D single molecule localization strategies for reproducible bioimaging', bioRxiv 385799 (2018); doi: https://doi.org/10.1101/385799]. Please refer to the manuscript for more details about the content and the use of the code.
# This script contains several functions kindly provided by Caroline Schou from the software team of Abbelight (contact: contact@abbelight.com). These functions were developed following a common discussion on the project between the authors.
# This work is placed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license. Users are free to use the code and to adapt it to their needs for non-commercial uses. The original work must be credited in publications using either the original or modified versions of the code. Should modified versions be redistributed, they must be licensed under the same terms.
#------------------------------------------------------------------------------
# Input format for the data:
# Column-wise: y, x, z_SAF, z_astigmatism, number of the image, width_EPI_y, width_EPI_x (all values in nm except the number of the image)
# Line-wise are all the localizations
# Note: for optimal performance, make sure that pre-filtered data are used. In particular, saturated values should be removed if necessary. We also advise to eliminate unastigmatic PSFs having a width (i.e. the standard deviation of the spatial distribution) below 100 nm or above 300 nm.
#------------------------------------------------------------------------------


class DAISY_correction():
    
    def __init__(self):
        
        # Input Files Data
        self.filepath = u"DAISY_data_test_COS7-alphatubulin-AF647.txt"      # path to the input localization file
        
        # Tilt correction parameters
        self.slopemax=0.01                                  # maximum value of the slope
        self.offsetmax=300.0                                # maximum axial offset value
        self.Niter=4                                        # number of iterations
        self.sampling=20                                    # sampling for each iteration
        
        # Drift correction parameters
        self.imsizeX_nm = 25400                             # image width (nm)
        self.imsizeY_nm = 25400                             # image height (nm)
        self.binsize_lateral = 100                          # lateral bin size (nm)
        self.binsize_axial = 30                             # axial bin size (nm)
        self.slice_size = 400                               # number frames per slice
        
        # Filtering parameters
        self.roi = np.array([0.0, np.inf, 0.0, np.inf])     # ROI boundaries (nm)
        self.zmin = -50                                     # minimum of the z_SAF (nm) values used for the correction
        self.zmax = 300                                     # maximum of the z_SAF (nm) values used for the correction
        
        # Options
        self.parallel=False                                 # =True to parallelize the calculations on the CPU, =False otherwise. Note that parallelization increases the speed, but might cause the calculation to crash on some computers for unknown reasons. Note also that GPU parallelization is not used in this code.
        self.plt_ON = False                                 # =True to display the plots, =False not to display them
        self.save_ON = True                                 # =True to save the corrected localization output files, =False not to save them
        
    
    def Correction_main(self):

        if self.parallel==True:
            self.NUM_CORES = multiprocessing.cpu_count()
        else:
            self.NUM_CORES = 1
        
        print ''
        print("Loading files...")
        filedata = np.loadtxt(self.filepath)
        coords = np.array(filedata)
        
        print ''
        print("Filtering...")
        coords_fil = self.filter_data(coords, self.zmin, self.zmax, self.roi)
        
        print ''
        print("Correcting tilt...")
        parammin=self.plane_fit(coords_fil)
        coords_fil = self.correct_tilt(coords_fil, parammin);
        coords = self.correct_tilt(coords, parammin);
        if self.save_ON:
            with open('Tilt.txt', 'w') as outfp:
                outfp.write('#Slope  Y (dZ/dY)    :\n'+str(self.parammin[0])+'\n')
                outfp.write('#Slope  X (dZ/dX)    :\n'+str(self.parammin[1])+'\n')
                outfp.write('#Offset Z (Z_0 in nm):\n'+str(self.parammin[2]))
        
        print ''
        print("Correcting drift...")
        nb_frames=np.max(coords[:,4])-np.min(coords[:,4])
        nb_seg=int(np.ceil(nb_frames/self.slice_size))
        drift, imin, imax = self.compute_drift_corr(coords_fil, nb_seg)
        coords_new = self.correct_drift(coords, drift, imin, imax)
        if self.save_ON:
            np.savetxt(self.filepath[:-4]+"_DAISYcorrected.txt", coords_new)
        
        print ''
        print("Saving data...")
        if self.plt_ON or self.save_ON:
            plt.figure("DriftZ")
            plt.plot(drift)
            plt.xlabel('Frame number')
            plt.ylabel('Axial drift (nm)')
            if self.save_ON:
                driftoutput=np.zeros((np.shape(drift)[0],2))
                driftoutput[:,0]=np.arange(np.shape(drift)[0])
                driftoutput[:,1]=drift
                np.savetxt("DriftZ.txt",driftoutput)
            if self.plt_ON:
                plt.show()
        
        print ''
        print("Done")
    
    
    def correct_tilt(self, coords, parammin):
        # The optimal plane found is subtracted from the astigmatic positions
       
        def plane_function((Y,X),dy,dx,z0):
            return dy*Y+dx*X+z0
    
        coords_new = coords.copy();
        y,x=coords[:,0]-np.max(coords[:,0])/2.0,coords[:,1]-np.max(coords[:,1])/2.0
        
        zinterp=plane_function((y,x),parammin[0],parammin[1],parammin[2])
        coords_new[:,3] += zinterp;
        return coords_new;
        
        
    def plane_fit(self,coords):
        # The tilt is measured using an iterative method: the parameters (y slope, x slope, z offset) are scanned over a broad range with a rough precision and the global optimum is found using a minimization of the median of the absolute deviations (rather than the mean of the square deviations). At each iteration, the precision is improved and the parameters are scanned around the previoulsy found optimal parameter set.
        
        def plane_function((Y,X),dy,dx,z0):
            return dy*Y+dx*X+z0
        
        y,x=coords[:,0]-np.max(coords[:,0])/2.0,coords[:,1]-np.max(coords[:,1])/2.0
        coords[:,3]+=np.median(coords[:,2]-coords[:,3])
        
        limitesparam=[[-self.slopemax,self.slopemax],[-self.slopemax,self.slopemax],[-self.offsetmax,self.offsetmax]]
        absc=np.linspace(-1.0,1.0,self.sampling)
        cptiter=0
        
        for j in np.arange(self.Niter):
            cptiter+=1
            print '    Iteration '+str(cptiter)+' / '+str(self.Niter)
            
            errors=np.zeros((self.sampling,self.sampling,self.sampling))
            slopeY=np.linspace(limitesparam[0][0],limitesparam[0][1],self.sampling)
            slopeX=np.linspace(limitesparam[1][0],limitesparam[1][1],self.sampling)
            offsetZ=np.linspace(limitesparam[2][0],limitesparam[2][1],self.sampling)
            
            # The operations are divided and sent to all the threads of the CPU to improve the speed.
            def do_k(k):
                for l in np.arange(np.shape(absc)[0]):
                    for m in np.arange(np.shape(absc)[0]):
                        zplan=plane_function((y,x),slopeY[k],slopeX[l],offsetZ[m])
                        errors[k,l,m]=np.median(np.abs(coords[:,2]-coords[:,3]-zplan))
            Parallel(n_jobs=self.NUM_CORES,backend="threading")(delayed(do_k)(i) for i in np.arange(np.shape(absc)[0]));
            
            errors=scipy.ndimage.filters.gaussian_filter(errors,1.0)
            posmin=np.unravel_index(np.argmin(errors, axis=None), errors.shape)
            parammin=[slopeY[posmin[0]],slopeX[posmin[1]],offsetZ[posmin[2]]]
            zinterp=plane_function((y,x),parammin[0],parammin[1],parammin[2])
            
            spacingparam=[slopeY[1]-slopeY[0],slopeX[1]-slopeX[0],offsetZ[1]-offsetZ[0]]
            limitesparam=[[parammin[0]-2.0*spacingparam[0],parammin[0]+2.0*spacingparam[0]],[parammin[1]-2.0*spacingparam[1],parammin[1]+2.0*spacingparam[1]],[parammin[2]-2.0*spacingparam[2],parammin[2]+2.0*spacingparam[2]]]
                
            print '    Optimal parameters:'
            print '        '+'Slope  Y: '+str(parammin[0]*1000.0)[:5]+' .10^-3'
            print '        '+'Slope  X: '+str(parammin[1]*1000.0)[:5]+' .10^-3'
            print '        '+'Offset Z: '+str(parammin[2])[:5]+' nm'
            print '    Minimum and maximum values of the fitted plane over the field:'
            print '        '+str(int(np.min(zinterp)))+'nm, '+str(int(np.max(zinterp)))+'nm'
            print ''
        
        self.parammin=parammin
        
        return parammin
        
        
    def filter_data(self,coords, zmin, zmax, roi):
        # Filtering over the ROI, the z_SAF and the z_astigmatism positions
        
        nb = coords.shape[0]
        z_cyl_max = np.max(coords[:,3]) - 10
        z_cyl_min = np.min(coords[:,3]) + 10
        mask = (coords[:,0] > roi[0]) & (coords[:,0] < roi[1]) & (coords[:,1] > roi[2]) & (coords[:,1] < roi[3]) & (coords[:,2] > zmin) & (coords[:,2] < zmax) & (coords[:,3] > z_cyl_min) & (coords[:,3] < z_cyl_max)
        coords_fil = coords[mask]
        nb_n = coords_fil.shape[0]
        print("    Keeping "+str(nb_n)+" of "+str(nb)+" localizations")
        return coords_fil
        
    
    def compute_drift_corr(self,coords, nb_seg):
        # The acquisition is divided in temporal slices, and for each slice, the z_SAF and z_astigmatism positions are cross-correlated to find the optimal axial translation to be applied to the z_astigmatism. By stacking the results obtained with all the slices, the drift curve over the whole dataset is generated.
        
        self.zrange = 1000
        l = np.linspace(np.min(coords[:,4]),np.max(coords[:,4]), nb_seg)
        coords_zcyl = np.zeros((coords.shape[0],3))
        coords_zcyl[:,0] = coords[:,0]
        coords_zcyl[:,1] = coords[:,1]
        coords_zcyl[:,2] = coords[:,3]
        coords_zsaf = np.zeros((coords.shape[0],3))
        coords_zsaf[:,0] = coords[:,0]
        coords_zsaf[:,1] = coords[:,1]
        coords_zsaf[:,2] = coords[:,2]
        
        coords_zcyl[:,2] += self.zrange/2.0 - (self.zmax-self.zmin)/2.0
        coords_zsaf[:,2] += self.zrange/2.0 - (self.zmax-self.zmin)/2.0
        
        # The operations are divided and sent to all the threads of the CPU to improve the speed.
        def do_i(li, lip,i):
            k = 0
            while (coords[k,4]<li):
                k += 1
            start = k
            while (coords[k,4]<lip):
                k += 1
            stop = k
            driftZ = self.AxialShiftCorrect(coords_zsaf[start:stop,:], coords_zcyl[start:stop,:], self.imsizeX_nm, self.imsizeY_nm, self.zrange, 1, self.binsize_lateral, self.binsize_axial, False)
            print '    Slice '+str(i+1)+' / '+str(l.shape[0]-1)+' ( '+str(int(100*(i+1)/(l.shape[0]-1)))+' % )'
            return driftZ
        R = Parallel(n_jobs=self.NUM_CORES,backend="threading")(delayed(do_i)(l[i], l[i+1],i) for i in range(l.shape[0]-1))
        
        R=scipy.ndimage.filters.gaussian_filter(R,1.0)
        tab_drift = np.zeros((np.int(np.max(coords[:,4])+1)))
        shift = np.int(l[1]-l[0])
        for k in range(0, len(R)-1):
            imin = np.int(l[k]+(l[k+1]-l[k])/2)
            imax = np.int(l[k+1]+(l[k+2]-l[k+1])/2)
            slope = (R[k+1]-R[k])/np.float(shift)
            for i in range(imin, imax):
                tab_drift[i] = R[k] + slope*(i-imin)
        imin = np.int(l[0]+(l[1]-l[0])/2)
        imax = np.int(l[len(R)-1]+(l[len(R)]-l[len(R)-1])/2)
        tab_drift[:imin]=tab_drift[imin]
        tab_drift[imax:]=tab_drift[imax-1]
        return tab_drift, imin, imax
        
    
    def correct_drift(self,coords, drift, imin, imax):
        # This function applies the drift results to the data
        coords_new = coords.copy()
        nb_min = 0
        nb_max = 0
        for i in range(coords_new.shape[0]):
            nb_max += 1
            if (coords_new[i,4]<=imin):
                nb_min += 1
            elif (coords_new[i,4]>=imax):
                break
        for i in range(nb_min, nb_max):
            if (coords_new[i,4]>=imin and coords_new[i,4]<=imax):
                coords_new[i,3] = coords_new[i,3] - drift[np.int(coords_new[i,4])]
        print("    Keeping "+str(nb_max-nb_min)+" of "+str(coords_new.shape[0])+" points")
        return coords_new[nb_min:nb_max,:]
    
    
#------------------------------------------------------------------------------
# The following functions are kindly provided by the software team of Abbelight. These functions were developed following a common discussion on the project between the authors.
    
    def AxialShiftCorrect(self, coords1, coords2, imsizeX, imsizeY, zrange, pixelsize, binsize_lateral, binsize_axial, plot):
        # This function returns the axial shift of one dataset relative to the other.
        
        zoomfactor_lateral = pixelsize/float(binsize_lateral)
        zoomfactor_axial   = 1/float(binsize_axial)
                                
        im1 = self.BinLocalizations3D(coords1, imsizeX, imsizeY, zrange, zoomfactor_lateral, zoomfactor_axial)
        im2 = self.BinLocalizations3D(coords2, imsizeX, imsizeY, zrange, zoomfactor_lateral, zoomfactor_axial)
        if plot:
            plt.subplot(211)
            plt.imshow(im1[:,100,:], cmap="hot")
            plt.subplot(212)
            plt.imshow(im2[:,100,:], cmap="hot")
            plt.show()

        # Auto-Correlation
        imsizeY, imsizeX, imsizeZ = im1.shape[:3]
        autocorrfunc = []
        for i in range(-imsizeZ/2, imsizeZ/2+1) :
            Tz = i
            im1_translation = np.roll(im1, Tz, axis=2)
            m = np.mean(im1 * im1_translation)
            autocorrfunc.append(m)

        # Cross-Correlation
        imsizeY, imsizeX, imsizeZ = im2.shape[:3]
        crosscorrfunc = []
        for i in range(-imsizeZ/2, imsizeZ/2+1) :
            Tz = i
            im2_translation = np.roll(im2, Tz, axis=2)
            m = np.mean(im1 * im2_translation)
            crosscorrfunc.append(m)

        autocorrfunc = np.asarray(autocorrfunc)
        crosscorrfunc = np.asarray(crosscorrfunc)

        autocorrfitcenter   = self.GaussianFit1D(autocorrfunc)
        crosscorrfitcenter  = self.GaussianFit1D(crosscorrfunc)
        driftZ = autocorrfitcenter - crosscorrfitcenter

        return driftZ/zoomfactor_axial
    
    
    def BinLocalizations3D(self, positions, imsizeX, imsizeY, zrange, zoomfactor_lateral, zoomfactor_axial):
        # This function is used to generate the 3D arrays containing the localization positions by distributing the localizations in the corresponding pixels.
        
        positions[:,0:2] = positions[:,0:2] * zoomfactor_lateral
        positions[:,2]   = positions[:,2]   * zoomfactor_axial
        
        ## Filter localizations outside the FOV
        binimsizeX = np.ceil(imsizeX    * zoomfactor_lateral)
        binimsizeY = np.ceil(imsizeY    * zoomfactor_lateral)
        binimsizeZ = np.ceil(zrange * zoomfactor_axial)

        keepXmin = positions[:,0] >= 0
        keepYmin = positions[:,1] >= 0
        keepZmin = positions[:,2] >= 0
        keepXmax = positions[:,0] <= binimsizeX-1
        keepYmax = positions[:,1] <= binimsizeY-1
        keepZmax = positions[:,2] <= binimsizeZ-1
        
        keep1 = keepYmin & keepXmin & keepZmin & keepYmax & keepXmax & keepZmax
        keep1 = keep1.nonzero()[0]
        positions = positions[keep1,:]
        
        ##Bin localizations
        imbin = np.zeros((int(binimsizeY), int(binimsizeX), int(binimsizeZ)))
        
        for i in range(np.shape(positions)[0]):
            
            x = np.round(positions[i,0])
            y = np.round(positions[i,1])
            z = np.round(positions[i,2])

            imbin[int(y),int(x),int(z)] = imbin[int(y),int(x),int(z)] + 1

        # Apply Gaussian Filter
        sigma = 2.;
        imbin = scipy.ndimage.filters.gaussian_filter(imbin, sigma)
        return imbin
    
    
    def GaussianFit1D(self,corrfunc):
        # Gaussian fitting used to find the optimal position with a sub-pixel precision
        
        N = corrfunc.size
        radiu = round(N/4.)
        center = int(N/2.)
        fitdata = corrfunc[center-int(radiu):center+int(radiu)+1]

        X = np.arange(np.shape(fitdata)[0])
                
        xoffset = center-radiu

        guess1D = np.zeros(4)
        guess1D[0] = corrfunc[center]
        guess1D[1] = 5
        guess1D[2] = radiu
        guess1D[3] = np.mean(corrfunc)
        
        ErrorFunc = lambda param, x, data: (self.Gaussmodel1D(x, param).ravel() - data.ravel())
        [opt,norm] = leastsq(ErrorFunc, guess1D, args=(X, fitdata), maxfev=100)

        x = opt[2] + xoffset

        return x
    
    
    def Gaussmodel1D(self,X,a):
        
        F=a[0]*np.exp(-((X-a[2])**2)/(a[1]**2)) + a[3]
        
        return F.ravel()
    
#------------------------------------------------------------------------------
    
    
DAISY_correction=DAISY_correction()
DAISY_correction.Correction_main()
