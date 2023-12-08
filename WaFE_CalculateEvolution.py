#2023-12-08 Fabian Reuter and Claus-Dieter Ohl
#
#calculates WaFE, the temporal Huygens-evolution of an intensity image
#Provide parameters starting at line 345
#
#
#please cite as:
# 
# "Shockwave Velocimetry Using Wave-based Image Processing to Measure Anisotropic Shock Emission" by
#Fabian Reuter, Jaka Mur, Jaka Petelin, Rok Petkovsek, and Claus-Dieter Ohl
# submitted to Physics of Fluids 2023
# doi: tbf
#

Version_WaFE_CalculateEvolution="2023-12-08"

import numpy as np
import skimage
from skimage.color import rgb2gray


class WaveSolverIn(object):
    
    """
    WaveSolver       Class calculates the wave equation in a homogeneous medium 
    nx:              size of square calculation grid
    CFL:             CFL parameter dt*c/dx should be smaller than 1/sqrt(2)
    sim_duration:    duration of simulation. Should be larger than t_id*c1/2 (where t_id = interpulse delay and factor 0.5 because numerical wave fronts meet in the space between a pair of two wave fronts)
    
    pml_length:     At the image boundaries reflections occur - these artefacts are largely supressed using by padding a frame around the initial condition image. pml_length defines the frame size. Reasonable value should be larger than sim_duration/2*c1

    plotcallback:    callback function for plotting/saving the result the callback receives two parameters self.u and time. Use "savewave" to save image files of the wave evolution, "plotwave" to plot the wave evolution using python or "dummywave" to do nothing
    OutputStep:          generate output every OutputStep-th computation step (OutputStep=1 maximum)
    rho1, c1:        density and speed of sound - set here to the material paramaters of water to achieve physically intuitive numbers (rho=1000 kg/m^3, c=1483 m/s)
    
    UpScaFak:       Geometric upscaling factor using for the internal grid (value of 3 is reasonable, large values significantly affect memory)
    
    InitialConditionsFile: Path to the image on which the wave evolution is calculated
        
    uHistoryDownscaled: uHistory is of shape (NumOf_TimeSteps,NumOf_yDataPoints,NumOf_xDataPoints) and contains the intensity values over time. To save memory uHistoryDownscaled=True allows to save uHistory at the resolution of the InitialConditionsFile
    
    px2m: conversion factor to convert 1 pixel to 1 m 
    
    Following the numerical scheme derived in http://arxiv.org/pdf/1001.0319v1.pdf
    Marcus J. Grote
    """
    
    def __init__(self, nx=64, ny=None, px2m=1, CFL=0.1,sim_duration=2., OutputStep=1, plotcallback=None,pml_length=16,\
                 rho1=1000, c1=1483, InitialConditionsFile=None,UpScaFak=1,uHistoryDownscaled=True):
        import math
        
        self.__UpScaFak=UpScaFak

        #load image as initial condition:
        self.__InitialConditionsFromFile(InitialConditionsFile)
        print(np.shape(self.__u))
        self.__nx = np.shape(self.__u)[0]
        self.__ny = np.shape(self.__u)[1]
        self.__ICFileName=InitialConditionsFile

        #pad frame to reduce reflection at domain boundaries
        self.__u=np.pad(self.__u,mode='linear_ramp',end_values=np.median(self.__u.flatten()),pad_width=pml_length)
        self.__pml_length=pml_length
        self.__nx=self.__nx+2*self.pml_length
        self.__ny=self.__ny+2*self.pml_length
        
        self.__un=self.__u #amplitude at t-dt
        self.__unn=self.__u #amplitude at t-2*dt
        self.__px2m=px2m
        self.__size =px2m*self.__nx
        self.__dx = self.__size/(self.__nx-1)
        self.__c1 = c1
        self.__rho1 = rho1
        self.__sim_duration = sim_duration
        self.__rho = np.ones((self.__nx,self.__ny))*self.__rho1
        self.__cc = np.ones((self.__nx,self.__ny))*self.__c1*self.__c1
        self.CFL = CFL
        
        self.OutputStep = OutputStep
        self.__n = 0
       
        if plotcallback!=None:
            self.__plotcallback = plotcallback
        else:
            self.__plotcallback = self.simplecallback

        self.__timestepper = self.__inhomogeneous_stable
        
     
        #__nt is determined in CFL
        
        self.__uHistoryDownscaled=uHistoryDownscaled
        if uHistoryDownscaled:
            self.uHistory=np.full([self.__nt,round((self.__nx-2*self.__pml_length)/self.__UpScaFak),round((self.__ny-2*self.__pml_length)/self.__UpScaFak)], np.nan)
        else:
            self.uHistory=np.full([self.__nt,round((self.__nx-2*self.__pml_length)),round(self.__ny-2*self.__pml_length)], np.nan)
    

    def __InitialConditionsFromFile(self,FileName):
        import numpy as np
        import cv2
        import os,glob
            
        print("Opening: ",FileName)
        img = skimage.io.imread(FileName)
        if np.ndim(img)>2:
            img=rgb2gray(img)
        img = skimage.img_as_float(img)

        if self.__UpScaFak!=1:
            print("...Upscaling image...")
            img=cv2.resize(img, dsize=(img.shape[1]*self.__UpScaFak,img.shape[0]*self.__UpScaFak), interpolation=cv2.INTER_LANCZOS4) #Lanczos typically yields slightly better results than bicubic

        DrawFinalICImage=False
        if DrawFinalICImage:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.pcolor(1-img,cmap='gray_r')
            ax.axis('equal')
            plt.show()
        self.__u=(1-img)
                
    def __inhomogeneous_stable(self):
        rho = self.__rho
        un = self.__un
        unn = self.__unn
        cc = self.__cc
        c=self.__c1
        dx = self.__dx
        dt = self.__dt
        cconst = dt*dt/dx/dx
        self.__u[1:-1,1:-1]=2.*un[1:-1,1:-1]-unn[1:-1,1:-1]+cconst*c*c*(un[1:-1,:-2]+un[:-2,1:-1]+un[2:,1:-1]+un[1:-1,2:]-4.*un[1:-1,1:-1])


    def simplecallback(self,u,t):
        """Prints the current time"""
        print("time {0:.2}".format(t))
    
    @property
    def u(self):
        return self.__u
    
    @property
    def UpScaFak(self):
        """Wave on the grid"""
        return self.__UpScaFak
    
    @property
    def uHistoryDownscaled(self):
        return self.__uHistoryDownscaled
    
    @property
    def pml_length(self):
        """size of PML layer"""
        return self.__pml_length
    
    @property
    def px2m(self):
        """size of PML layer"""
        return self.__px2m
     
    @property
    def CFL(self):
        """CFL parameter (see class decription)"""
        return self.__CFL
    
    @property
    def dt(self):
        """Length of each timestep"""
        return self.__dt

    @property
    def nx(self):
        """X-size of the computational domain """
        return self.__nx
 
    @property
    def ICFileName(self):
        return self.__ICFileName
 
    @property
    def ny(self):
        """Y-size of the computational domain """
        return self.__ny
    
    @property
    def nu(self):
        """Frequency"""
        return self.__nu
    
    @property
    def omega(self):
        """Angular frequency"""
        return 2.*self.__nu*np.pi
    
    @property
    def nt(self):
        """Number of timesteps"""
        return int(self.__sim_duration/self.__dt)
    
    
    @property
    def c1(self):
        """Background speed of sound"""
        return self.__c1
    
    @property
    def nu(self):
        """Frequency"""
        return self.__nu
    
    @property
    def OutputStep(self):
        """List of output timesteps using the plotcallback function"""
        return self.__OutputStep
    
    @OutputStep.setter
    def OutputStep(self, OutputStep):
        self.__OutputStep=max(OutputStep,1)
            
    @CFL.setter
    def CFL(self, CFL):
        self.__CFL = CFL  #CFL number < 1/sqrt(2)
        self.__dt = CFL*self.__dx/self.__cc.max()**.5
        self.__nt = int(self.__sim_duration/self.__dt)
        print("Time step dt = ",self.__dt)
        print("Number of time steps nt = ",self.__nt)
    
        
              
    def c(self):
        rho = self.__rho
        un = self.__un
        unn = self.__unn
        cc = self.__cc
        dx = self.__dx
        dt = self.__dt
        cconst = dt*dt/dx/dx
        self.__u[1:-1,1:-1] = 2.*un[1:-1,1:-1]-unn[1:-1,1:-1]+cconst*(cc[1:-1,1:-1]*\
        (un[:-2,1:-1]+un[2:,1:-1]+un[1:-1,:-2]+un[1:-1,2:]-4.*un[1:-1,1:-1])-\
        .25*rho[1:-1,1:-1]*((rho[2:,1:-1]-rho[:-2,1:-1])*(un[2:,1:-1]-un[:-2,1:-1])+\
        (rho[1:-1,2:]-rho[1:-1,:-2])*(un[1:-1,2:]-un[1:-1,:-2])))
        
        
    def __timestep(self):

            #One time step        
            self.__timestepper()
            
            #Impose reflective b.c. at boundary                 
            self.__u[0,:] = self.__u[1,:]
            self.__u[-1,:] = self.__u[-2,:]
            self.__u[:,0] = self.__u[:,1]
            self.__u[:,-1] = self.__u[:,-2]

            #save values for the time derivative 
            self.__unn = self.__un.copy() #n-1 time step
            self.__un = self.__u.copy()   #n time step

    
    def solvestep(self):
        """
        Solves the wave PDE
        Function returns False as long as t < than sim_duration
        """
        
        import cv2
        from pathlib import Path
        
        if self.__n>=self.__nt:
            return True
        
        if self.__n%100==0:
            print("TimeStep ",self.__n," / ",self.__nt)
        for self.__n in range (self.__n,self.__n+self.__OutputStep):
            self.__timestep()
            tmp=self.__u[self.pml_length:-self.pml_length,self.pml_length:-self.pml_length]
            if self.uHistoryDownscaled:
                self.uHistory[self.__n,:,:]=cv2.resize(tmp, dsize=(round(tmp.shape[1]/self.__UpScaFak),round(tmp.shape[0]/self.__UpScaFak)), interpolation=cv2.INTER_LANCZOS4)
            else:
                self.uHistory[self.__n,:,:]=tmp
        self.__plotcallback(self.__u[self.pml_length:-self.pml_length,self.pml_length:-self.pml_length],self.__n*self.__dt,Path(self.ICFileName).stem)
        self.__n=self.__n+1

        return False
    

def savewave(u,time,SaveFileNamePrefix):
    import matplotlib.pyplot as plt #plotting
    import os
    
    global PlotNo
    
    # time could be used for a time stamp
    
    DriveLetter="C:/"
    SavePath="WaFE\SyntheticImages/"
    FullSavePath=DriveLetter+SavePath+SaveFileNamePrefix

    try:
        PlotNo=PlotNo+1
    except:
        PlotNo=0
        try:
            os.mkdir(DriveLetter+SavePath+"/"+SaveFileNamePrefix)
        except OSError as error:
            print(error)   
            
    plt.imsave(fname=FullSavePath+"/"+SaveFileNamePrefix+"_"+str(PlotNo)+".png", arr=u, cmap='gray_r', format='png',dpi=1)

    
def dummywave(u,time,SaveFileNamePrefix):
    pass


def plotwave(u,time,SaveFileNamePrefix):
    import platform
    import matplotlib.pyplot as plt
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    global PlotNo
    try:
        PlotNo=PlotNo+1
    except:
        PlotNo=0
    
    plt.ioff()        
    plt.figure(figsize=(8, 8))
    ax=plt.gca()
    CropScaleToImproveContrast=False
    if CropScaleToImproveContrast:
        plt.imshow(u, vmax=np.quantile(u.flatten(),0.95), cmap='gray_r', vmin=np.quantile(u.flatten(),0.05))
    else:
        plt.imshow(u, vmax=max(u.flatten()), cmap='gray_r', vmin=min(u.flatten()))
    plt.close()


def CallSolvewave():
    import pickle
    import os,glob
    from pathlib import Path
    from ttictoc import tic,toc

    ################################################################################
    ######################### SET PARAMETERS HERE ##################################
    ################################################################################
    
    DriveLetter="C:/"
    
    FilePath=DriveLetter+"WaFE\SyntheticImages/"

    ImageFileExtension="tiff"
    FileLoop=False #Evaluates all image files with extension FileLoopFileExtension in the folder 
    
    SingleFileName='ArtificialTwoCircles_R1_20_R2_30_Sigma_1' #file name of image to evaluate. Is ignored is FileLoop=True
    
    UpScaFak=3 #geometric upscaling factor for internal mesh size, default = 3
    SaveuHistoryInSourceImageResolution=False #allows to save the time wave propagation intensity images in original image resolution (i.e. internal mesh grid is downscaled by UpScaFak) to save memory

    px2mHere=1e-6/UpScaFak #put scale factor that converts pixel to meters here and divide by UpScaFak
        
    InterpulseDelay=6.66667e-9 #is used to estimate the necessary wave front simulation duration to match the wave fronts
    NumOfPixelToPropagate=1484*1.2*InterpulseDelay/px2mHere/2 #estimate of the necessary wave front evolution time - factor 1.2 accounts for increases shock velocity
    
    SaveImageFiles=False #allows to save images of wave front evolution 
    
    ################################################################################
    ######################### SET PARAMETERS END ###################################
    ################################################################################

    if FileLoop:
        FileNameList=glob.glob(os.path.join(FilePath,'*.'+ImageFileExtension))
        print(str(len(FileNameList))+" IC-Bilder gefunden in "+os.path.join(FilePath,'*.'+ImageFileExtension))
    else:
        FileNameList=[os.path.join(FilePath,(SingleFileName+'.'+ImageFileExtension))]
    
    for iFileNo in range(len(FileNameList)):
        ThisFileName=os.path.basename(FileNameList[iFileNo])
        ThisFullFileName=FileNameList[iFileNo]
        iStrIndex=len(ThisFileName)
        try:
            del a #free mem of previous wavesolver object
        except:
            pass
    
        CFLNumber=0.15 #0.025 - 0.05 are usually sufficient but require a lot of memory
        if SaveImageFiles:
            a=WaveSolverIn(CFL=CFLNumber, px2m=px2mHere, plotcallback=savewave,pml_length=round(NumOfPixelToPropagate/2),sim_duration=NumOfPixelToPropagate*px2mHere/1483,InitialConditionsFile=ThisFullFileName,UpScaFak=UpScaFak,uHistoryDownscaled=SaveuHistoryInSourceImageResolution)
        else:
            a=WaveSolverIn(CFL=CFLNumber,px2m=px2mHere, plotcallback=dummywave,pml_length=round(NumOfPixelToPropagate/2),sim_duration=NumOfPixelToPropagate*px2mHere/1483,InitialConditionsFile=ThisFullFileName,UpScaFak=UpScaFak,uHistoryDownscaled=SaveuHistoryInSourceImageResolution)
        

        while True:
            #tic()
            if a.solvestep():
                break
            #print(toc())

        uHistory=a.uHistory
        if a.uHistoryDownscaled:
            px2m=a.px2m*UpScaFak
            UpScaFakTemporal=UpScaFak
            UpScaFakToSave=1
        else:
            UpScaFakTemporal=UpScaFak
            UpScaFakToSave=UpScaFak
            px2m=a.px2m

        pickle.dump([uHistory,ThisFullFileName,a.nt,a.dt,px2m,a.OutputStep,UpScaFakToSave,UpScaFakTemporal,Version_WaFE_CalculateEvolution], open(os.path.splitext(Path(a.ICFileName))[0]+"_AuswPickled.p", "wb"))
        print("Saved in ",os.path.splitext(Path(a.ICFileName))[0],"_Pickled.p")
        os.path.splitext(Path(a.ICFileName))[0]
        Path(a.ICFileName).with_name('setup.py') 
   
        
#main
CallSolvewave()
print("Finished.")
