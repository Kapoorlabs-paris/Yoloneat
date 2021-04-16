from __future__ import print_function, unicode_literals, absolute_import, division
from NEATUtils.helpers import normalizeFloat
from tqdm import tqdm
from tifffile import imread
from glob import glob
import numpy as np
import cv2
import elasticdeform
import random
from NEATUtils.helpers import save_tiff_imagej_compatible
import scipy
import shutil

from scipy import ndimage

try:
    from pathlib import Path
    Path().expanduser()
except (ImportError,AttributeError):
    from pathlib2 import Path

try:
    import tempfile
    tempfile.TemporaryDirectory

except (ImportError,AttributeError):
    from backports import tempfile


class Augmentation2D(object):
    
    """
    Data augmentation for input movies with Gaussian Blur, Random Rotations and optional Deformations
    
    Parameters
    ----------
    
    
    inputdir : input directory of non-augmenqtated images
    
    outputdir : output directory of augmented images
    
    
    resizeX : X dimension of the training image after resizing
    
    resizeY : Y dimension of the training image after resizing
    
    
    elasticDeform : if True then slight elastic deformation would be performed else not
    
    putNoise : if True then gaussian blur at three sigmas would be added and a random rotation on 4 movies
    
    
    """
    
    
    def __init__(self, inputdir, outputdir, resizeX, resizeY, elasticDeform = False, putNoise = False, Rotate = False, AppendName = "_"):
        
        Path(outputdir).mkdir(exist_ok=True)
        self.inputdir = inputdir
        self.outputdir = outputdir
        self.elasticDeform = elasticDeform
        self.putNoise = putNoise
        self.resizeX = resizeX
        self.resizeY = resizeY
        
        self.AppendName = AppendName
        self.Rotate = Rotate
        #Perform tasks
        self.do_augmentation()
        
        
    def do_augmentation(self):
        """
        Performs data augmentation on directory of images and stores result with appropriate name in target directory images
        
        """
    
        HEIGHT = self.resizeX
        WIDTH = self.resizeY
        axes = 'YX'
        
        ImageFiles = sorted(glob(self.inputdir + '/' + '*.tif'))
        CsvFile = sorted(glob(self.inputdir + '/' + '*.csv'))
        #Read in all images
        X = list(map(imread, ImageFiles))
       
        shutil.copyfile(CsvFile[0], self.outputdir + '/' + 'Label' + '.csv')
        
        noisecount = 0
        rotatecount = 0
        deformcount = 0
        origcount = 0
        for n in X:
    
        
          m = np.zeros([HEIGHT, WIDTH])
        
          #Resize all movies   
         
        
          m =    cv2.resize(n, (HEIGHT, WIDTH), interpolation = cv2.INTER_LANCZOS4)
          
          origcount = origcount + 1 
          Path(self.outputdir + '/ResizeOriginal/').mkdir(exist_ok=True)
          save_tiff_imagej_compatible((self.outputdir + '/ResizeOriginal/' + str(origcount) + self.AppendName + '.tif'  ) , m, axes)  
            
          if(self.putNoise):
           noisecount = noisecount + 1
           noisyA = random_noise(m)
           #Make 3 noisy movies for one input movie
 
           Path(self.outputdir + '/Sigma1/').mkdir(exist_ok=True)
           save_tiff_imagej_compatible((self.outputdir + '/Sigma1/' + str(noisecount) + self.AppendName + '.tif'  ) , noisyA, axes)
      
           if(self.Rotate):
             #Make rotations on original and three noisy movies  
             rotate_orig = random_rotation(m)
             rotatecount = rotatecount + 1
              
             Path(self.outputdir + '/RotatedOriginal/').mkdir(exist_ok=True)    
             save_tiff_imagej_compatible((self.outputdir + '/RotatedOriginal/' + str(rotatecount)+ self.AppendName+ '.tif'  ) , rotate_orig, axes)
             if(self.putNoise):
               rotate_noiseA = random_rotation(noisyA)
           
            
               Path(self.outputdir + '/RotatedSigma1/').mkdir(exist_ok=True)   
               save_tiff_imagej_compatible((self.outputdir + '/RotatedSigma1/' + str(rotatecount) + self.AppendName + '.tif'  ) , rotate_noiseA, axes)

          #Do deformations if asked for
          
          if(self.elasticDeform):
              deformcount = deformcount + 1
              deform_orig = random_deform(m)
              Path(self.outputdir + '/DeformedOriginal/').mkdir(exist_ok=True)    
              save_tiff_imagej_compatible((self.outputdir + '/DeformedOriginal/' + str(deformcount) + self.AppendName + '.tif'  ) , deform_orig, axes)
              if(self.putNoise):
               deform_noiseA = random_deform(noisyA)
               if self.Rotate: 
                 deform_rotate_orig = random_deform(rotate_orig)
                 deform_rotate_noiseA = random_deform(rotate_noiseA)

              

               Path(self.outputdir + '/DeformedSigma1/').mkdir(exist_ok=True)   
               save_tiff_imagej_compatible((self.outputdir + '/DeformedSigma1/' + str(deformcount) + self.AppendName + '.tif'  ) , deform_noiseA, axes)
               if self.Rotate:
                Path(self.outputdir + '/DeformedRotatedOriginal/').mkdir(exist_ok=True)    
                save_tiff_imagej_compatible((self.outputdir + '/DeformedRotatedOriginal/' + str(deformcount) + self.AppendName + '.tif'  ) , deform_rotate_orig, axes)
                Path(self.outputdir + '/DeformedRotatedSigma1/').mkdir(exist_ok=True)   
                save_tiff_imagej_compatible((self.outputdir + '/DeformedRotatedSigma1/' + str(deformcount) + self.AppendName + '.tif'  ) , deform_rotate_noiseA, axes)
           
        


    
def random_deform(image, sigma = 1, points = 3):
    
    deformedimage = elasticdeform.deform_random_grid(image,sigma = sigma, points = points, axis = (0,1))
   
    return deformedimage
    
    
def random_rotation(image):
      angle = random.uniform(-2, 2)
      
      
      rotatedimage = scipy.ndimage.interpolation.rotate(image, angle, mode = 'reflect', axes = (1,0), reshape = False)
      return rotatedimage

def random_noise(image, sigmaA = 0.2):
        noisyimageA = ndimage.gaussian_filter(image,sigmaA, mode = 'reflect')
   
              
        return noisyimageA


    
  
        
        
        
        
