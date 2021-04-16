import sys
sys.path.insert(0,"../NEATUtils")
import csv
import os
import cv2
import glob
import shutil
import numpy as np
from tifffile import imread 
from NEATUtils.helpers import save_tiff_imagej_compatible
import math
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
    
    
    
def UpsampleMovies(TargetDir, SourceDir, SizeX, SizeY, AppendName =""):
    
    Subdir = next(os.walk(SourceDir))
    Path(TargetDir).mkdir(exist_ok = True)
    for x in Subdir[1]:
      currentdir = SourceDir + '/' + x
      currentTargetDir = TargetDir + '/' + x
      Path(currentTargetDir).mkdir(exist_ok = True)  
      Raw_path = os.path.join(currentdir,'*.tif')
      Csv_path = os.path.join(currentdir,'*.csv')
      X = glob.glob(Raw_path)
      XCsv = glob.glob(Csv_path)  
      shutil.copyfile(XCsv[0], currentTargetDir + '/' + 'Label' + '.csv')  
      
      for y in X:
         image = imread(y)
         targetimage = np.zeros([image.shape[0], SizeX, SizeY])
         Name = os.path.basename((os.path.splitext(y)[0])) 
         for i in range(0,image.shape[0]):
            
            targetimage[i,:] = cv2.resize(image[i,:], (SizeX, SizeY), interpolation = cv2.INTER_LINEAR)
         save_tiff_imagej_compatible((currentTargetDir + '/' + Name + AppendName + '.tif'  ) , targetimage, 'TYX')

def UpsampleImages(TargetDir, SourceDir, SizeX, SizeY, AppendName =""):
    
    Subdir = next(os.walk(SourceDir))
    Path(TargetDir).mkdir(exist_ok = True)
    for x in Subdir[1]:
      currentdir = SourceDir + '/' + x
      currentTargetDir = TargetDir + '/' + x
      Path(currentTargetDir).mkdir(exist_ok = True)  
      Raw_path = os.path.join(currentdir,'*.tif')
      Csv_path = os.path.join(currentdir,'*.csv')
      X = glob.glob(Raw_path)
      XCsv = glob.glob(Csv_path)  
      shutil.copyfile(XCsv[0], currentTargetDir + '/' + 'Label' + '.csv')  
      
      for y in X:
         image = imread(y)
         targetimage = np.zeros([SizeX, SizeY])
         Name = os.path.basename((os.path.splitext(y)[0])) 
        
            
         targetimage = cv2.resize(image, (SizeX, SizeY), interpolation = cv2.INTER_LINEAR)
         save_tiff_imagej_compatible((currentTargetDir + '/' + Name + AppendName + '.tif'  ) , targetimage, 'YX')
        
        
        
def DownsampleMovies(TargetDir, SourceDir, SizeX, SizeY, AppendName =""):
    
    Subdir = next(os.walk(SourceDir))
    Path(TargetDir).mkdir(exist_ok = True)
    for x in Subdir[1]:
      currentdir = SourceDir + '/' + x
      currentTargetDir = TargetDir + '/' + x
      Path(currentTargetDir).mkdir(exist_ok = True)  
      Raw_path = os.path.join(currentdir,'*.tif')
      Csv_path = os.path.join(currentdir,'*.csv')
      X = glob.glob(Raw_path)
      XCsv = glob.glob(Csv_path)  
      shutil.copyfile(XCsv[0], currentTargetDir + '/' + 'Label' + '.csv')  
      
      for y in X:
         image = imread(y)
         targetimage = np.zeros([image.shape[0], SizeX, SizeY])
         Name = os.path.basename((os.path.splitext(y)[0])) 
         for i in range(0,image.shape[0]):
            
            targetimage[i,:] = cv2.resize(image[i,:], (SizeX, SizeY), interpolation = cv2.INTER_LINEAR)
         save_tiff_imagej_compatible((currentTargetDir + '/' + Name + AppendName + '.tif'  ) , targetimage, 'TYX')

def DownsampleImages(TargetDir, SourceDir, SizeX, SizeY, AppendName =""):
    
    Subdir = next(os.walk(SourceDir))
    Path(TargetDir).mkdir(exist_ok = True)
    for x in Subdir[1]:
      currentdir = SourceDir + '/' + x
      currentTargetDir = TargetDir + '/' + x
      Path(currentTargetDir).mkdir(exist_ok = True)  
      Raw_path = os.path.join(currentdir,'*.tif')
      Csv_path = os.path.join(currentdir,'*.csv')
      X = glob.glob(Raw_path)
      XCsv = glob.glob(Csv_path)  
      shutil.copyfile(XCsv[0], currentTargetDir + '/' + 'Label' + '.csv')  
      
      for y in X:
         image = imread(y)
         targetimage = np.zeros([SizeX, SizeY])
         Name = os.path.basename((os.path.splitext(y)[0])) 
        
            
         targetimage = cv2.resize(image, (SizeX, SizeY), interpolation = cv2.INTER_LINEAR)
         save_tiff_imagej_compatible((currentTargetDir + '/' + Name + AppendName + '.tif'  ) , targetimage, 'YX')
        
        
        
                
        