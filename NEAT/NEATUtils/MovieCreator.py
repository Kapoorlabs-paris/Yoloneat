import sys
sys.path.append("../NEAT")
import csv
import numpy as np
from tifffile import imread, imwrite 
import pandas as pd
import os
import glob
from skimage.measure import regionprops
from skimage import measure
from scipy import spatial 
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
from .helpers import  normalizeFloatZeroOne
    
"""
@author: Varun Kapoor
In this program we create training movies and training images for ONEAT. The training data comprises of images and text labels attached to them.
TrainingMovies: This program is for action recognition training data creation. The inputs are the training image, the corresponding integer labelled segmentation image,
csv file containing time, ylocation, xlocation, angle (optional)
Additional parameters to be supplied are the 
1) sizeTminus: action events are centered at the time location, this parameter is the start time of the time volume the network carved out from the image.
2) sizeTplus: this parameter is the end of the time volume to be carved out from the image.
3) total_categories: It is the number of total action categories the network is supposed to predict, Vanilla ONEAT has these labels:
   0: NormalEvent
   1: ApoptosisEvent
   2: DivisionEvent
   3: Macrocheate as static dynamic event
   4: Non MatureP1 cells as static dynamic event
   5: MatureP1 cells as static dynamic event
    
TrainingImages: This program is for cell type recognition training data creation. The inputs are the trainng image, the corresponding integer labelled segmentation image,
Total categories for cell classification part of vanilla ONEAT are:
    0: Normal cells
    1: Central time frame of apoptotic cell
    2: Central time frame of dividing cell
    3: Macrocheates
    4: Non MatureP1 cells
    5: MatureP1 cells
csv file containing time, ylocation, xlocation of that event/cell type
"""    
    
def MovieLabelDataSet(image_dir, seg_image_dir, csv_dir, save_dir, static_name, static_label, csv_name_diff, crop_size, gridx = 1, gridy = 1, offset = 0, yolo_v0 = True):
    
    
            raw_path = os.path.join(image_dir, '*tif')
            Seg_path = os.path.join(seg_image_dir, '*tif')
            Csv_path = os.path.join(csv_dir, '*csv')
            files_raw = glob.glob(raw_path)
            files_raw.sort
            filesSeg = glob.glob(Seg_path)
            filesSeg.sort
            filesCsv = glob.glob(Csv_path)
            filesCsv.sort
            Path(save_dir).mkdir(exist_ok=True)
            total_categories = len(static_name)
            
            for csvfname in filesCsv:
              count = 0  
              print(csvfname)
              Csvname =  os.path.basename(os.path.splitext(csvfname)[0])
            
              for fname in files_raw:
                  
                 name = os.path.basename(os.path.splitext(fname)[0])   
                 for Segfname in filesSeg:
                      
                      Segname = os.path.basename(os.path.splitext(Segfname)[0])
                        
                      if name == Segname:
                          
                          
                         image = imread(fname)
                         segimage = imread(Segfname)
                         for i in  range(0, len(static_name)):
                             event_name = static_name[i]
                             trainlabel = static_label[i]
                             if Csvname == csv_name_diff + name + event_name:
                                            dataset = pd.read_csv(csvfname)
                                            if len(dataset.keys() >= 3):
                        
                                                time = dataset[dataset.keys()[0]][1:]
                                                y = dataset[dataset.keys()[1]][1:]
                                                x = dataset[dataset.keys()[2]][1:]
                                                angle = np.full(time.shape, 2)                        
                                            if len(dataset.keys() > 3):
                                                
                                                angle = dataset[dataset.keys()[3]][1:]                          
                                            #Categories + XYHW + Confidence 
                                            for t in range(1, len(time)):
                                               MovieMaker(time[t], y[t], x[t], angle[t], image, segimage, crop_size, gridx, gridy, offset, total_categories, trainlabel, name + event_name + str(count), save_dir,yolo_v0)
                                               count = count + 1
                                                

def CreateTrainingMovies(csv_file, image, segimage, crop_size, total_categories, trainlabel, save_dir, gridX = 1, gridY = 1, offset = 0, defname = "" ):

            Path(save_dir).mkdir(exist_ok=True)
            name = 1
            #Check if the csv file exists
            if os.path.exists(csv_file):
                    dataset = pd.read_csv(csv_file)
                    # The csv files contain TYX or TYX + Angle
                    if len(dataset.keys() >= 3):
                        
                        time = dataset[dataset.keys()[0]][1:]
                        y = dataset[dataset.keys()[1]][1:]
                        x = dataset[dataset.keys()[2]][1:]
                        angle = np.full(time.shape, 2)                        
                    if len(dataset.keys() > 3):
                        
                        angle = dataset[dataset.keys()[3]][1:]      
                    
                    #Categories + XYTHW + Confidence + Angle
                    for t in time:
                       MovieMaker(time[t], y[t], x[t], angle[t], image, segimage, crop_size, gridX, gridY, offset, total_categories, trainlabel, defname + str(name), save_dir)
                       name = name + 1
               

            
def MovieMaker(time, y, x, angle, image, segimage, crop_size, gridx, gridy, offset, total_categories, trainlabel, name, save_dir, yolo_v0):
    
       sizex, sizey, size_tminus, size_tplus = crop_size
       
       imagesizex = sizex * gridx
       imagesizey = sizey * gridx
       
       shiftNone = [0,0]
       if offset > 0 and trainlabel > 0:
                 shift_lx = [int(offset), 0] 
                 shift_rx = [-offset, 0]
                 shift_lxy = [int(offset), int(offset)]
                 shift_rxy = [-int(offset), int(offset)]
                 shift_dlxy = [int(offset), -int(offset)]
                 shift_drxy = [-int(offset), -int(offset)]
                 shift_uy = [0, int(offset)]
                 shift_dy = [0, -int(offset)]
                 AllShifts = [shiftNone, shift_lx, shift_rx,shift_lxy,shift_rxy,shift_dlxy,shift_drxy,shift_uy,shift_dy]

       else:
           
          AllShifts = [shiftNone]


       
       currentsegimage = segimage[time,:].astype('uint16')
       height, width, center, seg_label = getHW(x, y, trainlabel, currentsegimage)
       for shift in AllShifts:
           
                newname = name + 'shift' + str(shift)
                Event_data = []
                newcenter = (center[0] - shift[1],center[1] - shift[0] )
                x = center[1]
                y = center[0]
                if yolo_v0:
                    Label = np.zeros([total_categories + 6])
                else:    
                    Label = np.zeros([total_categories + 7])
                Label[trainlabel] = 1
                #T co ordinate
                Label[total_categories + 2] = (size_tminus) / (size_tminus + size_tplus)
                if x + shift[0]> sizex/2 and y + shift[1] > sizey/2 and x + shift[0] + int(imagesizex/2) < image.shape[2] and y + shift[1]+ int(imagesizey/2) < image.shape[1] and time > size_tminus and time + size_tplus + 1 < image.shape[0]:
                        crop_xminus = x  - int(imagesizex/2)
                        crop_xplus = x  + int(imagesizex/2)
                        crop_yminus = y  - int(imagesizey/2)
                        crop_yplus = y   + int(imagesizey/2)
                        # Cut off the region for training movie creation
                        region =(slice(int(time - size_tminus),int(time + size_tplus  + 1)),slice(int(crop_yminus)+ shift[1], int(crop_yplus)+ shift[1]),
                              slice(int(crop_xminus) + shift[0], int(crop_xplus) + shift[0]))
                        #Define the movie region volume that was cut
                        crop_image = image[region]   
                        
                        seglocationx = (newcenter[1] - crop_xminus)
                        seglocationy = (newcenter[0] - crop_yminus)
                         
                        Label[total_categories] =  seglocationx/sizex
                        Label[total_categories + 1] = seglocationy/sizey
                        if height >= imagesizey:
                                        height = 0.5 * imagesizey
                        if width >= imagesizex:
                                        width = 0.5 * imagesizex
                        #Height
                        Label[total_categories + 3] = height/imagesizey
                        #Width
                        Label[total_categories + 4] = width/imagesizex
               
                          
                        Label[total_categories + 5] = angle  
                        if yolo_v0 == False:
                                if seg_label > 0:
                                  Label[total_categories + 6] = 1 
                                else:
                                  Label[total_categories + 6] = 0   
                      
                        #Write the image as 32 bit tif file 
                        if(crop_image.shape[0] == size_tplus + size_tminus + 1 and crop_image.shape[1]== imagesizey and crop_image.shape[2]== imagesizex):
                                  
                                   imwrite((save_dir + '/' + newname + '.tif'  ) , crop_image.astype('float32'))    
                                   Event_data.append([Label[i] for i in range(0,len(Label))])
                                   if(os.path.exists(save_dir + '/' + (newname) + ".csv")):
                                                os.remove(save_dir + '/' + (newname) + ".csv")
                                   writer = csv.writer(open(save_dir + '/' + (newname) + ".csv", "a"))
                                   writer.writerows(Event_data)

       
   
def Readname(fname):
    
    return os.path.basename(os.path.splitext(fname)[0])


def ImageLabelDataSet(image_dir, seg_image_dir, csv_dir,save_dir, static_name, static_label, csv_name_diff,crop_size, gridx = 1, gridy = 1, offset = 0, yolo_v0 = True):
    
    
            raw_path = os.path.join(image_dir, '*tif')
            Seg_path = os.path.join(seg_image_dir, '*tif')
            Csv_path = os.path.join(csv_dir, '*csv')
            files_raw = glob.glob(raw_path)
            files_raw.sort
            filesSeg = glob.glob(Seg_path)
            filesSeg.sort
            filesCsv = glob.glob(Csv_path)
            filesCsv.sort
            Path(save_dir).mkdir(exist_ok=True)
            total_categories = len(static_name)
            
            for csvfname in filesCsv:
              print(csvfname)
              count = 0
              Csvname =  os.path.basename(os.path.splitext(csvfname)[0])
            
              for fname in files_raw:
                  
                 name = os.path.basename(os.path.splitext(fname)[0])   
                 for Segfname in filesSeg:
                      
                      Segname = os.path.basename(os.path.splitext(Segfname)[0])
                        
                      if name == Segname:
                          
                          
                         image = imread(fname)
                         segimage = imread(Segfname)
                         for i in  range(0, len(static_name)):
                             event_name = static_name[i]
                             trainlabel = static_label[i]
                             if Csvname == csv_name_diff + name + event_name:
                                            dataset = pd.read_csv(csvfname)
                                            time = dataset[dataset.keys()[0]][1:]
                                            y = dataset[dataset.keys()[1]][1:]
                                            x = dataset[dataset.keys()[2]][1:]     
                                            
                                            #Categories + XYHW + Confidence 
                                            for t in range(1, len(time)):
                                               ImageMaker(time[t], y[t], x[t], image, segimage, crop_size, gridx, gridy, offset, total_categories, trainlabel, name + event_name + str(count), save_dir,yolo_v0)    
                                               count = count + 1
    


    
                 
    
def createNPZ(save_dir, axes, save_name = 'Yolov0oneat', save_name_val = 'Yolov0oneatVal'):
            
            data = []
            label = []   
             
            raw_path = os.path.join(save_dir, '*tif')
            files_raw = glob.glob(raw_path)
            files_raw.sort
            
            Images= [imread(fname)[0,:] for fname in files_raw]
            names = [Readname(fname)  for fname in files_raw]
            #Normalize everything before it goes inside the training
            NormalizeImages = [normalizeFloatZeroOne(image.astype('uint16'),1,99.8) for image in tqdm(Images)]


            for i in range(0,len(NormalizeImages)):
               
               n = NormalizeImages[i]
               blankX = n
               csvfname = save_dir + '/' + names[i] + '.csv'   
               arr = [] 
               with open(csvfname) as csvfile:
                     reader = csv.reader(csvfile, delimiter = ',')
                     for train_vec in reader:

                             arr =  [float(s) for s in train_vec[0:]]
               blankY = arr

               blankY = np.expand_dims(blankY, -1)
               blankX = np.expand_dims(blankX, -1)

               data.append(blankX)
               label.append(blankY)



            dataarr = np.asarray(data)
            labelarr = np.asarray(label)
            print(dataarr.shape, labelarr.shape)
            traindata, validdata, trainlabel, validlabel = train_test_split(dataarr, labelarr, train_size=0.95,test_size=0.05, shuffle= True)
            save_full_training_data(save_dir, save_name, traindata, trainlabel, axes)
            save_full_training_data(save_dir, save_name_val, validdata, validlabel, axes)
    

def _raise(e):
    raise e
def  ImageMaker(time, y, x, image, segimage, crop_size, gridX, gridY, offset, total_categories, trainlabel, name, save_dir, yolo_v0):

               sizeX, sizeY = crop_size

               ImagesizeX = sizeX * gridX
               ImagesizeY = sizeY * gridY

               shiftNone = [0,0]
               if offset > 0 and trainlabel > 0:
                         shiftLX = [int(offset), 0] 
                         shiftRX = [-offset, 0]
                         shiftLXY = [int(offset), int(offset)]
                         shiftRXY = [-int(offset), int(offset)]
                         shiftDLXY = [int(offset), -int(offset)]
                         shiftDRXY = [-int(offset), -int(offset)]
                         shiftUY = [0, int(offset)]
                         shiftDY = [0, -int(offset)]
                         AllShifts = [shiftNone, shiftLX, shiftRX,shiftLXY,shiftRXY,shiftDLXY,shiftDRXY,shiftUY,shiftDY]

               else:

                  AllShifts = [shiftNone]

       
               if time < segimage.shape[0] - 1:
                 currentsegimage = segimage[int(time),:].astype('uint16')
                
                 height, width, center, SegLabel  = getHW(x, y,trainlabel, currentsegimage)
                 for shift in AllShifts:
                   
                        newname = name + 'shift' + str(shift)
                        newcenter = (center[0] - shift[1],center[1] - shift[0] )
                        Event_data = []
                        
                        x = center[1]
                        y = center[0]
                        if yolo_v0:
                          Label = np.zeros([total_categories + 4])
                        else:
                          Label = np.zeros([total_categories + 5])  
                        Label[trainlabel] = 1
                        if x + shift[0]> sizeX/2 and y + shift[1] > sizeY/2 and x + shift[0] + int(ImagesizeX/2) < image.shape[2] and y + shift[1]+ int(ImagesizeY/2) < image.shape[1]:
                                    crop_Xminus = x  - int(ImagesizeX/2)
                                    crop_Xplus = x   + int(ImagesizeX/2)
                                    crop_Yminus = y  - int(ImagesizeY/2)
                                    crop_Yplus = y   + int(ImagesizeY/2)
                                 
                                    
                                    region =(slice(int(time - 1),int(time)),slice(int(crop_Yminus)+ shift[1], int(crop_Yplus)+ shift[1]),
                                           slice(int(crop_Xminus) + shift[0], int(crop_Xplus) + shift[0]))
                                   
                                    crop_image = image[region]      
                                    seglocationX = (newcenter[1] - crop_Xminus)
                                    seglocationY = (newcenter[0] - crop_Yminus)
                                      
                                    Label[total_categories] =  seglocationX/sizeX
                                    Label[total_categories + 1] = seglocationY/sizeY
                                    
                                    if height >= ImagesizeY:
                                        height = 0.5 * ImagesizeY
                                    if width >= ImagesizeX:
                                        width = 0.5 * ImagesizeX
                                    
                                    Label[total_categories + 2] = height/ImagesizeY
                                    Label[total_categories + 3] = width/ImagesizeX
                                    
                                        
                                    
                                   
                                    if yolo_v0==False:
                                            if SegLabel > 0:
                                              Label[total_categories + 4] = 1 
                                            else:
                                              Label[total_categories + 4] = 0  
                                 
                                    if(crop_image.shape[1]== ImagesizeY and crop_image.shape[2]== ImagesizeX):
                                             imwrite((save_dir + '/' + newname + '.tif'  ) , crop_image.astype('float32'))  
                                             Event_data.append([Label[i] for i in range(0,len(Label))])
                                             if(os.path.exists(save_dir + '/' + (newname) + ".csv")):
                                                os.remove(save_dir + '/' + (newname) + ".csv")
                                             writer = csv.writer(open(save_dir + '/' + (newname) + ".csv", "a"))
                                             writer.writerows(Event_data)

       
def getHW(defaultX, defaultY, trainlabel, currentsegimage):
    
    properties = measure.regionprops(currentsegimage, currentsegimage)
    TwoDLocation = (defaultY,defaultX)
    TwoDCoordinates = [(prop.centroid[0], prop.centroid[1]) for prop in properties]
    SegLabel = currentsegimage[int(TwoDLocation[0]), int(TwoDLocation[1])]
    for prop in properties:
                                               
                  if SegLabel > 0 and prop.label == SegLabel:
                                    minr, minc, maxr, maxc = prop.bbox
                                    center = prop.centroid
                                    height =  abs(maxc - minc)
                                    width =  abs(maxr - minr)
                                
                  if SegLabel == 0:
                    
                             center = (defaultY, defaultX)
                             height = 10
                             width = 10
                               
                    
                                
    return height, width, center, SegLabel     

def save_full_training_data(directory, filename, data, label, axes):
    """Save training data in ``.npz`` format."""
  

    len(axes) == data.ndim or _raise(ValueError())
    np.savez(directory + filename, data = data, label = label, axes = axes) 
