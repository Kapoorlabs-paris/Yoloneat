from __future__ import print_function, unicode_literals, absolute_import, division
#import matplotlib.pyplot as plt
import numpy as np
import os
import collections
import warnings
import csv
import json
import cv2
from matplotlib import cm
from tifffile import imsave
from skimage import measure
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
    

"""
 
   Here we have added some of the useful functions taken from the csbdeep package which are a part of third party software called CARE
   https://github.com/CSBDeep/CSBDeep

"""    
  ##Save image data as a tiff file, function defination taken from CARE csbdeep python package  
    
def MarkerToCSV(MarkerImage):
    
    MarkerImage = MarkerImage.astype('uint16')
    MarkerList = []
    for i in range(0, MarkerImage.shape[0]):
          waterproperties = measure.regionprops(MarkerImage, MarkerImage)
          indices = [prop.centroid for prop in waterproperties]
          MarkerList.append([i, indices[0], indices[1]])
    return  MarkerList
    
  
def load_json(fpath):
    with open(fpath, 'r') as f:
        return json.load(f)
    
def save_json(data,fpath,**kwargs):
    with open(fpath,'w') as f:
        f.write(json.dumps(data,**kwargs))    
  
def save_tiff_imagej_compatible(file, img, axes, **imsave_kwargs):
    """Save image in ImageJ-compatible TIFF format.

    Parameters
    ----------
    file : str
        File name
    img : numpy.ndarray
        Image
    axes: str
        Axes of ``img``
    imsave_kwargs : dict, optional
        Keyword arguments for :func:`tifffile.imsave`

    """
    t = np.uint8
    # convert to imagej-compatible data type
    t_new = t
    img = img.astype(t_new, copy=False)
    if t != t_new:
        warnings.warn("Converting data type from '%s' to ImageJ-compatible '%s'." % (t, np.dtype(t_new)))

    imsave_kwargs['imagej'] = True
    imsave(file, img, **imsave_kwargs)


   ##CARE csbdeep modification of implemented function
def normalizeFloat(x, pmin = 3, pmax = 99.8, axis = None, eps = 1e-20, dtype = np.float32):
    """Percentile based Normalization
    
    Normalize patches of image before feeding into the network
    
    Parameters
    ----------
    x : np array Image patch
    pmin : minimum percentile value for normalization
    pmax : maximum percentile value for normalization
    axis : axis along which the normalization has to be carried out
    eps : avoid dividing by zero
    dtype: type of numpy array, float 32 default
    """
    mi = np.percentile(x, pmin, axis = axis, keepdims = True)
    ma = np.percentile(x, pmax, axis = axis, keepdims = True)
    return normalize_mi_ma(x, mi, ma, eps = eps, dtype = dtype)

def normalizeFloatZeroOne(x, pmin = 3, pmax = 99.8, axis = None, eps = 1e-20, dtype = np.float32):
    """Percentile based Normalization
    
    Normalize patches of image before feeding into the network
    
    Parameters
    ----------
    x : np array Image patch
    pmin : minimum percentile value for normalization
    pmax : maximum percentile value for normalization
    axis : axis along which the normalization has to be carried out
    eps : avoid dividing by zero
    dtype: type of numpy array, float 32 default
    """
    mi = np.percentile(x, pmin, axis = axis, keepdims = True)
    ma = np.percentile(x, pmax, axis = axis, keepdims = True)
    return normalizer(x, mi, ma, eps = eps, dtype = dtype)




def normalize_mi_ma(x, mi , ma, eps = 1e-20, dtype = np.float32):
    
    
    """
    Number expression evaluation for normalization
    
    Parameters
    ----------
    x : np array of Image patch
    mi : minimum input percentile value
    ma : maximum input percentile value
    eps: avoid dividing by zero
    dtype: type of numpy array, float 32 defaut
    """
    
    
    if dtype is not None:
        x = x.astype(dtype, copy = False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy = False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy = False)
        eps = dtype(eps)
        
    try: 
        import numexpr
        x = numexpr.evaluate("(x - mi ) / (ma - mi + eps)")
    except ImportError:
        x = (x - mi) / (ma - mi + eps)

        
    return x

def normalizer(x, mi , ma, eps = 1e-20, dtype = np.float32):


    """
    Number expression evaluation for normalization
    
    Parameters
    ----------
    x : np array of Image patch
    mi : minimum input percentile value
    ma : maximum input percentile value
    eps: avoid dividing by zero
    dtype: type of numpy array, float 32 defaut
    """


    if dtype is not None:
        x = x.astype(dtype, copy = False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy = False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy = False)
        eps = dtype(eps)

    try:
        import numexpr
        x = numexpr.evaluate("(x - mi ) / (ma - mi + eps)")
    except ImportError:
        x = (x - mi) / (ma - mi + eps)

        x = normalizeZeroOne(x)
    return x

def normalizeZeroOne(x):

     x = x.astype('float32')

     minVal = np.min(x)
     maxVal = np.max(x)
     
     x = ((x-minVal) / (maxVal - minVal + 1.0e-20))
     
     return x
    
       
    

def load_training_data(directory, filename,axes=None, verbose= True):
    """ Load training data in .npz format.
    The data file is expected to have the keys 'data' and 'label'     
    """
    if directory is not None:
      npzdata=np.load(directory + filename)
    else : 
      npzdata=np.load(filename)
   
    
    X = npzdata['data']
    Y = npzdata['label']
    Z = npzdata['label2']
    
        
    
    if axes is None:
        axes = npzdata['axes']
    axes = axes_check_and_normalize(axes)
    assert 'C' in axes
    n_images = X.shape[0]
    assert X.shape[0] == Y.shape[0]
    assert 0 < n_images <= X.shape[0]
  
    
    X, Y = X[:n_images], Y[:n_images]
    channel = axes_dict(axes)['C']
    

       

    X = move_channel_for_backend(X,channel=channel)
    
    axes = axes.replace('C','') # remove channel
    if backend_channels_last():
        axes = axes+'C'
    else:
        axes = axes[:1]+'C'+axes[1:]

   

    if verbose:
        ax = axes_dict(axes)
        n_train = len(X)
        image_size = tuple( X.shape[ax[a]] for a in 'TZYX' if a in axes )
        n_dim = len(image_size)
        n_channel_in = X.shape[ax['C']]

        print('number of  images:\t', n_train)
       
        print('image size (%dD):\t\t'%n_dim, image_size)
        print('axes:\t\t\t\t', axes)
        print('channels in / out:\t\t', n_channel_in)

    return (X,Y,Z), axes
  
    
def _raise(e):
    raise e

# https://docs.python.org/3/library/itertools.html#itertools-recipes
def consume(iterator):
    collections.deque(iterator, maxlen=0)



def axes_check_and_normalize(axes,length=None,disallowed=None,return_allowed=False):
    """
    S(ample), T(ime), C(hannel), Z, Y, X
    """
    allowed = 'STCZYX'
    axes = str(axes).upper()
    consume(a in allowed or _raise(ValueError("invalid axis '%s', must be one of %s."%(a,list(allowed)))) for a in axes)
    disallowed is None or consume(a not in disallowed or _raise(ValueError("disallowed axis '%s'."%a)) for a in axes)
    consume(axes.count(a)==1 or _raise(ValueError("axis '%s' occurs more than once."%a)) for a in axes)
    length is None or len(axes)==length or _raise(ValueError('axes (%s) must be of length %d.' % (axes,length)))
    return (axes,allowed) if return_allowed else axes
def axes_dict(axes):
    """
    from axes string to dict
    """
    axes, allowed = axes_check_and_normalize(axes,return_allowed=True)
    return { a: None if axes.find(a) == -1 else axes.find(a) for a in allowed }
    # return collections.namedt      
    
def load_full_training_data(directory, filename,axes=None, verbose= True):
    """ Load training data in .npz format.
    The data file is expected to have the keys 'data' and 'label'     
    """
    
    if directory is not None:
      npzdata=np.load(directory + filename)
    else:
      npzdata=np.load(filename)  
    
    
    X = npzdata['data']
    Y = npzdata['label']
    
    
        
    
    if axes is None:
        axes = npzdata['axes']
    axes = axes_check_and_normalize(axes)
    assert 'C' in axes
    n_images = X.shape[0]
    assert X.shape[0] == Y.shape[0]
    assert 0 < n_images <= X.shape[0]
  
    
    X, Y = X[:n_images], Y[:n_images]
    channel = axes_dict(axes)['C']
    

       

    X = move_channel_for_backend(X,channel=channel)
    
    axes = axes.replace('C','') # remove channel
    if backend_channels_last():
        axes = axes+'C'
    else:
        axes = axes[:1]+'C'+axes[1:]

   

    if verbose:
        ax = axes_dict(axes)
        n_train = len(X)
        image_size = tuple( X.shape[ax[a]] for a in 'TZYX' if a in axes )
        n_dim = len(image_size)
        n_channel_in = X.shape[ax['C']]

        print('number of  images:\t', n_train)
       
        print('image size (%dD):\t\t'%n_dim, image_size)
        print('axes:\t\t\t\t', axes)
        print('channels in / out:\t\t', n_channel_in)

    return (X,Y), axes
        
    
    
def backend_channels_last():
    import keras.backend as K
    assert K.image_data_format() in ('channels_first','channels_last')
    return K.image_data_format() == 'channels_last'


def move_channel_for_backend(X,channel):
    if backend_channels_last():
        return np.moveaxis(X, channel, -1)
    else:
        return np.moveaxis(X, channel,  1)    
    
def time_pad(image, TimeFrames):

         time = image.shape[0]
         
         timeextend = time
         
         while timeextend%TimeFrames!=0:
              timeextend = timeextend + 1
              
         extendimage = np.zeros([timeextend, image.shape[1], image.shape[2]])
              
         extendimage[0:time,:,:] = image
              
         return extendimage      
    
def chunk_list(image, patchshape, stride, pair):
            rowstart = pair[0]
            colstart = pair[1]
            
            endrow = rowstart + patchshape[0]
            endcol = colstart + patchshape[1]
            
            if endrow > image.shape[1]:
                endrow = image.shape[1]
            if endcol > image.shape[2]:
                endcol = image.shape[2]    
            
            
            region = (slice(0,image.shape[0]),slice(rowstart, endrow),
                      slice(colstart, endcol))
            
            # The actual pixels in that region.
            patch = image[region]
                
            # Always normalize patch that goes into the netowrk for getting a prediction score 
            patch = normalizeFloatZeroOne(patch,1,99.8)
            patch = zero_pad(patch, stride,stride)
         
        
            return patch, rowstart, colstart     
           

def Printpredict(idx, model, data, Truelabel, Categories_name,  cols=5, threshold=.8, plot = False, simple = False, catsimple = False):
    try:
        idx = list(idx)
    except:
        idx = [idx]
        

    data = data[idx]
    Truelabel = Truelabel[idx]
    mean = np.mean(data)
    
    if mean > 0:
      prediction = model.predict(data)
   
      i = 0
   
      while i < prediction.shape[0]:
        if plot:  
          import matplotlib.pyplot as plt  
          fig, ax = plt.subplots(1,data.shape[1],figsize=(5*cols,5))
          fig.figsize=(20,10)
        
        for i in range(0,(prediction.shape[0])): 
           
           for j in range(0,data.shape[1]):
           
            img = data[i,j,:,:,0]
            if plot:
              ax[j].imshow(img, cm.Spectral)
     
           if len(Categories_name) > 1: 
            for k in range(0, len(Categories_name)):
               
               Name, Label = Categories_name[k]
               
               print('Top predictions : ' , Name, 'Probability', ':' , prediction[i,:,:, int(Label)])
       
            if simple == False and catsimple == False:
                   print('X Y T H W',prediction[i,:,:,int(Label)+1:])
           
            
           print('True Label : ', Truelabel)
            
         

           if plot:
              plt.show()     
          
           i += 1
           if i >= prediction.shape[0]:
                  break
              
                
def PrintStaticpredict(idx, model, data, Truelabel, Categories_name,  cols=5, threshold=.8, plot = False, simple = False, catsimple = False):

    try:
        idx = list(idx)
    except:
        idx = [idx]
        

    data = data[idx]
    Truelabel = Truelabel[idx]
    mean = np.mean(data)
    
    if mean > 0:
      prediction = model.predict(data)
   
      i = 0
   
      while i < prediction.shape[0]:
           
          
        
        for i in range(0,(prediction.shape[0])): 
           
           
           
            img = data[i,:,:,0]
            if plot:
              import matplotlib.pyplot as plt   
              plt.imshow(img, cm.Spectral)
              plt.show()   
            if len(Categories_name) > 1: 
              for k in range(0, len(Categories_name)):
               
               Name, Label = Categories_name[k]
               
               print('Top predictions : ' , Name, 'Probability', ':' , prediction[i,:,:, int(Label)])
  
               if simple == False or catsimple == False:
                    print('X Y',prediction[i,:,:,int(Label)+1:int(Label)+3])
              
           
            print('True Label : ', Truelabel)
            
         

            if plot:
              plt.show()     
          
            i += 1
            if i >= prediction.shape[0]:
                  break              

def X_right_prediction(image,sY, sX, time_prediction, stride, inputtime, Categories_Name, Categories_event_threshold, TrainshapeX, TrainshapeY, TimeFrames):
    
                         LocationBoxes = []
                         j = 0
                         k = 1
                         while True:
                                      j = j + 1
                                      if j > time_prediction.shape[1]:
                                           j = 1
                                           k = k + 1
                             
                                      if k > time_prediction.shape[0]:
                                          break;
                                      
                                      y = (k - 1) * stride
                                      x = (j - 1) * stride
                                      prediction_vector = time_prediction[k-1,j-1,:]
                                      #Note to self k,1 is x j,0 is y 
                                      for p in range(1, len(Categories_Name)):
                                           if prediction_vector[p] > (Categories_event_threshold[p]):
                                                 
                                                 Xbox =  x + sX + prediction_vector[len(Categories_Name)] * TrainshapeX
                                                 Ybox =  y + sY + prediction_vector[len(Categories_Name) + 1] * TrainshapeY
                                                 Tbox = int(inputtime + 1 + prediction_vector[len(Categories_Name) + 2] * TimeFrames)
                                                 
                                                 Score = prediction_vector[p]
                                                 Traw = prediction_vector[len(Categories_Name) + 2]
                                                 Name, Label = Categories_Name[p] 
                                                 box = (x, y,x + TrainshapeX, y + TrainshapeY, Xbox,Ybox, Score, Tbox, Label,Traw )
                                                 
                                                 boxregion = (slice(0,image.shape[0]),slice(y  , y   + TrainshapeY,),slice(x  , x   + TrainshapeX))
                                                 sliceboxregion = image[boxregion]
                                                 try:
                                                  if np.mean(sliceboxregion) > 0.3:
                                                      LocationBoxes.append([box, Label])
                                                 except ValueError:
                                                     print('No box', x, y)
                                                    
                                                    
                         return LocationBoxes      


def draw_labelimages(image, location, time, timelocation ):

     cv2.circle(image, location, 2,(255,0,0), thickness = -1 )
     

     return image 

def zero_pad(image, TrainshapeX, TrainshapeY):

          time = image.shape[0]
          sizeY = image.shape[2]
          sizeX = image.shape[1]
          
          sizeXextend = sizeX
          sizeYextend = sizeY
         
 
          while sizeXextend%TrainshapeX!=0:
              sizeXextend = sizeXextend + 1
        
          while sizeYextend%TrainshapeY!=0:
              sizeYextend = sizeYextend + 1

          extendimage = np.zeros([time, sizeXextend, sizeYextend])
          
          extendimage[0:time, 0:sizeX, 0:sizeY] = image
              
              
          return extendimage
      
def extra_pad(image, patchX, patchY):

          extendimage = np.zeros([image.shape[0],image.shape[1] + patchX, image.shape[2] + patchY])
          
          extendimage[0:image.shape[0], 0:image.shape[1], 0:image.shape[2]] = image     
          
          return extendimage
 
def save_labelimages(save_dir, image, axes, fname, Name):

    
             save_tiff_imagej_compatible((save_dir + Name + os.path.basename(fname) ) , image, axes)
        
    
                
def save_csv(save_dir, Event_Count, Name):
      
    Event_data= []

    Path(save_dir).mkdir(exist_ok = True)

    for line in Event_Count:
      Event_data.append(line)
    writer = csv.writer(open(save_dir + "/" + (Name )  +".csv", "w"))
    writer.writerows(Event_data)                
                
