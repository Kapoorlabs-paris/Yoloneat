from NEATUtils import plotters
import numpy as np
from NEATUtils import helpers
from NEATUtils.helpers import get_nearest, save_json, load_json, yoloprediction, normalizeFloatZeroOne, GenerateMarkers, \
    DensityCounter, MakeTrees, nonfcn_yoloprediction, fastnms, averagenms
from keras import callbacks
import os
import math
import tensorflow as tf
from tqdm import tqdm
from NEATModels import nets
from NEATModels.nets import Concat
from NEATModels.loss import dynamic_yolo_loss
from keras import backend as K
# from IPython.display import clear_output
from keras import optimizers
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
from keras.models import load_model
from tifffile import imread, imwrite
import csv
import napari
import glob
from scipy import spatial
import itertools
from napari.qt.threading import thread_worker
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QPushButton, QSlider
import h5py
import cv2
import imageio

Boxname = 'ImageIDBox'
EventBoxname = 'EventIDBox'


class NEATDynamic(object):
    """
    Parameters
    ----------
    
    NpzDirectory : Specify the location of npz file containing the training data with movies and labels
    
    TrainModelName : Specify the name of the npz file containing training data and labels
    
    ValidationModelName :  Specify the name of the npz file containing validation data and labels
    
    categories : Number of action classes
    
    Categories_Name : List of class names and labels
    
    model_dir : Directory location where trained model weights are to be read or written from
    
    model_name : The h5 file of CNN + LSTM + Dense Neural Network to be used for training
    
    model_keras : The model as it appears as a Keras function
    
    model_weights : If re-training model_weights = model_dir + model_name else None as default
    
    lstm_hidden_units : Number of hidden uniots for LSTm layer, 64 by default
    
    epochs :  Number of training epochs, 55 by default
    
    batch_size : batch_size to be used for training, 20 by default
    
    
    
    """

    def __init__(self, config, model_dir, model_name, catconfig=None, cordconfig=None):

        self.config = config
        self.catconfig = catconfig
        self.cordconfig = cordconfig
        self.model_dir = model_dir
        self.model_name = model_name
        if self.config != None:
            self.npz_directory = config.npz_directory
            self.npz_name = config.npz_name
            self.npz_val_name = config.npz_val_name
            self.key_categories = config.key_categories
            self.stage_number = config.stage_number
            self.last_conv_factor = config.last_conv_factor
            self.show = config.show
            self.key_cord = config.key_cord
            self.box_vector = len(config.key_cord)
            self.categories = len(config.key_categories)
            self.depth = config.depth
            self.start_kernel = config.start_kernel
            self.mid_kernel = config.mid_kernel
            self.lstm_kernel = config.lstm_kernel
            self.learning_rate = config.learning_rate
            self.epochs = config.epochs
            self.residual = config.residual
            self.startfilter = config.startfilter
            self.batch_size = config.batch_size
            self.multievent = config.multievent
            self.imagex = config.imagex
            self.imagey = config.imagey
            self.imaget = config.size_tminus + config.size_tplus + 1
            self.size_tminus = config.size_tminus
            self.size_tplus = config.size_tplus
            
            self.nboxes = config.nboxes
            self.gridx = 1
            self.gridy = 1
            self.gridt = 1
            self.yolo_v0 = config.yolo_v0
            self.yolo_v1 = config.yolo_v1
            self.yolo_v2 = config.yolo_v2
            self.stride = config.stride
            self.lstm_hidden_unit = config.lstm_hidden_unit
        if self.config == None:

            try:
                self.config = load_json(self.model_dir + os.path.splitext(self.model_name)[0] + '_Parameter.json')
            except:
                self.config = load_json(self.model_dir + self.model_name + '_Parameter.json')

            self.npz_directory = self.config['npz_directory']
            self.npz_name = self.config['npz_name']
            self.npz_val_name = self.config['npz_val_name']
            self.key_categories = self.catconfig
            self.box_vector = self.config['box_vector']
            self.show = self.config['show']
            self.key_cord = self.cordconfig
            self.categories = len(self.catconfig)
            self.depth = self.config['depth']
            self.start_kernel = self.config['start_kernel']
            self.mid_kernel = self.config['mid_kernel']
            self.lstm_kernel = self.config['lstm_kernel']
            self.lstm_hidden_unit = self.config['lstm_hidden_unit']
            self.learning_rate = self.config['learning_rate']
            self.epochs = self.config['epochs']
            self.residual = self.config['residual']
            self.startfilter = self.config['startfilter']
            self.batch_size = self.config['batch_size']
            self.multievent = self.config['multievent']
            self.imagex = self.config['imagex']
            self.imagey = self.config['imagey']
            self.imaget = self.config['size_tminus'] + self.config['size_tplus'] + 1
            self.size_tminus = self.config['size_tminus']
            self.size_tplus = self.config['size_tplus']
            self.nboxes = self.config['nboxes']
            self.stage_number = self.config['stage_number']
            self.last_conv_factor = self.config['last_conv_factor']
            self.gridx = 1
            self.gridy = 1
            self.gridt = 1
            self.yolo_v0 = self.config['yolo_v0']
            self.yolo_v1 = self.config['yolo_v1']
            self.yolo_v2 = self.config['yolo_v2']
            self.stride = self.config['stride']
            self.lstm_hidden_unit = self.config['lstm_hidden_unit']

        self.X = None
        self.Y = None
        self.axes = None
        self.X_val = None
        self.Y_val = None
        self.Trainingmodel = None
        self.Xoriginal = None
        self.Xoriginal_val = None

        if self.residual:
            self.model_keras = nets.ORNET
        else:
            self.model_keras = nets.OSNET

        if self.multievent == True:
            self.last_activation = 'sigmoid'
            self.entropy = 'binary'

        if self.multievent == False:
            self.last_activation = 'softmax'
            self.entropy = 'notbinary'
        self.yololoss = dynamic_yolo_loss(self.categories, self.gridx, self.gridy, self.gridt, self.nboxes,
                                          self.box_vector, self.entropy, self.yolo_v0, self.yolo_v1, self.yolo_v2)

    def loadData(self):

        (X, Y), axes = helpers.load_full_training_data(self.npz_directory, self.npz_name, verbose=True)

        (X_val, Y_val), axes = helpers.load_full_training_data(self.npz_directory, self.npz_val_name, verbose=True)

        self.Xoriginal = X
        self.Xoriginal_val = X_val

        self.X = X
        self.Y = Y[:, :, 0]
        self.X_val = X_val
        self.Y_val = Y_val[:, :, 0]

        self.axes = axes
        self.Y = self.Y.reshape((self.Y.shape[0], 1, 1, self.Y.shape[1]))
        self.Y_val = self.Y_val.reshape((self.Y_val.shape[0], 1, 1, self.Y_val.shape[1]))

    def TrainModel(self):

        input_shape = (self.X.shape[1], self.X.shape[2], self.X.shape[3], self.X.shape[4])

        Path(self.model_dir).mkdir(exist_ok=True)

        if self.yolo_v2:

            for i in range(self.Y.shape[0]):

                if self.Y[i, :, :, 0] == 1:
                    self.Y[i, :, :, -1] = 1
            for i in range(self.Y_val.shape[0]):

                if self.Y_val[i, :, :, 0] == 1:
                    self.Y_val[i, :, :, -1] = 1
        Y_rest = self.Y[:, :, :, self.categories:]
        Y_main = self.Y[:, :, :, 0:self.categories - 1]

        y_integers = np.argmax(Y_main, axis=-1)
        y_integers = y_integers[:, 0, 0]

        d_class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
        d_class_weights = d_class_weights.reshape(1, d_class_weights.shape[0])

        model_weights = self.model_dir + self.model_name
        if os.path.exists(model_weights):

            self.model_weights = model_weights
            print('loading weights')
        else:

            self.model_weights = None

        dummyY = np.zeros(
            [self.Y.shape[0], self.Y.shape[1], self.Y.shape[2], self.categories + self.nboxes * self.box_vector])
        dummyY[:, :, :, :self.Y.shape[3]] = self.Y

        dummyY_val = np.zeros([self.Y_val.shape[0], self.Y_val.shape[1], self.Y_val.shape[2],
                               self.categories + self.nboxes * self.box_vector])
        dummyY_val[:, :, :, :self.Y_val.shape[3]] = self.Y_val

        for b in range(1, self.nboxes):
            dummyY[:, :, :, self.categories + b * self.box_vector:self.categories + (b + 1) * self.box_vector] = self.Y[
                                                                                                                 :, :,
                                                                                                                 :,
                                                                                                                 self.categories: self.categories + self.box_vector]
            dummyY_val[:, :, :,
            self.categories + b * self.box_vector:self.categories + (b + 1) * self.box_vector] = self.Y_val[:, :, :,
                                                                                                 self.categories: self.categories + self.box_vector]

        self.Y = dummyY
        self.Y_val = dummyY_val

        print(self.Y.shape, self.nboxes)

        self.Trainingmodel = self.model_keras(input_shape, self.categories, unit=self.lstm_hidden_unit,
                                              box_vector=Y_rest.shape[-1], nboxes=self.nboxes,
                                              stage_number=self.stage_number, last_conv_factor=self.last_conv_factor,
                                              depth=self.depth, start_kernel=self.start_kernel,
                                              mid_kernel=self.mid_kernel, lstm_kernel=self.lstm_kernel,
                                              startfilter=self.startfilter, input_weights=self.model_weights,
                                              last_activation=self.last_activation)

        sgd = optimizers.SGD(lr=self.learning_rate, momentum=0.99, decay=1e-6, nesterov=True)
        self.Trainingmodel.compile(optimizer=sgd, loss=self.yololoss, metrics=['accuracy'])

        self.Trainingmodel.summary()
        print(self.startfilter)
        # Keras callbacks
        lrate = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=4, verbose=1)
        hrate = callbacks.History()
        srate = callbacks.ModelCheckpoint(self.model_dir + self.model_name, monitor='loss', verbose=1,
                                          save_best_only=False, save_weights_only=False, mode='auto', period=1)
        prate = plotters.PlotHistory(self.Trainingmodel, self.X_val, self.Y_val, self.key_categories, self.key_cord,
                                     self.gridx, self.gridy, plot=self.show, nboxes=self.nboxes)

        # Train the model and save as a h5 file
        self.Trainingmodel.fit(self.X, self.Y, class_weight=d_class_weights, batch_size=self.batch_size,
                               epochs=self.epochs, validation_data=(self.X_val, self.Y_val), shuffle=True,
                               callbacks=[lrate, hrate, srate, prate])

        # Removes the old model to be replaced with new model, if old one exists
        if os.path.exists(self.model_dir + self.model_name):
            os.remove(self.model_dir + self.model_name)

        self.Trainingmodel.save(self.model_dir + self.model_name)

    def get_markers(self, imagename, starmodel, savedir, n_tiles, markerdir=None, star=True):

        self.starmodel = starmodel
        self.imagename = imagename
        self.image = imread(imagename)
        self.density_location = {}
        Name = os.path.basename(os.path.splitext(self.imagename)[0])
        self.savedir = savedir
        self.star = star
        Path(self.savedir).mkdir(exist_ok=True)

        self.n_tiles = n_tiles
        print('Obtaining Markers')
        if markerdir is None:
            self.markers = GenerateMarkers(self.image, self.starmodel, self.n_tiles)
            markerdir = self.savedir + '/' + 'Markers'
            Path(markerdir).mkdir(exist_ok=True)
            imwrite(markerdir + '/' + Name + '.tif', self.markers.astype('float32'))
        else:
            try:
                self.markers = imread(markerdir + '/' + Name + '.tif')
            except:
                self.markers = GenerateMarkers(self.image, self.starmodel, self.n_tiles)
                markerdir = self.savedir + '/' + 'Markers'
                Path(markerdir).mkdir(exist_ok=True)
                imwrite(markerdir + '/' + Name + '.tif', self.markers.astype('float32'))
        self.marker_tree = MakeTrees(self.markers)

        # print('Computing density of each marker')
        # self.density_location = DensityCounter(self.markers, self.imagex, self.imagey)

        return self.markers, self.marker_tree, self.density_location

    def predict(self, imagename, markers, marker_tree, density_location, savedir, n_tiles=(1, 1), overlap_percent=0.8,
                event_threshold=0.5, iou_threshold=0.1, density_veto=5):

        self.imagename = imagename
        self.image = imread(imagename)
        self.Colorimage = np.zeros([self.image.shape[0], self.image.shape[1], self.image.shape[2], 3], dtype='uint16')
        print(self.Colorimage.shape)
        self.Colorimage[:, :, :, 0] = self.image
        self.markers = markers
        self.marker_tree = marker_tree
        self.density_location = density_location
        self.savedir = savedir
        self.n_tiles = n_tiles
        self.density_veto = density_veto
        self.overlap_percent = overlap_percent
        self.iou_threshold = iou_threshold
        self.event_threshold = event_threshold
        self.downsample_regions = {}
        self.upsample_regions = {}
        self.candidate_regions = {}
        f = h5py.File(self.model_dir + self.model_name + '.h5', 'r+')
        data_p = f.attrs['training_config']
        data_p = data_p.decode().replace("learning_rate", "lr").encode()
        f.attrs['training_config'] = data_p
        f.close()
        self.model = load_model(self.model_dir + self.model_name + '.h5',
                                custom_objects={'loss': self.yololoss, 'Concat': Concat})

        # self.first_pass_predict()
        self.second_pass_predict()

    def second_pass_predict(self):

        print('Detecting event locations')
        count = 0
        eventboxes = []
        refinedeventboxes = []
        classedboxes = {}
        savename = self.savedir + "/" + (os.path.splitext(os.path.basename(self.imagename))[0]) + '_Colored'
        for inputtime in tqdm(range(0, self.image.shape[0])):
            if inputtime >= self.size_tminus + 1 and  inputtime < self.image.shape[0] - self.imaget:

                smallimage = CreateVolume(self.image, self.size_tminus, self.size_tplus, inputtime, self.imagex,
                                          self.imagey)
                smallimage = normalizeFloatZeroOne(smallimage, 1, 99.8)
                # Cut off the region for training movie creation
                # Break image into tiles if neccessary
                predictions, allx, ally = self.predict_main(smallimage)
                # Iterate over tiles
                for p in range(0, len(predictions)):

                    sum_time_prediction = predictions[p]

                    if sum_time_prediction is not None:
                        # For each tile the prediction vector has shape N H W Categories + Training Vector labels
                        for i in range(0, sum_time_prediction.shape[0]):
                            time_prediction = sum_time_prediction[i]
                            boxprediction = yoloprediction(ally[p], allx[p], time_prediction, self.stride,
                                                           inputtime - self.size_tminus - 1, self.config,
                                                           self.key_categories, self.key_cord, self.nboxes, 'detection',
                                                           'dynamic')

                            if boxprediction is not None:
                                eventboxes = eventboxes + boxprediction
                print('Total initial predictions:', len(eventboxes))
                for (event_name, event_label) in self.key_categories.items():

                    if event_label > 0:
                        for box in eventboxes:

                            event_prob = box[event_name]
                            if event_prob >= self.event_threshold:


                                X = box['xcenter']
                                Y = box['ycenter']
                                T = box['real_time_event']

                                crop_xminus = X - int(self.imagex / 2)
                                crop_xplus = X + int(self.imagex / 2)
                                crop_yminus = Y - int(self.imagey / 2)
                                crop_yplus = T + int(self.imagey / 2)
                                region = (slice(T - self.size_tminus - 1, T + self.size_tplus), slice(int(crop_yminus), int(crop_yplus)),
                                          slice(int(crop_xminus), int(crop_xplus)))

                                crop_image = smallimage[region]

                                if crop_image.shape[0] >= self.imaget and crop_image.shape[1] >= self.imagey and \
                                        crop_image.shape[
                                            2] >= self.imagex:

                                    # Now apply the prediction for counting real events

                                    prediction_vector = self.make_patches(crop_image)

                                    boxprediction = yoloprediction(crop_yminus, crop_xminus, prediction_vector[0],
                                                                   self.stride, inputtime,
                                                                   self.config, self.key_categories, self.key_cord,
                                                                   self.nboxes, 'detection', 'dynamic',
                                                                   self.marker_tree)

                                    if boxprediction is not None:
                                        refinedeventboxes = refinedeventboxes + boxprediction

                print('Total refined predictions:',len(refinedeventboxes))

                current_event_box = []
                for box in refinedeventboxes:

                    event_prob = box[event_name]

                    if event_prob >= self.event_threshold:
                        current_event_box.append(box)
                classedboxes[event_name] = [current_event_box]
                print('Valid events:', event_name,  len(current_event_box))
                self.classedboxes = classedboxes
                self.eventboxes = eventboxes
                # nms over time
                if inputtime%(self.imaget//2) == 0:
                        self.nms()
                        self.to_csv()
                        eventboxes = []
                        classedboxes = {}
                        count = 0



    def nms(self):

        best_iou_classedboxes = {}
        self.iou_classedboxes = {}
        for (event_name, event_label) in self.key_categories.items():
            if event_label > 0:
                # Get all events

                sorted_event_box = self.classedboxes[event_name][0]
                sorted_event_box = sorted(sorted_event_box, key=lambda x: x[event_name], reverse=True)
                scores = [ sorted_event_box[i][event_name]  for i in range(len(sorted_event_box))]
                best_sorted_event_box = averagenms(sorted_event_box, scores, self.iou_threshold, self.event_threshold, event_name, 'dynamic',self.imagex, self.imagey, self.imaget)
                best_iou_classedboxes[event_name] = [sorted_event_box]

        self.iou_classedboxes = best_iou_classedboxes

    def to_csv(self):

        for (event_name, event_label) in self.key_categories.items():

            if event_label > 0:
                xlocations = []
                ylocations = []
                scores = []
                confidences = []
                tlocations = []
                radiuses = []
                angles = []

                iou_current_event_boxes = self.iou_classedboxes[event_name][0]
                iou_current_event_boxes = sorted(iou_current_event_boxes, key=lambda x: x[event_name], reverse=True)
                for iou_current_event_box in iou_current_event_boxes:
                    xcenter = iou_current_event_box['xcenter']
                    ycenter = iou_current_event_box['ycenter']
                    tcenter = iou_current_event_box['real_time_event']
                    confidence = iou_current_event_box['confidence']
                    angle = iou_current_event_box['realangle']
                    score = iou_current_event_box[event_name]
                    radius = np.sqrt(
                        iou_current_event_box['height'] * iou_current_event_box['height'] + iou_current_event_box[
                            'width'] * iou_current_event_box['width']) // 2
                    # Replace the detection with the nearest marker location

                    xlocations.append(xcenter)
                    ylocations.append(ycenter)
                    scores.append(score)
                    confidences.append(confidence)
                    tlocations.append(tcenter)
                    radiuses.append(radius)
                    angles.append(angle)

                event_count = np.column_stack(
                    [tlocations, ylocations, xlocations, scores, radiuses, confidences, angles])
                event_count = sorted(event_count, key=lambda x: x[0], reverse=False)
                event_data = []
                csvname = self.savedir + "/" + event_name + "Location" + (
                os.path.splitext(os.path.basename(self.imagename))[0])
                writer = csv.writer(open(csvname + ".csv", "a"))
                filesize = os.stat(csvname + ".csv").st_size
                if filesize < 1:
                    writer.writerow(['T', 'Y', 'X', 'Score', 'Size', 'Confidence', 'Angle'])
                for line in event_count:
                    if line not in event_data:
                        event_data.append(line)
                    writer.writerows(event_data)
                    event_data = []

                self.saveimage(xlocations, ylocations, tlocations, angles, radiuses, scores)

    def saveimage(self, xlocations, ylocations, tlocations,angles, radius, scores):

        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
        # fontScale
        fontScale = 1

        # Blue color in BGR
        textcolor = (255, 0, 0)

        # Line thickness of 2 px
        thickness = 2
        for j in range(len(xlocations)):
            startlocation = (int(xlocations[j] - radius[j]), int(ylocations[j] - radius[j]))
            endlocation = (int(xlocations[j] + radius[j]), int(ylocations[j] + radius[j]))
            Z = int(tlocations[j])

            image = self.Colorimage[Z, :, :, 1]
            color = (0, 255, 0)
            if scores[j] >= 1.0 - 1.0E-5:
                color = (0, 0, 255)
                image = self.Colorimage[Z, :, :, 2]
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.rectangle(img, startlocation, endlocation, textcolor, thickness)

            cv2.putText(img, str('%.4f' % (scores[j])), startlocation, cv2.FONT_HERSHEY_SIMPLEX, 1, textcolor,
                        thickness, cv2.LINE_AA)
            if scores[j] >= 1.0 - 1.0E-5:
                self.Colorimage[Z, :, :, 2] = img[:, :, 0]
            else:
                self.Colorimage[Z, :, :, 1] = img[:, :, 0]
            if self.yolo_v2:
                x1 = xlocations[j]
                y1 = ylocations[j]
                x2 = x1 + radius[j] * math.cos(angles[j])
                y2 = y1 + radius[j] * math.sin(angles[j])
                #cv2.line(self.Colorimage[tlocation, :], (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 1)
        #imwrite((csvimagename + '.tif'), self.Colorimage.astype('uint8'))

    def showNapari(self, imagedir, savedir, yolo_v2=False):

        Raw_path = os.path.join(imagedir, '*tif')
        X = glob.glob(Raw_path)
        self.savedir = savedir
        Imageids = []
        self.viewer = napari.Viewer()
        napari.run()
        for imagename in X:
            Imageids.append(imagename)

        eventidbox = QComboBox()
        eventidbox.addItem(EventBoxname)
        for (event_name, event_label) in self.key_categories.items():
            eventidbox.addItem(event_name)

        imageidbox = QComboBox()
        imageidbox.addItem(Boxname)
        detectionsavebutton = QPushButton(' Save detection Movie')

        for i in range(0, len(Imageids)):
            imageidbox.addItem(str(Imageids[i]))

        figure = plt.figure(figsize=(4, 4))
        multiplot_widget = FigureCanvas(figure)
        ax = multiplot_widget.figure.subplots(1, 1)
        width = 400
        dock_widget = self.viewer.window.add_dock_widget(
            multiplot_widget, name="EventStats", area='right')
        multiplot_widget.figure.tight_layout()
        self.viewer.window._qt_window.resizeDocks([dock_widget], [width], Qt.Horizontal)
        eventidbox.currentIndexChanged.connect(lambda eventid=eventidbox: EventViewer(
            self.viewer,
            imread(imageidbox.currentText()),
            eventidbox.currentText(),
            self.key_categories,
            os.path.basename(os.path.splitext(imageidbox.currentText())[0]),
            savedir,
            multiplot_widget,
            ax,
            figure,
            yolo_v2,

        )
                                               )

        imageidbox.currentIndexChanged.connect(
            lambda trackid=imageidbox: EventViewer(
                self.viewer,
                imread(imageidbox.currentText()),
                eventidbox.currentText(),
                self.key_categories,
                os.path.basename(os.path.splitext(imageidbox.currentText())[0]),
                savedir,
                multiplot_widget,
                ax,
                figure,
                yolo_v2,

            )
        )

        self.viewer.window.add_dock_widget(eventidbox, name="Event", area='left')
        self.viewer.window.add_dock_widget(imageidbox, name="Image", area='left')

    def overlaptiles(self, sliceregion):

        if self.n_tiles == (1, 1):
            patch = []
            rowout = []
            column = []
            patchx = sliceregion.shape[2] // self.n_tiles[0]
            patchy = sliceregion.shape[1] // self.n_tiles[1]
            patchshape = (patchy, patchx)
            smallpatch, smallrowout, smallcolumn = chunk_list(sliceregion, patchshape, self.stride, [0, 0])
            patch.append(smallpatch)
            rowout.append(smallrowout)
            column.append(smallcolumn)

        else:
            patchx = sliceregion.shape[2] // self.n_tiles[0]
            patchy = sliceregion.shape[1] // self.n_tiles[1]

            if patchx > self.imagex and patchy > self.imagey:
                if self.overlap_percent > 1 or self.overlap_percent < 0:
                    self.overlap_percent = 0.8

                jumpx = int(self.overlap_percent * patchx)
                jumpy = int(self.overlap_percent * patchy)

                patchshape = (patchy, patchx)
                rowstart = 0;
                colstart = 0
                pairs = []
                # row is y, col is x

                while rowstart < sliceregion.shape[1]:
                    colstart = 0
                    while colstart < sliceregion.shape[2]:
                        # Start iterating over the tile with jumps = stride of the fully convolutional network.
                        pairs.append([rowstart, colstart])
                        colstart += jumpx
                    rowstart += jumpy

                    # Include the last patch
                rowstart = sliceregion.shape[1] - patchy
                colstart = 0
                while colstart < sliceregion.shape[2] - patchx:
                    pairs.append([rowstart, colstart])
                    colstart += jumpx
                rowstart = 0
                colstart = sliceregion.shape[2] - patchx
                while rowstart < sliceregion.shape[1] - patchy:
                    pairs.append([rowstart, colstart])
                    rowstart += jumpy

                if sliceregion.shape[1] >= self.imagey and sliceregion.shape[2] >= self.imagex:

                    patch = []
                    rowout = []
                    column = []
                    for pair in pairs:
                        smallpatch, smallrowout, smallcolumn = chunk_list(sliceregion, patchshape, self.stride, pair)
                        if smallpatch.shape[1] >= self.imagey and smallpatch.shape[2] >= self.imagex:
                            patch.append(smallpatch)
                            rowout.append(smallrowout)
                            column.append(smallcolumn)

            else:

                patch = []
                rowout = []
                column = []
                patchx = sliceregion.shape[2] // self.n_tiles[0]
                patchy = sliceregion.shape[1] // self.n_tiles[1]
                patchshape = (patchy, patchx)
                smallpatch, smallrowout, smallcolumn = chunk_list(sliceregion, patchshape, self.stride, [0, 0])
                patch.append(smallpatch)
                rowout.append(smallrowout)
                column.append(smallcolumn)
        self.patch = patch
        self.sy = rowout
        self.sx = column



    def predict_main(self, sliceregion):
        try:
            self.overlaptiles(sliceregion)
            predictions = []
            allx = []
            ally = []
            if len(self.patch) > 0:
                for i in range(0, len(self.patch)):
                    sum_time_prediction = self.make_patches(self.patch[i])
                    predictions.append(sum_time_prediction)
                    allx.append(self.sx[i])
                    ally.append(self.sy[i])


            else:

                sum_time_prediction = self.make_patches(self.patch)
                predictions.append(sum_time_prediction)
                allx.append(self.sx)
                ally.append(self.sy)

        except tf.errors.ResourceExhaustedError:

            print('Out of memory, increasing overlapping tiles for prediction')
            self.list_n_tiles = list(self.n_tiles)
            self.list_n_tiles[0] = self.n_tiles[0] + 1
            self.list_n_tiles[1] = self.n_tiles[1] + 1
            self.n_tiles = tuple(self.list_n_tiles)

            self.predict_main(sliceregion)

        return predictions, allx, ally
    def make_patches(self, sliceregion):

        predict_im = np.expand_dims(sliceregion, 0)

        prediction_vector = self.model.predict(np.expand_dims(predict_im, -1), verbose=0)

        return prediction_vector

    def make_batch_patches(self, sliceregion):

        prediction_vector = self.model.predict(np.expand_dims(sliceregion, -1), verbose=0)
        return prediction_vector


def CreateVolume(patch, imagetminus, imagetplus, timepoint, imagey, imagex):
    starttime = timepoint - imagetminus - 1
    endtime = timepoint + imagetplus
    smallimg = patch[starttime:endtime, :]

    return smallimg

def chunk_list(image, patchshape, stride, pair):
        rowstart = pair[0]
        colstart = pair[1]

        endrow = rowstart + patchshape[0]
        endcol = colstart + patchshape[1]

        if endrow > image.shape[1]:
            endrow = image.shape[1]
        if endcol > image.shape[2]:
            endcol = image.shape[2]

        region = (slice(0, image.shape[0]), slice(rowstart, endrow),
                  slice(colstart, endcol))

        # The actual pixels in that region.
        patch = image[region]

        # Always normalize patch that goes into the netowrk for getting a prediction score

        return patch, rowstart, colstart
class EventViewer(object):

    def __init__(self, viewer, image, event_name, key_categories, imagename, savedir, canvas, ax, figure, yolo_v2):

        self.viewer = viewer
        self.image = image
        self.event_name = event_name
        self.imagename = imagename
        self.canvas = canvas
        self.key_categories = key_categories
        self.savedir = savedir
        self.ax = ax
        self.yolo_v2 = yolo_v2
        self.figure = figure
        self.plot()

    def plot(self):

        self.ax.cla()

        for (event_name, event_label) in self.key_categories.items():
            if event_label > 0 and self.event_name == event_name:
                csvname = self.savedir + "/" + event_name + "Location" + (
                            os.path.splitext(os.path.basename(self.imagename))[0] + '.csv')
                event_locations, size_locations, angle_locations, line_locations, timelist, eventlist = self.event_counter(
                    csvname)

                for layer in list(self.viewer.layers):
                    if event_name in layer.name or layer.name in event_name or event_name + 'angle' in layer.name or layer.name in event_name + 'angle':
                        self.viewer.layers.remove(layer)
                    if 'Image' in layer.name or layer.name in 'Image':
                        self.viewer.layers.remove(layer)
                self.viewer.add_image(self.image, name='Image')
                self.viewer.add_points(np.asarray(event_locations), size=size_locations, name=event_name,
                                       face_color=[0] * 4, edge_color="red", edge_width=1)
                if self.yolo_v2:
                    self.viewer.add_shapes(np.asarray(line_locations), name=event_name + 'angle', shape_type='line',
                                           face_color=[0] * 4, edge_color="red", edge_width=1)
                self.viewer.theme = 'light'
                self.ax.plot(timelist, eventlist, '-r')
                self.ax.set_title(event_name + "Events")
                self.ax.set_xlabel("Time")
                self.ax.set_ylabel("Counts")
                self.figure.canvas.draw()
                self.figure.canvas.flush_events()
                plt.savefig(self.savedir + event_name + '.png')

    def event_counter(self, csv_file):

        time, y, x, score, size, confidence, angle = np.loadtxt(csv_file, delimiter=',', skiprows=1, unpack=True)

        radius = 10
        eventcounter = 0
        eventlist = []
        timelist = []
        listtime = time.tolist()
        listy = y.tolist()
        listx = x.tolist()
        listsize = size.tolist()
        listangle = angle.tolist()

        event_locations = []
        size_locations = []
        angle_locations = []
        line_locations = []
        for i in range(len(listtime)):
            tcenter = int(listtime[i])
            print(tcenter)
            ycenter = listy[i]
            xcenter = listx[i]
            size = listsize[i]
            angle = listangle[i]
            eventcounter = listtime.count(tcenter)
            timelist.append(tcenter)
            eventlist.append(eventcounter)

            event_locations.append([tcenter, ycenter, xcenter])
            size_locations.append(size)

            xstart = xcenter + radius * math.cos(angle)
            xend = xcenter - radius * math.cos(angle)

            ystart = ycenter + radius * math.sin(angle)
            yend = ycenter - radius * math.sin(angle)
            line_locations.append([[tcenter, ystart, xstart], [tcenter, yend, xend]])
            angle_locations.append(angle)

        return event_locations, size_locations, angle_locations, line_locations, timelist, eventlist
