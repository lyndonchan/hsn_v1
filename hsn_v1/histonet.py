import os
import keras
import numpy as np
from keras.models import model_from_json
from keras import optimizers
import scipy
from scipy import io

class HistoNet:
    """Class for implementing the classification CNN stage (HistoNet)"""

    def __init__(self, params):
        # Set constant parameters
        self.train_mean = 193.09203
        self.train_std = 56.450138

        # Set user-defined parameters
        self.model_dir = params['model_dir']
        self.model_name = params['model_name']
        self.batch_size = params['batch_size']
        self.relevant_inds = params['relevant_inds']
        self.input_name = params['input_name']
        self.class_names = params['class_names']

    def build_model(self):
        """Load model architecture, weights from file and compile the model"""

        # Load architecture from json
        model_json_path = os.path.join(self.model_dir, self.model_name + '.json')
        json_file = open(model_json_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)

        # Load weights from h5
        model_h5_path = os.path.join(self.model_dir, self.model_name + '.h5')
        self.model.load_weights(model_h5_path)

        # Evaluate model
        opt = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['binary_accuracy'])

    def normalize_image(self, X, is_glas=False):
        """Normalize the input images

        Parameters
        ----------
        X : numpy 3D array (size: W x H x 3)
            The input image, before normalizing
        is_glas : bool, optional
            True if segmenting GlaS images, False otherwise

        Returns
        -------
        Y : numpy 3D array (size: W x H x 3)
            The input image, after normalizing
        """

        if is_glas:
            # Clip values between 0 and 255
            X = np.clip(X, 0, 255)
        # Zero-mean, unit-variance normalization
        Y = (X - self.train_mean) / (self.train_std + 1e-7)
        return Y

    def load_thresholds(self, thresh_dir, model_name):
        """Load confidence score thresholds from file

        Parameters
        ----------
        thresh_dir : str
            File path to the directory holding the threshold file
        model_name : str
            The name of the model
        """

        thresh_path = os.path.join(thresh_dir, model_name)
        tmp = scipy.io.loadmat(thresh_path)
        self.thresholds = tmp.get('optimalScoreThresh')

    def predict(self, input_images, is_glas=False):
        """Predict classification CNN confidence scores on input images

        Parameters
        ----------
        input_images : numpy array (size: self.batch_size x W x H x 3)
            Input images, single batch
        is_glas : bool, optional
            True if segmenting GlaS images, False otherwise
        Returns
        -------
        pass_threshold_image_inds : numpy 1D array (size: num_pass_threshold)
            The indices of the images
        pass_threshold_class_inds : numpy 1D array (size: num_pass_threshold)
            The indices of the predicted classes
        pass_threshold_scores : numpy 1D array (size: num_pass_threshold)
            The scores of the predicted classes
        """
        predicted_scores = self.model.predict(input_images, batch_size=self.batch_size)
        is_pass_threshold = np.greater_equal(predicted_scores, self.thresholds)
        if is_glas:
            exocrine_class_ind = self.class_names.index('G.O')
            is_pass_threshold[:, exocrine_class_ind] = True #
        (pass_threshold_image_inds, pass_threshold_class_inds) = np.where(is_pass_threshold)
        pass_threshold_scores = predicted_scores[is_pass_threshold]

        is_class_in_level3 = np.array([np.isin(x, self.relevant_inds) for i,x in enumerate(pass_threshold_class_inds)])
        pass_threshold_image_inds = pass_threshold_image_inds[is_class_in_level3]
        pass_threshold_class_inds = pass_threshold_class_inds[is_class_in_level3]
        pass_threshold_scores = pass_threshold_scores[is_class_in_level3]

        return pass_threshold_image_inds, pass_threshold_class_inds, pass_threshold_scores

    def split_by_htt_class(self, pred_image_inds, pred_class_inds, pred_scores, htt_mode, atlas):
        """Split predicted classes into morphological and functional classes

        Parameters
        ----------
        pred_image_inds : numpy 1D array (size: num_pass_threshold)
            The indices of the images
        pred_class_inds : numpy 1D array (size: num_pass_threshold)
            The indices of the predicted classes
        pred_scores : numpy 1D array (size: num_pass_threshold)
            The scores of the predicted classes
        htt_class : str
            The type of segmentation set to solve
        atlas : hsn_v1.adp.Atlas object
            The Atlas of Digital Pathology object

        Returns
        -------
        httclass_pred_image_inds :
        httclass_pred_class_inds :
        httclass_pred_scores :
        """

        httclass_pred_image_inds = []
        httclass_pred_class_inds = []
        httclass_pred_scores = []

        if htt_mode in ['glas']:
            glas_serial_inds = [i for i, x in enumerate(pred_class_inds) if atlas.level5[x] in atlas.glas_valid_classes]
            httclass_pred_image_inds.append(pred_image_inds[glas_serial_inds])
            pred_valid_class_inds = atlas.convert_class_inds(pred_class_inds[glas_serial_inds], atlas.level5,
                                                             atlas.glas_valid_classes)
            httclass_pred_class_inds.append(pred_valid_class_inds)
            httclass_pred_scores.append(pred_scores[glas_serial_inds])
        if htt_mode in ['both', 'morph']:
            morph_serial_inds = [i for i, x in enumerate(pred_class_inds) if atlas.level5[x] in
                                 atlas.morph_valid_classes]
            httclass_pred_image_inds.append(pred_image_inds[morph_serial_inds])
            pred_valid_class_inds = atlas.convert_class_inds(pred_class_inds[morph_serial_inds], atlas.level5,
                                                             atlas.morph_valid_classes)
            httclass_pred_class_inds.append(pred_valid_class_inds)
            httclass_pred_scores.append(pred_scores[morph_serial_inds])
        if htt_mode in ['both', 'func']:
            func_serial_inds = [i for i, x in enumerate(pred_class_inds) if atlas.level5[x] in atlas.func_valid_classes]
            httclass_pred_image_inds.append(pred_image_inds[func_serial_inds])
            pred_valid_class_inds = atlas.convert_class_inds(pred_class_inds[func_serial_inds], atlas.level5,
                                                             atlas.func_valid_classes)
            httclass_pred_class_inds.append(pred_valid_class_inds)
            httclass_pred_scores.append(pred_scores[func_serial_inds])

        if htt_mode == 'both' and sum([x.shape[0] for x in httclass_pred_image_inds]) != len(pred_image_inds):
            raise Exception('Error splitting Grad-CAM into HTT-class-specific Grad-CAMs: in and out sizes don\'t match')
        return httclass_pred_image_inds, httclass_pred_class_inds, httclass_pred_scores

    def find_final_layer(self):
        """Find the layer index of the last activation layer before the flatten layer"""
        is_after_flatten = False
        for iter_layer, layer in reversed(list(enumerate(self.model.layers))):
            if type(layer) == keras.layers.core.Flatten:
                is_after_flatten = True
            if is_after_flatten and type(layer) == keras.layers.core.Activation:
                return layer.name
        raise Exception('Could not find the final layer in provided HistoNet')