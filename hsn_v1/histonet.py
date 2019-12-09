import os
import keras
import numpy as np
from keras.models import model_from_json
from keras import optimizers
import scipy
from scipy import io

class HistoNet:
    def __init__(self, params):
        # Set constant parameters
        # TODO: extract mean, std dynamically
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

    # Image normalization (zero-mean, unit-variance)
    def normalize_image(self, X, is_glas=False):
        if is_glas:
            # X = X - np.expand_dims(np.expand_dims(np.array([5.8604, 5.8955, -0.1963]), axis=0), axis=0)
            X = np.clip(X, 0, 255)
        Y = (X - self.train_mean) / (self.train_std + 1e-7)
        return Y

    # Load HTT confidence score thresholds
    def load_thresholds(self, thresh_dir, model_name):
        thresh_path = os.path.join(thresh_dir, model_name)
        tmp = scipy.io.loadmat(thresh_path)
        self.thresholds = tmp.get('optimalScoreThresh')

    # Obtain HistoNet confidence scores on a set of input images
    def predict(self, input_images, is_glas=False):
        predicted_scores = self.model.predict(input_images, batch_size=self.batch_size)
        is_pass_threshold = np.greater_equal(predicted_scores, self.thresholds)
        if is_glas:
            exocrine_class_ind = self.class_names.index('G.O')
            # predicted_scores[:, exocrine_class_ind] = 1 ##
            # is_pass_threshold = np.greater_equal(predicted_scores, self.thresholds) ##
            is_pass_threshold[:, exocrine_class_ind] = True #
        (pass_threshold_image_inds, pass_threshold_class_inds) = np.where(is_pass_threshold)
        pass_threshold_scores = predicted_scores[is_pass_threshold]

        is_class_in_level3 = np.array([np.isin(x, self.relevant_inds) for i,x in enumerate(pass_threshold_class_inds)])
        pass_threshold_image_inds = pass_threshold_image_inds[is_class_in_level3]
        pass_threshold_class_inds = pass_threshold_class_inds[is_class_in_level3]
        pass_threshold_scores = pass_threshold_scores[is_class_in_level3]

        return pass_threshold_image_inds, pass_threshold_class_inds, pass_threshold_scores, predicted_scores

    def split_by_htt_class(self, pred_image_inds, pred_class_inds, pred_scores, htt_mode, atlas):
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

    # Find the layer index of the last activation layer before the flatten layer
    def find_final_layer(self):
        is_after_flatten = False
        for iter_layer, layer in reversed(list(enumerate(self.model.layers))):
            if type(layer) == keras.layers.core.Flatten:
                is_after_flatten = True
            if is_after_flatten and type(layer) == keras.layers.core.Activation:
                return layer.name
        raise Exception('Could not find the final layer in provided HistoNet')