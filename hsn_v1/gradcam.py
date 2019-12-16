import keras.backend as K
import tensorflow as tf
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import scipy
import matplotlib.pyplot as plt

class GradCAM:
    """Class for Grad-CAM and HTT modifications"""

    def __init__(self, params):
        self.htt_mode = params['htt_mode']
        self.size = params['size']
        self.num_imgs = params['num_imgs']
        self.batch_size = params['batch_size']
        self.cnn_model = params['cnn_model']
        self.final_layer = params['final_layer']
        self.tmp_dir = params['tmp_dir']

    def gen_gradcam(self, pred_image_inds, pred_class_inds, pred_scores, input_images_norm, atlas, valid_classes):
        """Generate Grad-CAM

        Parameters
        ----------
        pred_image_inds : numpy 1D array (size: num_pass_threshold)
            The indices of the images
        pred_class_inds : numpy 1D array (size: num_pass_threshold)
            The indices of the predicted classes
        pred_scores : numpy 1D array (size: num_pass_threshold)
            The scores of the predicted classes
        input_images_norm : numpy 4D array (size: B x H x W x 3)
            The normalized input images
        atlas : hsn_v1.adp.Atlas object
            The Atlas of Digital Pathology object
        valid_classes : list
            The segmentation classes valid for the current problem

        Returns
        -------
        gradcam : numpy 3D array (size: num_pass_threshold x H x W)
            The Grad-CAM continuous values for predicted images/classes of the current batch
        """

        # Find number of HTTs across all images that passed their thresholds
        num_pass_threshold = len(pred_image_inds)

        gradcam = np.zeros((num_pass_threshold, self.size[0], self.size[1]))
        num_batches = (num_pass_threshold + self.batch_size - 1) // self.batch_size
        pred_scores_3d = np.expand_dims(np.expand_dims(pred_scores, axis=1), axis=1)
        pred_class_inds_full = atlas.convert_class_inds(pred_class_inds, valid_classes, atlas.level5)

        # For each batch, obtain Grad-CAM, then multiply by confidence score
        for iter_batch in range(num_batches):
            start = iter_batch * self.batch_size
            end = min((iter_batch + 1) * self.batch_size, num_pass_threshold)
            cur_gradcam_batch = self.grad_cam_batch(self.cnn_model, input_images_norm[pred_image_inds[start:end]],
                                                    pred_class_inds_full[start:end], self.final_layer)
            gradcam[start:end] = cur_gradcam_batch * pred_scores_3d[start:end]
        return gradcam

    def grad_cam_batch(self, input_model, images, classes, layer_name):
        """Generate Grad-CAM for a single batch of images

        Parameters
        ----------
        input_model : keras.engine.sequential.Sequential object
            The input model to run Grad-CAM on
        images : numpy 4D array (size: B x H x W x 3)
            The normalized input images in the current batch
        classes : numpy 1D array
            The indices of the predicted classes in the current batch
        layer_name : str
            The name of the model layer to run Grad-CAM on

        Returns
        -------
        heatmaps : numpy 3D array (size: B x H x W)
            The generated Grad-CAM for the current batch
        """

        y_c = tf.gather_nd(input_model.layers[-2].output, np.dstack([range(images.shape[0]), classes])[0])
        conv_output = input_model.get_layer(layer_name).output

        def normalize(x):
            # utility function to normalize a tensor by its L2 norm
            return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

        grads = normalize(K.gradients(y_c, conv_output))[0]
        gradient_function = K.function([input_model.layers[0].input], [conv_output, grads])

        output, grads_val = gradient_function([images, 0])
        weights = np.mean(grads_val, axis=(1, 2))
        cams = np.einsum('ijkl,il->ijk', output, weights)

        new_cams = np.empty((images.shape[0], images.shape[1], images.shape[2]))
        heatmaps = np.empty((images.shape[0], images.shape[1], images.shape[2]))
        for i in range(cams.shape[0]):
            new_cams[i] = cv2.resize(cams[i], (self.size[0], self.size[1]))
            new_cams[i] = np.maximum(new_cams[i], 0)
            heatmaps[i] = new_cams[i] / np.maximum(np.max(new_cams[i]), 1e-7)

        return heatmaps

    def expand_image_wise(self, gradcam_serial, pred_image_inds, pred_class_inds, valid_classes):
        """Expand the serialized Grad-CAM into 4D array, i.e. insert arrays of zeroes for unpredicted classes

        Parameters
        ----------
        gradcam_serial : numpy 3D array (size: self.num_imgs x W x H)
            The generated Grad-CAMs for predicted classes in the current batch, in serial form
        pred_image_inds : numpy 1D array (size: self.num_imgs)
            The indices of the images in the current batch, in serial form
        pred_class_inds : numpy 1D array (size: self.num_imgs)
            The indices of the predicted classes in the current batch, in serial form
        valid_classes : list
            The segmentation classes valid for the current problem

        Returns
        -------
        gradcam_image_wise : numpy 4D array (size: self.num_imgs x C x H x W), where C = number of classes
            The serialized Grad-CAM for the current batch
        """

        gradcam_image_wise = np.zeros((self.num_imgs, len(valid_classes), self.size[0], self.size[1]))
        for iter_input_file in range(self.num_imgs):
            # Convert serial indices to valid out indices
            cur_serial_inds = [i for i, x in enumerate(pred_image_inds) if x == iter_input_file]
            cur_class_inds = pred_class_inds[cur_serial_inds]
            if len(cur_class_inds) > 0:
                gradcam_image_wise[iter_input_file, cur_class_inds] = gradcam_serial[cur_serial_inds]
        return gradcam_image_wise

    def modify_by_htt(self, gradcam, images, atlas, htt_class, gradcam_adipose=None):
        """Generates non-foreground class activations and appends to the foreground class activations

        Parameters
        ----------
        gradcam : numpy 4D array (size: self.batch_size x C x W x H), where C = number of classes
            The serialized Grad-CAM for the current batch
        images : numpy 3D array (size: self.batch_size x W x H x 3)
            The input images for the current batch
        atlas : hsn_v1.adp.Atlas object
            The Atlas of Digital Pathology object
        htt_class : str
            The type of segmentation set to solve
        gradcam_adipose : numpy 4D array (size: self.num_imgs x C x H x W), where C = number of classes,
                          or None, optional
            Adipose class Grad-CAM (if segmenting functional types) or None (if not segmenting functional types)

        Returns
        -------
        gradcam : numpy 4D array (size: self.batch_size x C x W x H), where C = number of classes
            The modified Grad-CAM for the current batch, with non-foreground class activations appended
        """
        if htt_class == 'morph':
            background_max = 0.75
            background_exception_classes = ['A.W', 'A.B', 'A.M']
            classes = atlas.morph_valid_classes
        elif htt_class == 'func':
            background_max = 0.75
            other_tissue_mult = 0.05
            background_exception_classes = ['G.O', 'G.N', 'T']
            classes = atlas.func_valid_classes
            if gradcam_adipose is None:
                raise Exception('You must feed in adipose heatmap for functional type')
            other_ind = classes.index('Other')
        elif htt_class == 'glas':
            other_tissue_mult = 0.05
            classes = atlas.glas_valid_classes
            other_ind = classes.index('Other')
            # Get other tissue class prediction
            other_moh = np.max(gradcam, axis=1)
            other_gradcam = np.expand_dims(other_tissue_mult * (1 - other_moh), axis=1)
            other_gradcam = np.max(other_gradcam, axis=1)
            other_gradcam = np.clip(other_gradcam, 0, 1)
            gradcam[:, other_ind] = other_gradcam

        if htt_class in ['morph', 'func']:
            background_ind = classes.index('Background')

            # Get background class prediction
            sigmoid_input = 4 * (np.mean(images, axis=-1) - 240)
            background_gradcam = background_max * scipy.special.expit(sigmoid_input)
            background_exception_cur_inds = [i for i, x in enumerate(classes) if x in background_exception_classes]
            for iter_input_image in range(background_gradcam.shape[0]):
                background_gradcam[iter_input_image] = gaussian_filter(background_gradcam[iter_input_image], sigma=2)
            background_gradcam -= np.max(gradcam[:, background_exception_cur_inds], axis=1)
            background_gradcam = np.clip(background_gradcam, 0, 1)
            gradcam[:, background_ind] = background_gradcam

            # Get other tissue class prediction
            if htt_class == 'func':
                other_moh = np.max(gradcam, axis=1)
                other_gradcam = np.expand_dims(other_tissue_mult * (1 - other_moh), axis=1)
                other_gradcam = np.max(np.concatenate((other_gradcam, gradcam_adipose), axis=1), axis=1)
                other_gradcam = np.clip(other_gradcam, 0, 1)
                gradcam[:, other_ind] = other_gradcam
        return gradcam

    def get_cs_gradcam(self, gradcam, atlas, htt_class):
        """Performs class subtraction operation to modified Grad-CAM

        Parameters
        ----------
        gradcam : numpy 4D array (size: self.batch_size x C x W x H), where C = number of classes
            The modified Grad-CAM for the current batch, with non-foreground class activations appended
        atlas : hsn_v1.adp.Atlas object
            The Atlas of Digital Pathology object
        htt_class : str
            The type of segmentation set to solve

        Returns
        -------
        cs_gradcam : numpy 4D array (size: self.batch_size x C x W x H), where C = number of classes
            The class-subtracted Grad-CAM for the current batch
        """

        if htt_class == 'func':
            classes = atlas.func_valid_classes
            other_ind = classes.index('Other')
        elif htt_class == 'glas':
            classes = atlas.glas_valid_classes
            other_ind = classes.index('Other')
        class_inds = range(gradcam.shape[1])
        cs_gradcam = gradcam
        for iter_class in range(gradcam.shape[1]):
            if not (htt_class in ['func', 'glas'] and iter_class == other_ind):
                cs_gradcam[:, iter_class] -= np.max(gradcam[:, np.delete(class_inds, iter_class)], axis=1)
        cs_gradcam = np.clip(cs_gradcam, 0, 1)
        return cs_gradcam