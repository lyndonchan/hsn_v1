import keras.backend as K
import tensorflow as tf
import numpy as np
import cv2
import time
import os
from scipy.ndimage import gaussian_filter
import scipy
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, params):
        self.htt_mode = params['htt_mode']
        self.size = params['size']
        self.num_imgs = params['num_imgs']
        self.batch_size = params['batch_size']
        self.cnn_model = params['cnn_model']
        self.final_layer = params['final_layer']
        self.tmp_dir = params['tmp_dir']
        self.should_normalize = True

    def gen_gradcam(self, pred_image_inds, pred_class_inds, pred_scores, input_images_norm, atlas, valid_classes):

        # This is the number of HTTs across all images that passed their thresholds
        num_pass_threshold = len(pred_image_inds)

        BATCH_SIZES = [48]
        for iter_batchsize in range(len(BATCH_SIZES)):
            # start_time = time.time()
            self.batch_size = BATCH_SIZES[iter_batchsize]
            gradcam = np.zeros((num_pass_threshold, self.size[0], self.size[1]))
            num_batches = (num_pass_threshold + self.batch_size - 1) // self.batch_size
            pred_scores_3d = np.expand_dims(np.expand_dims(pred_scores, axis=1), axis=1)

            pred_class_inds_full = atlas.convert_class_inds(pred_class_inds, valid_classes, atlas.level5)
            for iter_batch in range(num_batches):
                start = iter_batch * self.batch_size
                end = min((iter_batch + 1) * self.batch_size, num_pass_threshold)
                cur_gradcam_batch = self.grad_cam_batch(self.cnn_model, input_images_norm[pred_image_inds[start:end]],
                                                        pred_class_inds_full[start:end], self.final_layer)
                if self.should_normalize:
                    gradcam[start:end] = cur_gradcam_batch * pred_scores_3d[start:end]
                else:
                    gradcam[start:end] = cur_gradcam_batch
                # print('Batch size: %d (%s seconds)' % (self.batch_size, time.time() - start_time))
        return gradcam

    def grad_cam_batch(self, input_model, images, classes, layer_name):
        # start_time = time.time()
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
        # print('Grad-CAM getting CAM time: %s seconds' % (time.time() - start_time))

        # start_time = time.time()
        new_cams = np.empty((images.shape[0], images.shape[1], images.shape[2]))
        heatmaps = np.empty((images.shape[0], images.shape[1], images.shape[2]))
        for i in range(cams.shape[0]):
            new_cams[i] = cv2.resize(cams[i], (self.size[0], self.size[1]))
            if self.should_normalize:
                new_cams[i] = np.maximum(new_cams[i], 0)
                heatmaps[i] = new_cams[i] / np.maximum(np.max(new_cams[i]), 1e-7)
            else:
                heatmaps[i] = new_cams[i]
        # print('Grad-CAM resizing CAM time: %s seconds' % (time.time() - start_time))
        return heatmaps

    def expand_image_wise(self, gradcam_serial, pred_image_inds, pred_class_inds, valid_classes):
        gradcam_image_wise = np.zeros((self.num_imgs, len(valid_classes), self.size[0], self.size[1]))
        for iter_input_file in range(self.num_imgs):
            # Convert serial indices to valid out indices
            cur_serial_inds = [i for i, x in enumerate(pred_image_inds) if x == iter_input_file]
            cur_class_inds = pred_class_inds[cur_serial_inds]
            if len(cur_class_inds) > 0:
                gradcam_image_wise[iter_input_file, cur_class_inds] = gradcam_serial[cur_serial_inds]
            a=1
        return gradcam_image_wise

    def modify_by_htt(self, gradcam, images, atlas, htt_class, gradcam_adipose=None):
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

        def img_sigmoid(X, X_mult=2, X_shift=240):
            return 1 / (1 + scipy.special.expit(-X_mult * (X - X_shift)))
        # Y = img_sigmoid(np.array(range(255)), 0.1, 1, 240)
        # plt.plot(Y)

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

    def get_cs_gradcam(self, gradcam, atlas, htt_class, mode=1, mat_mode=1):
        if htt_class == 'func':
            other_ind = atlas.func_valid_classes.index('Other')
        elif htt_class == 'glas':
            other_ind = atlas.glas_valid_classes.index('Other')

        if mode == 0:   # Original
            class_inds = range(gradcam.shape[1])
            cs_gradcam = gradcam[:]
            for iter_class in range(gradcam.shape[1]):
                if not (htt_class in ['func', 'glas'] and iter_class == other_ind):
                    cs_gradcam[:, iter_class] -= np.max(gradcam[:, np.delete(class_inds, iter_class)], axis=1)
            cs_gradcam = np.clip(cs_gradcam, 0, 1)
        elif mode == 1: # Experimental
            # - Find max difference value, ind map
            gradcam_sorted = np.sort(gradcam, axis=1)
            maxdiff = gradcam_sorted[:, -1] - gradcam_sorted[:, -2]
            maxind = np.argmax(gradcam, axis=1)
            # - Find CS-Grad-CAM
            if mat_mode == 0:   # Original
                cs_gradcam = np.zeros_like(gradcam)
            elif mat_mode == 1: # Experimental
                cs_gradcam = np.transpose(np.tile(np.expand_dims(maxdiff, axis=-1), gradcam.shape[1]), (0, 3, 1, 2))
            for iter_class in range(gradcam.shape[1]):
                if not (htt_class in ['func', 'glas'] and iter_class == other_ind):
                    if mat_mode == 0:
                        cs_gradcam[:, iter_class] = maxdiff * (maxind == iter_class)
                    elif mat_mode == 1:
                        cs_gradcam[:, iter_class] *= (maxind == iter_class)
                else:
                    cs_gradcam[:, iter_class] = gradcam[:, iter_class]
        return cs_gradcam