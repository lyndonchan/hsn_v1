import numpy as np
import os
from scipy.ndimage import gaussian_filter
import pandas as pd
import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import time
import math
from skimage import measure
from skimage import filters

import tensorflow as tf
from keras import backend as K

from .adp import Atlas
from .utilities import *
from .histonet import HistoNet
from .gradcam import GradCAM
from .densecrf import DenseCRF

class HistoSegNetV1:
    def __init__(self, params):
        self.input_name = params['input_name']
        self.input_size = params['input_size']
        self.input_mode = params['input_mode']
        self.down_fac = params['down_fac']
        self.batch_size = params['batch_size']
        self.htt_mode = params['htt_mode']
        self.gt_mode = params['gt_mode']
        self.run_level = params['run_level']
        self.save_types = params['save_types']
        self.verbosity = params['verbosity']

        if len(self.input_size) != 2:
            raise Exception('User-defined variable input_size must be a list of length 2!')
        if type(self.input_size[0]) != int or self.input_size[0] < 1 or \
           type(self.input_size[1]) != int or self.input_size[1] < 1:
            raise Exception('User-defined parameter input_size is either non-integer or less than 1')
        if self.input_mode not in ['wsi', 'patch']:
            raise Exception('User-defined parameter input_mode is neither \'wsi\' nor \'patch\'')
        if type(self.batch_size) != int and self.batch_size <= 1:
            raise Exception('User-defined variable batch_size ' + self.batch_size +
                            ' is either non-integer or less than 1')
        if self.htt_mode not in ['both', 'morph', 'func', 'glas']:
            raise Exception(
                'User-defined parameter htt_mode ' + self.htt_mode +
                ' not in {\'both\', \'morph\', \'func\', \'glas\'}')
        if self.gt_mode not in ['on', 'off']:
            raise Exception('User-defined variable gt_mode ' + self.gt_mode + ' is not in {\'on\', \'off\'}')
        if self.run_level not in [1, 2, 3]:
            raise Exception('User-defined variable run_level ' + self.run_level + ' is not in [1, 2, 3]')
        if len(self.save_types) != 4:
            raise Exception('User-defined variable save_level ' + self.save_level + ' not of length 4')
        if self.verbosity not in ['NORMAL', 'QUIET']:
            raise Exception('User-defined variable verbosity ' + self.verbosity + ' is not in {\'NORMAL\', \'QUIET\'}')

        # Define folder paths
        cur_path = os.path.abspath(os.path.curdir)
        self.data_dir = os.path.join(cur_path, 'data')
        self.gt_dir = os.path.join(cur_path, 'gt')
        self.img_dir = os.path.join(cur_path, 'img')
        self.tmp_dir = os.path.join(cur_path, 'tmp', self.input_name)
        self.out_dir = os.path.join(cur_path, 'out', self.input_name)
        input_dir = os.path.join(self.img_dir, self.input_name)
        if not os.path.exists(input_dir):
            raise Exception('Could not find user-defined input directory ' + input_dir)

        # Create folders if they don't exist
        mkdir_if_nexist(self.tmp_dir)
        mkdir_if_nexist(self.out_dir)

        # Read in pre-defined ADP taxonomy
        self.atlas = Atlas()

        # Define valid classes and colours
        self.httclass_valid_classes = []
        self.httclass_valid_colours = []
        if self.htt_mode in ['glas']:
            self.httclass_valid_classes.append(self.atlas.glas_valid_classes)
            self.httclass_valid_colours.append(self.atlas.glas_valid_colours)
        if self.htt_mode in ['both', 'morph']:
            self.httclass_valid_classes.append(self.atlas.morph_valid_classes)
            self.httclass_valid_colours.append(self.atlas.morph_valid_colours)
        if self.htt_mode in ['both', 'func']:
            self.httclass_valid_classes.append(self.atlas.func_valid_classes)
            self.httclass_valid_colours.append(self.atlas.func_valid_colours)

        # Define GT paths
        self.htt_classes = []
        if self.gt_mode == 'on':
            self.httclass_gt_dirs = []
            self.httclass_intersect_counts = []
            self.httclass_union_counts = []
        if self.htt_mode in ['glas']:
            self.htt_classes.append('glas')
            if self.gt_mode == 'on':
                glas_gt_dir = os.path.join(self.gt_dir, self.input_name, self.htt_mode)
                if not os.path.exists(glas_gt_dir):
                    raise Exception('GlaS GT directory does not exist: ' + glas_gt_dir)
                self.httclass_gt_dirs.append(glas_gt_dir)
                self.httclass_intersect_counts.append(np.zeros((len(self.atlas.glas_valid_classes))))
                self.httclass_union_counts.append(np.zeros((len(self.atlas.glas_valid_classes))))
            self.glas_confscores = []
        if self.htt_mode in ['both', 'morph']:
            self.htt_classes.append('morph')
            if self.gt_mode == 'on':
                morph_gt_dir = os.path.join(self.gt_dir, self.input_name, 'morph')
                if not os.path.exists(morph_gt_dir):
                    raise Exception('Morph GT directory does not exist: ' + morph_gt_dir)
                self.httclass_gt_dirs.append(morph_gt_dir)
                self.httclass_intersect_counts.append(np.zeros((len(self.atlas.morph_valid_classes))))
                self.httclass_union_counts.append(np.zeros((len(self.atlas.morph_valid_classes))))
        if self.htt_mode in ['both', 'func']:
            self.htt_classes.append('func')
            if self.gt_mode == 'on':
                func_gt_dir = os.path.join(self.gt_dir, self.input_name, 'func')
                if not os.path.exists(func_gt_dir):
                    raise Exception('Func GT directory does not exist: ' + func_gt_dir)
                self.httclass_gt_dirs.append(func_gt_dir)
                self.httclass_intersect_counts.append(np.zeros((len(self.atlas.func_valid_classes))))
                self.httclass_union_counts.append(np.zeros((len(self.atlas.func_valid_classes))))

    def find_img(self):
        # Find images
        if self.verbosity == 'NORMAL':
            print('Finding images', end='')
            start_time = time.time()
        input_dir = os.path.join(self.img_dir, self.input_name)
        if self.input_mode == 'patch':
            self.input_files_all = [x for x in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, x)) and
                                    os.path.splitext(x)[-1].lower() == '.png']
        elif self.input_mode == 'wsi':
            self.input_files_all = [x for x in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, x)) and
                                    # os.path.splitext(x)[-1].lower() == '.png' and
                                    os.path.splitext(x)[0].split('_f')[1] == '1']
        if self.verbosity == 'NORMAL':
            print(' (%s seconds)' % (time.time() - start_time))

    def analyze_img(self):
        # Find dataset mean and standard deviation
        # images_as_vec = []
        # for iter_input_file, input_file in enumerate(self.input_files_all):
        #     input_path = os.path.join(self.img_dir, self.input_name, input_file)
        #     if not os.path.exists(input_path):
        #         raise Exception('Requested input path ' + input_path + ' does not exist')
        #     images_as_vec.append(read_image(input_path).flatten())
        # self.img_mean = np.average(images_as_vec)
        # self.img_std = np.std(images_as_vec)

        # Find HTT log inverse frequencies
        if self.gt_mode == 'on':
            def convert_to_log_freq(x):
                is_zero = np.where(x == 0)
                x_log = np.log(x)
                x_log[is_zero] = 0
                y = np.sum(x_log) / x_log
                y[is_zero] = 0
                y = y / np.sum(y)
                return y
            self.httclass_loginvfreq = []
            for iter_httclass, htt_class in enumerate(self.htt_classes):
                httweights_path = os.path.join(self.tmp_dir, 'httweights_' + htt_class + '.npy')
                if not os.path.exists(httweights_path):
                    num_classes = len(self.httclass_valid_classes[iter_httclass])
                    gt_counts = np.zeros((num_classes))
                    for iter_input_file, input_file in enumerate(self.input_files_all):
                        gt_segmask_path = os.path.join(self.httclass_gt_dirs[iter_httclass], input_file)
                        gt_segmask = read_image(gt_segmask_path)
                        for iter_class in range(num_classes):
                            cur_class_mask = np.all(gt_segmask ==
                                                    self.httclass_valid_colours[iter_httclass][iter_class], axis=-1)
                            gt_counts[iter_class] += np.sum(cur_class_mask)
                    self.httclass_loginvfreq.append(convert_to_log_freq(gt_counts))
                    np.save(httweights_path, self.httclass_loginvfreq[iter_httclass])
                else:
                    self.httclass_loginvfreq.append(np.load(httweights_path))
        a=1

    def load_histonet(self, params):
        # Save user-defined settings
        self.model_name = params['model_name']

        # Validate user-defined settings
        model_threshold_path = os.path.join(self.data_dir, self.model_name + '.mat')
        model_json_path = os.path.join(self.data_dir, self.model_name + '.json')
        model_h5_path = os.path.join(self.data_dir, self.model_name + '.h5')
        if not os.path.exists(model_threshold_path) or not os.path.exists(model_json_path) or not \
                os.path.exists(model_h5_path):
            raise Exception('The files corresopnding to user-defined model ' + self.model_name + ' do not exist in ' +
                            self.data_dir)

        if self.verbosity == 'NORMAL':
            print('Loading HistoNet', end='')
            start_time = time.time()
        # Load HistoNet
        self.hn = HistoNet(params={'model_dir': self.data_dir, 'model_name': self.model_name,
                                   'batch_size': self.batch_size, 'relevant_inds': self.atlas.level3_valid_inds,
                                   'input_name': self.input_name, 'class_names': self.atlas.level5})
        self.hn.build_model()

        # Load HistoNet HTT score thresholds
        self.hn.load_thresholds(self.data_dir, self.model_name)
        if self.verbosity == 'NORMAL':
            print(' (%s seconds)' % (time.time() - start_time))

    def run_batch(self):
        confscores = np.zeros((len(self.input_files_all), len(self.hn.class_names)))
        num_batches = (len(self.input_files_all) + self.batch_size - 1) // self.batch_size
        for iter_batch in range(num_batches):
        # for iter_batch in range(852, num_batches):
        # for iter_batch in range(17, 18):
            if self.verbosity == 'NORMAL':
                print('\tBatch #' + str(iter_batch + 1) + ' of ' + str(num_batches))
                batch_start_time = time.time()

            # a. Load image(s)expand_image_wise
            if self.verbosity == 'NORMAL':
                print('\t\tLoading images', end='')
                start_time = time.time()
            start = iter_batch * self.batch_size
            end = min((iter_batch + 1) * self.batch_size, len(self.input_files_all))
            self.input_files_batch = self.input_files_all[start:end]
            self.load_norm_imgs()
            if self.verbosity == 'NORMAL':
                print(' (%s seconds)' % (time.time() - start_time))

            # b. Load ground-truth data, if available
            if self.verbosity == 'NORMAL':
                print('\t\tLoading ground-truth data', end='')
                start_time = time.time()
            self.load_gt()
            if self.verbosity == 'NORMAL':
                print(' (%s seconds)' % (time.time() - start_time))

            # c. Segment image(s) with HistoSegNetV1, saving/loading to/from tmp files if so requested
            if self.verbosity == 'NORMAL':
                print('\t\tSegmenting images')
                start_time = time.time()
            self.segment_img()
            confscores[start:end] = self.confscores_batch
            if self.verbosity == 'NORMAL':
                print('\t\t(%s seconds)' % (time.time() - start_time))

            # d. Evaluate segmentation quality, if available
            if self.gt_mode == 'on' and self.run_level == 3:
                if self.verbosity == 'NORMAL':
                    print('\t\tEvaluating segmentation quality', end='')
                    start_time = time.time()
                self.eval_segmentation()
                if self.verbosity == 'NORMAL':
                    print(' (%s seconds)' % (time.time() - start_time))

            if self.verbosity == 'NORMAL':
                print('\t(%s seconds)' % (time.time() - batch_start_time))
            a=1

        # Save confidence scores for GlaS
        if self.htt_mode == 'glas' and len(self.glas_confscores) > 0:
            items = []
            for iter_image, file in enumerate(self.input_files_all):
                items.append((file, [self.glas_confscores[iter_image]]))
            glas_confscores_path = os.path.join(self.out_dir, 'glas_confscores.csv')
            res = pd.DataFrame.from_dict(dict(items))
            res.to_csv(glas_confscores_path)

        # Write confidence scores to CSV
        confscores_path = os.path.join(self.out_dir, self.input_name + '_confscores.csv')
        df = pd.DataFrame(data=confscores, index=self.input_files_all, columns=self.hn.class_names)
        df.to_csv(confscores_path)

        # Overlap and segment
        if self.run_level == 3 and self.input_mode == 'wsi':
            self.overlap_and_segment()

    def load_norm_imgs(self):
        input_dir = os.path.join(self.img_dir, self.input_name)
        # Load raw images
        self.orig_images = [None] * len(self.input_files_batch)
        self.orig_images_cropped = [None] * len(self.input_files_batch)
        self.orig_sizes = [None] * len(self.input_files_batch)
        self.crop_offsets = [None] * len(self.input_files_batch)
        self.num_crops = [None] * len(self.input_files_batch)
        for iter_input_file, input_file in enumerate(self.input_files_batch):
            input_path = os.path.join(input_dir, input_file)
            self.orig_images[iter_input_file] = read_image(input_path)
            self.orig_sizes[iter_input_file] = self.orig_images[iter_input_file].shape[:2]
            downsampled_size = [round(x / self.down_fac) for x in self.orig_sizes[iter_input_file]]

            # If downsampled image is smaller than the patch size, then mirror pad first, then downsample
            if downsampled_size[0] < self.input_size[0] or downsampled_size[1] < self.input_size[1]:
                pad_vert = math.ceil(max(self.input_size[0] * self.down_fac - self.orig_sizes[iter_input_file][0], 0) / 2)
                pad_horz = math.ceil(max(self.input_size[1] * self.down_fac - self.orig_sizes[iter_input_file][1], 0) / 2)
                downsampled_size[0] = round((self.orig_sizes[iter_input_file][0] + 2 * pad_vert) / self.down_fac)
                downsampled_size[1] = round((self.orig_sizes[iter_input_file][1] + 2 * pad_horz) / self.down_fac)
            self.num_crops[iter_input_file] = [math.ceil(downsampled_size[i] / self.input_size[i]) for i in range(2)]
        self.orig_images = np.array(self.orig_images)

        num_patches = sum([np.prod(np.array(x)) for x in self.num_crops])
        self.input_images = np.zeros((num_patches, self.input_size[0], self.input_size[1], 3))
        start = 0
        for iter_input_file in range(len(self.input_files_batch)):
            end = start + np.prod(np.array(self.num_crops[iter_input_file]))
            self.input_images[start:end], self.orig_images_cropped[iter_input_file] = crop_into_patches(
                self.orig_images[iter_input_file], self.down_fac, self.input_size)
            start += np.prod(np.array(self.num_crops[iter_input_file]))

        # Normalize images
        self.input_images_norm = np.zeros_like(self.input_images)
        for iter_input_image, input_image in enumerate(self.input_images):
            self.input_images_norm[iter_input_image] = self.hn.normalize_image(input_image, self.htt_mode=='glas')

    def load_gt(self):
        self.httclass_gt_segmasks = []
        self.httclass_gt_class_inds = [None] * len(self.htt_classes)
        if self.gt_mode == 'on':
            self.httclass_gt_legends = [None] * len(self.htt_classes)
        elif self.gt_mode == 'off':
            self.httclass_gt_legends = [[None] * len(self.input_files_batch)] * len(self.htt_classes)
        for iter_httclass, htt_class in enumerate(self.htt_classes):
            gt_segmasks = []
            # Load gt segmentation images
            if self.gt_mode == 'on':
                for iter_input_file, input_file in enumerate(self.input_files_batch):
                    gt_segmask_path = os.path.join(self.httclass_gt_dirs[iter_httclass], input_file)
                    gt_segmasks.append(read_segmask(gt_segmask_path, size=self.orig_sizes[iter_input_file]))
                # Load gt class labels
                self.httclass_gt_class_inds[iter_httclass] = segmask_to_class_inds(gt_segmasks,
                                                                           self.httclass_valid_colours[iter_httclass])
                # Load gt legend
                self.httclass_gt_legends[iter_httclass] = get_legends(self.httclass_gt_class_inds[iter_httclass],
                                                        self.orig_sizes[0], self.httclass_valid_classes[iter_httclass],
                                                        self.httclass_valid_colours[iter_httclass])
            elif self.gt_mode == 'off':
                for iter_input_file in range(len(self.input_files_batch)):
                    gt_segmasks.append(np.zeros((self.orig_sizes[iter_input_file][0],
                                                                        self.orig_sizes[iter_input_file][1], 3)))
                    self.httclass_gt_legends[iter_httclass][iter_input_file] = np.zeros((self.orig_sizes[iter_input_file][0],
                                                                        self.orig_sizes[iter_input_file][1], 3))
            self.httclass_gt_segmasks.append(gt_segmasks)
        self.httclass_gt_segmasks = np.array(self.httclass_gt_segmasks)
        self.httclass_gt_legends = np.array(self.httclass_gt_legends)

    def segment_img(self):
        # Obtain HistoNet confidence scores on input images
        if self.verbosity == 'NORMAL':
            print('\t\t\tApplying HistoNet', end='')
            start_time = time.time()
        pred_image_inds, pred_class_inds, pred_scores, self.confscores_batch = self.hn.predict(self.input_images_norm, self.htt_mode=='glas')
        if self.verbosity == 'NORMAL':
            print(' (%s seconds)' % (time.time() - start_time))

        # Split by HTT class
        if self.verbosity == 'NORMAL':
            print('\t\t\tSplitting by HTT class', end='')
            start_time = time.time()
        httclass_pred_image_inds, httclass_pred_class_inds, httclass_pred_scores = self.hn.split_by_htt_class(
            pred_image_inds, pred_class_inds, pred_scores, self.htt_mode, self.atlas)
        if self.verbosity == 'NORMAL':
            print(' (%s seconds)' % (time.time() - start_time))

        # Obtain Grad-CAM
        final_layer = self.hn.find_final_layer()
        gc = GradCAM(params={'htt_mode': self.htt_mode, 'size': self.input_size,
                             'num_imgs': self.input_images_norm.shape[0], 'batch_size': len(self.input_files_batch),
                             'cnn_model': self.hn.model, 'final_layer': final_layer, 'tmp_dir': self.tmp_dir})
        httclass_gradcam_serial = []
        httclass_gradcam_image_wise = []
        httclass_pred_legends = []
        httclass_cs_gradcam_post = []
        self.httclass_pred_segmasks = []
        for iter_httclass in range(len(self.htt_classes)):
            htt_class = self.htt_classes[iter_httclass]
            if self.save_types[0]:
                if htt_class != 'glas' and 'mmmp' not in self.input_name:
                    out_patchconf_dir = os.path.join(self.out_dir, htt_class, 'patchconfidence')
                    mkdir_if_nexist(out_patchconf_dir)
                    save_patchconfidence(httclass_pred_image_inds[iter_httclass], httclass_pred_class_inds[iter_httclass],
                                         httclass_pred_scores[iter_httclass], self.input_size, out_patchconf_dir,
                                         self.input_files_batch, self.httclass_valid_classes[iter_httclass])
                elif htt_class == 'glas':
                    exocrine_class_ind = self.atlas.glas_valid_classes.index('G.O')
                    exocrine_scores = httclass_pred_scores[iter_httclass][httclass_pred_class_inds[iter_httclass] == exocrine_class_ind]
                    if len(exocrine_scores) < self.input_images.shape[0]:
                        raise Exception('Number of detected GlaS exocrine scores ' + str(len(exocrine_scores)) +
                                        ' less than number of crops in image' + str(self.input_images.shape[0]) + '!')
                    self.glas_confscores.append(np.mean(exocrine_scores))
                    a=1
            if self.run_level == 1:
                continue

            # Generate serial Grad-CAM
            if self.verbosity == 'NORMAL':
                print('\t\t\t[' + htt_class + '] Generating Grad-CAM', end='')
                start_time = time.time()
            gradcam_serial = gc.gen_gradcam(httclass_pred_image_inds[iter_httclass],
                                            httclass_pred_class_inds[iter_httclass],
                                            httclass_pred_scores[iter_httclass],
                                            self.input_images_norm, self.atlas,
                                            self.httclass_valid_classes[iter_httclass])
            if self.verbosity == 'NORMAL':
                print(' (%s seconds)' % (time.time() - start_time))
            httclass_gradcam_serial.append(gradcam_serial)

            # Expand Grad-CAM for each image
            if self.verbosity == 'NORMAL':
                print('\t\t\t[' + htt_class + '] Expanding Grad-CAM', end='')
                start_time = time.time()
            gradcam_image_wise = gc.expand_image_wise(gradcam_serial, httclass_pred_image_inds[iter_httclass],
                                                      httclass_pred_class_inds[iter_httclass],
                                                      self.httclass_valid_classes[iter_httclass])
            if self.verbosity == 'NORMAL':
                print(' (%s seconds)' % (time.time() - start_time))
            httclass_gradcam_image_wise.append(gradcam_image_wise)

            # Stitch Grad-CAMs if in glas mode
            if 'glas_full' in self.input_name or 'mmmp' in self.input_name:
                gradcam_image_wise = stitch_patch_activations(gradcam_image_wise, self.down_fac, self.orig_sizes[0])

            # Perform HTT modifications to Grad-CAM
            if self.verbosity == 'NORMAL':
                print('\t\t\t[' + htt_class + '] Modifying Grad-CAM by HTT', end='')
                start_time = time.time()

            getting_figures = True
            patch_to_get = 'PP7_i17953_j28561_f1'
            if getting_figures and patch_to_get + '.png' in self.input_files_batch:
                ind = self.input_files_batch.index(patch_to_get + '.png')
                if htt_class == 'morph':
                    disp = 1
                elif htt_class == 'func':
                    disp = 2
                noadjust_gradcam = gradcam_image_wise[ind][disp:]
                argmax_unadjusted = maxconf_class_as_colour(np.expand_dims(np.argmax(noadjust_gradcam, axis=0) + disp, axis=0),
                                                  self.httclass_valid_colours[iter_httclass],
                                                  self.orig_sizes[0])
                out_path = os.path.join(os.path.abspath(os.path.curdir), 'visuals', 'tmp',
                                        patch_to_get + '_' + htt_class + '_argmax_unadjusted.png')
                cv2.imwrite(out_path, cv2.cvtColor(argmax_unadjusted[0], cv2.COLOR_RGB2BGR))
                a=1
            if htt_class == 'func':
                adipose_inds = [i for i,x in enumerate(self.atlas.morph_valid_classes) if x in ['A.W', 'A.B', 'A.M']]
                gradcam_adipose = httclass_gradcam_image_wise[iter_httclass-1][:, adipose_inds]
                gradcam_image_wise = gc.modify_by_htt(gradcam_image_wise, self.orig_images, self.atlas, htt_class,
                                                      gradcam_adipose=gradcam_adipose)
            else:
                gradcam_image_wise = gc.modify_by_htt(gradcam_image_wise, self.orig_images, self.atlas, htt_class)
            if self.verbosity == 'NORMAL':
                print(' (%s seconds)' % (time.time() - start_time))
            if getting_figures and patch_to_get + '.png' in self.input_files_batch:
                ind = self.input_files_batch.index(patch_to_get + '.png')
                argmax_adjusted = maxconf_class_as_colour(
                    np.expand_dims(np.argmax(gradcam_image_wise[ind], axis=0), axis=0),
                    self.httclass_valid_colours[iter_httclass],
                    self.orig_sizes[0])
                out_path = os.path.join(os.path.abspath(os.path.curdir), 'visuals', 'tmp',
                                        patch_to_get + '_' + htt_class + '_argmax_adjusted.png')
                cv2.imwrite(out_path, cv2.cvtColor(argmax_adjusted[0], cv2.COLOR_RGB2BGR))

                out_path = os.path.join(os.path.abspath(os.path.curdir), 'visuals', 'tmp', 'gray',
                                        patch_to_get + '_' + htt_class + '_background.png')
                cv2.imwrite(out_path, 255 * gradcam_image_wise[ind, 0])

                if htt_class == 'func':
                    out_path = os.path.join(os.path.abspath(os.path.curdir), 'visuals', 'tmp', 'gray',
                                            patch_to_get + '_' + htt_class + '_other.png')
                    cv2.imwrite(out_path, 255 * gradcam_image_wise[ind, 1])
            # Get Class-Specific Grad-CAM
            if self.verbosity == 'NORMAL':
                print('\t\t\t[' + htt_class + '] Getting Class-Specific Grad-CAM', end='')
                start_time = time.time()
            cs_gradcam = gc.get_cs_gradcam(gradcam_image_wise, self.atlas, htt_class)
            if self.save_types[1]:
                out_cs_gradcam_dir = os.path.join(self.out_dir, htt_class, 'gradcam')
                mkdir_if_nexist(out_cs_gradcam_dir)
                save_cs_gradcam(cs_gradcam, out_cs_gradcam_dir, self.input_files_batch,
                                self.httclass_valid_classes[iter_httclass])
            if self.verbosity == 'NORMAL':
                print(' (%s seconds)' % (time.time() - start_time))
            if self.run_level == 2 or 'overlap' in self.input_name:
                continue

            # Get legends
            if self.verbosity == 'NORMAL':
                print('\t\t\t[' + htt_class + '] Getting prediction legends', end='')
                start_time = time.time()
            gradcam_mod_class_inds = cs_gradcam_to_class_inds(cs_gradcam)
            # TOOD: ensure that all of self.orig_sizes are the same
            pred_legends = get_legends(gradcam_mod_class_inds, self.orig_sizes[0],
                                       self.httclass_valid_classes[iter_httclass],
                                       self.httclass_valid_colours[iter_httclass])
            if self.verbosity == 'NORMAL':
                print(' (%s seconds)' % (time.time() - start_time))
            httclass_pred_legends.append(pred_legends)

            # Perform dense CRF post-processing
            dcrf = DenseCRF()
            # if htt_class == 'morph':
            dcrf_config_path = os.path.join(self.data_dir, htt_class + '_optimal_pcc.npy')
            dcrf.load_config(dcrf_config_path)

            if self.verbosity == 'NORMAL':
                print('\t\t\t[' + htt_class + '] Performing post-processing', end='')
                start_time = time.time()
            cs_gradcam_post_maxconf = dcrf.process(cs_gradcam, self.orig_images)
            httclass_cs_gradcam_post.append(cs_gradcam_post_maxconf)
            if self.verbosity == 'NORMAL':
                print(' (%s seconds)' % (time.time() - start_time))

            cs_gradcam_post_discrete = maxconf_class_as_colour(cs_gradcam_post_maxconf,
                                                               self.httclass_valid_colours[iter_httclass],
                                                               self.orig_sizes[0])
            self.httclass_pred_segmasks.append(cs_gradcam_post_discrete)
            if self.save_types[2]:
                out_patch_dir = os.path.join(self.out_dir, htt_class, 'patch')
                mkdir_if_nexist(out_patch_dir)
                save_pred_segmasks(cs_gradcam_post_discrete, out_patch_dir, self.input_files_batch)

            if self.save_types[3]:
                if self.verbosity == 'NORMAL':
                    print('\t\t\t[' + htt_class + '] Exporting segmentation summary images', end='')
                    start_time = time.time()
                cs_gradcam_pre_argmax = np.argmax(cs_gradcam, axis=1)
                cs_gradcam_pre_discrete = maxconf_class_as_colour(cs_gradcam_pre_argmax,
                                                                  self.httclass_valid_colours[iter_httclass],
                                                                  self.orig_sizes[0])
                cs_gradcam_pre_continuous = gradcam_as_continuous(cs_gradcam,
                                                                  self.httclass_valid_colours[iter_httclass],
                                                                  self.orig_sizes[0])
                export_summary_image(self.input_files_batch, self.orig_images, self.out_dir,
                                     self.httclass_gt_legends[iter_httclass], pred_legends,
                                     self.httclass_gt_segmasks[iter_httclass], cs_gradcam_post_discrete,
                                     cs_gradcam_pre_discrete, cs_gradcam_pre_continuous, htt_class)
                if self.verbosity == 'NORMAL':
                    print(' (%s seconds)' % (time.time() - start_time))
            if htt_class == 'glas':
                save_glas_bmps(self.input_files_batch, cs_gradcam_post_maxconf, self.out_dir, htt_class,
                               self.orig_sizes[0])
            a=1
        a=1

    def overlap_and_segment(self):

        def find_patch_htts(file, dir):
            files = [x for x in os.listdir(dir) if file in x]
            return [x.split('_h')[-1].split('.png')[0] for x in files]

        def rotate(l, n):
            return l[n:] + l[:n]

        def read_gradcam(file):
            return cv2.imread(file, cv2.IMREAD_GRAYSCALE).astype('float64') / 255

        self.orig_patch_size = [1088, 1088]
        self.overlap_ratio = 0.25
        sz = self.input_size

        shift = [int((1-self.overlap_ratio) * self.orig_patch_size[i]) for i in range(2)]
        ov = [int(self.overlap_ratio * sz[i]) for i in range(2)]

        for iter_httclass in range(len(self.htt_classes)):
            htt_class = self.htt_classes[iter_httclass]
            gradcam_dir = os.path.join(self.out_dir, htt_class, 'gradcam')
            if self.save_types[1]:
                overlap_gradcam_dir = os.path.join(self.out_dir, htt_class, 'gradcam_overlap')
                mkdir_if_nexist(overlap_gradcam_dir)

            dcrf = DenseCRF()
            dcrf_config_path = os.path.join(self.data_dir, htt_class + '_optimal_pcc.npy')
            dcrf.load_config(dcrf_config_path)
            # Overlap: 123_R01_i10609_j4081_f1.png
            for iter_file, input_file in enumerate(self.input_files_all):
                if iter_file % 100 == 0:
                    print('Overlapping file #%d of %d' % (iter_file+1, len(self.input_files_all)))
                cur_patch_path = os.path.join(self.img_dir, self.input_name, input_file)
                cur_patch_img = read_image(cur_patch_path)

                patch_name = os.path.splitext(input_file)[0]
                pyramid_id = patch_name.split('_i')[0]
                overlap_gradcam_imagewise = np.zeros((len(self.httclass_valid_classes[iter_httclass]), sz[0], sz[1]))

                # Get location of top-left pixel
                cur_i = int(patch_name.split('_i')[-1].split('_')[0])
                cur_j = int(patch_name.split('_j')[-1].split('_')[0])

                # Get neighbour patch locations
                neigh_i_list = cur_i + np.array([-shift[0], -shift[0], 0, shift[0], shift[0], shift[0], 0, -shift[0]])
                neigh_j_list = cur_j + np.array([0, shift[1], shift[1], shift[1], 0, -shift[1], -shift[1], -shift[1]])
                neighbour_patch_names = [pyramid_id + '_i' + str(neigh_i_list[i]) + '_j' + str(neigh_j_list[i]) + '_f1'
                                         for i in range(8)]
                neigh_start_i = [sz[0] - ov[0], sz[0] - ov[0],
                                 0, 0, 0, 0, 0,
                                 sz[0] - ov[0]]
                neigh_end_i = [sz[0], sz[0], sz[0],
                               ov[0], ov[0], ov[0],
                               sz[0], sz[0]]
                neigh_start_j = [0, 0, 0, 0, 0,
                                 sz[1] - ov[1], sz[1] - ov[1], sz[1] - ov[1]]
                neigh_end_j = [sz[1],
                               ov[1], ov[1], ov[1],
                               sz[1], sz[1], sz[1], sz[1]]
                
                cur_start_i = rotate(neigh_start_i, 4)
                cur_end_i = rotate(neigh_end_i, 4)
                cur_start_j = rotate(neigh_start_j, 4)
                cur_end_j = rotate(neigh_end_j, 4)

                # Get union of current and neighbour patch HTTs
                cur_patch_htts = find_patch_htts(patch_name, gradcam_dir)
                union_htts = cur_patch_htts[:]
                for neighbour_patch_name in neighbour_patch_names:
                    neighbour_patch_htts = find_patch_htts(neighbour_patch_name, gradcam_dir)
                    union_htts += neighbour_patch_htts
                union_htts = list(set(union_htts))
                
                # Go through each class
                for htt in union_htts:
                    cur_htt_gradcam_path = os.path.join(gradcam_dir, patch_name + '_h' + htt + '.png')
                    # - Initialize overlapped Grad-CAM patch with current patch's Grad-CAM
                    if htt in cur_patch_htts:
                        overlap_gradcam = read_gradcam(cur_htt_gradcam_path)
                    # - Create new overlapped Grad-CAM patch if not already detected
                    else:
                        overlap_gradcam = np.zeros((self.input_size[0], self.input_size[1]))
                    # - Create counter patch
                    counter_patch = np.ones((self.input_size[0], self.input_size[1]))
                    # Go through each neighbour
                    for iter_neigh, neighbour_patch_name in enumerate(neighbour_patch_names):
                        neigh_htt_gradcam_path = os.path.join(gradcam_dir, neighbour_patch_name + '_h' + htt + '.png')
                        if os.path.exists(neigh_htt_gradcam_path):
                            neigh_htt_gradcam = read_gradcam(neigh_htt_gradcam_path)
                            overlap_gradcam[cur_start_i[iter_neigh]:cur_end_i[iter_neigh],
                                            cur_start_j[iter_neigh]:cur_end_j[iter_neigh]] += \
                                neigh_htt_gradcam[neigh_start_i[iter_neigh]:neigh_end_i[iter_neigh],
                                                  neigh_start_j[iter_neigh]:neigh_end_j[iter_neigh]]
                            counter_patch[cur_start_i[iter_neigh]:cur_end_i[iter_neigh],
                                            cur_start_j[iter_neigh]:cur_end_j[iter_neigh]] += 1
                    # Divide overlap Grad-CAM by counter patch
                    overlap_gradcam /= counter_patch
                    overlap_gradcam_imagewise[self.httclass_valid_classes[iter_httclass].index(htt)] = overlap_gradcam

                    if self.save_types[1]:
                        overlap_gradcam_path = os.path.join(overlap_gradcam_dir, patch_name + '_h' + htt + '.png')
                        cv2.imwrite(overlap_gradcam_path, 255 * overlap_gradcam)
                    a=1
                if self.run_level == 2:
                    continue

                ##
                # Perform dense CRF post-processing
                overlap_gradcam_post_maxconf = dcrf.process(np.expand_dims(overlap_gradcam_imagewise, axis=0),
                                                            np.expand_dims(cur_patch_img, axis=0))

                cs_gradcam_post_discrete = maxconf_class_as_colour(overlap_gradcam_post_maxconf,
                                                                   self.httclass_valid_colours[iter_httclass], sz)
                if self.save_types[2]:
                    out_patch_dir = os.path.join(self.out_dir, htt_class, 'patch')
                    mkdir_if_nexist(out_patch_dir)
                    save_pred_segmasks(cs_gradcam_post_discrete, out_patch_dir, [input_file])
                a=1
                # cs_gradcam

    def eval_segmentation(self):
        items = []
        for iter_httclass in range(len(self.httclass_gt_segmasks)):
            colours = self.httclass_valid_colours[iter_httclass]
            loginvfreq = self.httclass_loginvfreq[iter_httclass]
            intersect_count = self.httclass_intersect_counts[iter_httclass]
            union_count = self.httclass_union_counts[iter_httclass]
            # Find the GT, intersection, union counts for each HTT
            for iter_class in range(colours.shape[0]):
                pred_segmask_cur = np.all(self.httclass_pred_segmasks[iter_httclass] == colours[iter_class], axis=-1)
                gt_segmask_cur = np.all(self.httclass_gt_segmasks[iter_httclass] == colours[iter_class], axis=-1)
                intersect_count[iter_class] +=   np.sum(np.bitwise_and(pred_segmask_cur, gt_segmask_cur))
                union_count[iter_class] += np.sum(np.bitwise_or(pred_segmask_cur, gt_segmask_cur))
            # Find fIoU and mIoU
            httIoU = intersect_count / (union_count + 1e-12)
            iou_items = []
            for iter_class, valid_class in enumerate(self.httclass_valid_classes[iter_httclass]):
                iou_items.append((valid_class, [httIoU[iter_class]]))
            IoU_metric_path = os.path.join(self.out_dir, self.htt_classes[iter_httclass] + '_IoU_metric_results.csv')
            res = pd.DataFrame.from_dict(dict(iou_items))
            res.to_csv(IoU_metric_path)

            fIoU = np.sum(loginvfreq * httIoU)
            mIoU = np.average(httIoU)
            fIoU_name = self.htt_classes[iter_httclass] + '_fIoU'
            mIoU_name = self.htt_classes[iter_httclass] + '_mIoU'
            items.append((fIoU_name, [fIoU]))
            items.append((mIoU_name, [mIoU]))

        # Export results
        metric_path = os.path.join(self.out_dir, 'metric_results.csv')
        res = pd.DataFrame.from_dict(dict(items))
        res.to_csv(metric_path)