import os
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

class DenseCRF:
    def __init__(self):
        # self.gauss_sxy = 3
        # self.gauss_compat = 30
        # self.bilat_sxy = 10
        # self.bilat_srgb = 20
        # self.bilat_compat = 50
        # self.n_infer = 5

        self.gauss_sxy = 3
        self.gauss_compat = 30
        self.bilat_sxy = 10
        self.bilat_srgb = 20
        self.bilat_compat = 50
        self.n_infer = 5

    def load_config(self, path):
        if os.path.exists(path):
            config = np.load(path)
            self.gauss_sxy, self.gauss_compat, self.bilat_sxy, self.bilat_srgb, self.bilat_compat, self.n_config = \
            config[0]
        else:
            print('Warning: dense CRF config file ' + path + ' does not exist - using defaults')

    def process(self, probs, images):
        # Set up variable sizes
        num_input_images = probs.shape[0]
        num_classes = probs.shape[1]
        size = images.shape[1:3]
        crf = np.zeros((num_input_images, num_classes, size[0], size[1]))
        for iter_input_image in range(num_input_images):
            pass_class_inds = np.where(np.sum(np.sum(probs[iter_input_image], axis=1), axis=1) > 0)
            # Set up dense CRF 2D
            d = dcrf.DenseCRF2D(size[1], size[0], len(pass_class_inds[0]))
            if len(pass_class_inds[0]) > 0:
                cur_probs = probs[iter_input_image, pass_class_inds[0]]
                # Unary energy
                U = np.ascontiguousarray(unary_from_softmax(cur_probs))
                d.setUnaryEnergy(U)
                # Penalize small, isolated segments
                # (sxy are PosXStd, PosYStd)
                d.addPairwiseGaussian(sxy=self.gauss_sxy, compat=self.gauss_compat)
                # Incorporate local colour-dependent features
                # (sxy are Bi_X_Std and Bi_Y_Std,
                #  srgb are Bi_R_Std, Bi_G_Std, Bi_B_Std)
                d.addPairwiseBilateral(sxy=self.bilat_sxy, srgb=self.bilat_srgb, rgbim=np.uint8(images[iter_input_image]),
                                       compat=self.bilat_compat)
                # Do inference
                Q = d.inference(self.n_infer)
                crf[iter_input_image, pass_class_inds] = np.array(Q).reshape((len(pass_class_inds[0]), size[0], size[1]))
        maxconf_crf = np.argmax(crf, axis=1)
        return maxconf_crf