import numpy as np
import cv2
from keras.models import model_from_json
from keras import optimizers
import keras
import numpy as np
import os
import scipy
from scipy import io
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from skimage import measure, filters
import math

def mkdir_if_nexist(pth):
    if not os.path.exists(pth):
        os.makedirs(pth)

# def load_if_nexist(pth):

def read_image(path):
    # Read image in BGR format
    x = cv2.cvtColor(cv2.imread(path), cv2.COLOR_RGB2BGR)
    img = keras.preprocessing.image.load_img(path, target_size=(x.shape[0], x.shape[1]))
    y = keras.preprocessing.image.img_to_array(img)
    return y

def crop_into_patches(image, down_fac, out_size):
    orig_size = image.shape[:2]
    downsampled_size = [round(x / down_fac) for x in orig_size]
    # If downsampled image is smaller than the patch size, then mirror pad first, then downsample
    if downsampled_size[0] < out_size[0] or downsampled_size[1] < out_size[1]:
        pad_vert = math.ceil(max(out_size[0] * down_fac - orig_size[0], 0) / 2)
        pad_horz = math.ceil(max(out_size[1] * down_fac - orig_size[1], 0) / 2)
        image = cv2.copyMakeBorder(image, pad_vert, pad_vert, pad_horz, pad_horz, cv2.BORDER_REFLECT)
        downsampled_size = [round(x / down_fac) for x in image.shape[:2]]
    if downsampled_size != orig_size:
        downsampled_image = cv2.resize(image, dsize=(downsampled_size[1], downsampled_size[0]),
                                       interpolation=cv2.INTER_LINEAR)
    else:
        downsampled_image = image

    num_crops = [math.ceil(downsampled_size[i] / out_size[i]) for i in range(2)]
    # crop_offset = [downsampled_size[i] - out_size[i] for i in range(2)]
    crop_offset = []
    if num_crops[0] > 1:
        crop_offset.append(int(np.floor((num_crops[0] * out_size[0] - downsampled_size[0]) / (num_crops[0] - 1))))
    else:
        crop_offset.append(0)
    if num_crops[1] > 1:
        crop_offset.append(int(np.floor((num_crops[1] * out_size[1] - downsampled_size[1]) / (num_crops[1] - 1))))
    else:
        crop_offset.append(0)
    # crop_offset = [int(np.floor((num_crops[i] * out_size[i] - downsampled_size[i]) / (num_crops[i] - 1))) for i in range(2)]

    total_crops = np.prod(np.array(num_crops))
    patches = np.zeros((total_crops, out_size[0], out_size[1], 3))
    iter_patch = 0
    for iter_row in range(num_crops[0]):
        if iter_row < num_crops[0] - 1:
            start_i = iter_row * (out_size[0] - crop_offset[0])
        else:
            start_i = downsampled_size[0] - out_size[0]
        end_i = start_i + out_size[0]
        for iter_col in range(num_crops[1]):
            if iter_col < num_crops[1] - 1:
                start_j = iter_col * (out_size[1] - crop_offset[1])
            else:
                start_j = downsampled_size[1] - out_size[1]
            end_j = start_j + out_size[1]
            patches[iter_patch] = downsampled_image[start_i:end_i, start_j:end_j, :]
            iter_patch += 1
    return patches, image

def stitch_patch_activations(patch_activations, down_fac, out_size):
    num_classes = patch_activations.shape[1]
    input_size = patch_activations.shape[2:]
    upsampled_size = [round(x * down_fac) for x in input_size]

    # If upsampled image is larger than the original size, then remove padding
    pad_vert = math.ceil(max(input_size[0] * down_fac - out_size[0], 0) / 2)
    pad_horz = math.ceil(max(input_size[1] * down_fac - out_size[1], 0) / 2)
    out_size_padded = [out_size[0] + 2 * pad_vert, out_size[1] + 2 * pad_horz]

    num_crops = [math.ceil(out_size_padded[i] / upsampled_size[i]) for i in range(2)]
    # crop_offset = [out_size_padded[i] - upsampled_size[i] for i in range(2)]
    crop_offset = [int(np.floor((num_crops[i] * upsampled_size[i] - out_size_padded[i]) / (num_crops[i] - 1))) for i in range(2)]

    iter_patch = 0
    G = np.zeros((1, num_classes, out_size_padded[0], out_size_padded[1]))
    H = np.zeros((1, 1, out_size_padded[0], out_size_padded[1]))
    for iter_row in range(num_crops[0]):
        if iter_row < num_crops[0] - 1:
            start_i = iter_row * (upsampled_size[0] - crop_offset[0])
        else:
            start_i = out_size_padded[0] - upsampled_size[0]
        end_i = start_i + upsampled_size[0]
        for iter_col in range(num_crops[1]):
            if iter_col < num_crops[1] - 1:
                start_j = iter_col * (upsampled_size[1] - crop_offset[1])
            else:
                start_j = out_size_padded[1] - upsampled_size[1]
            end_j = start_j + upsampled_size[1]
            upsampled_activation = np.zeros((num_classes, upsampled_size[0], upsampled_size[1]))
            for iter_class in range(num_classes):
                upsampled_activation[iter_class] = cv2.resize(patch_activations[iter_patch, iter_class],
                                                              dsize=(upsampled_size[1], upsampled_size[0]),
                                                              interpolation=cv2.INTER_LINEAR)
            G[0, :, start_i:end_i, start_j:end_j] += upsampled_activation
            H[0, 0, start_i:end_i, start_j:end_j] += 1
            iter_patch += 1
    G /= H

    # Remove padding
    G = G[0, :, pad_vert:G.shape[2]-pad_vert, pad_horz:G.shape[3]-pad_horz]
    G = np.expand_dims(G, axis=0)
    return G

def read_segmask(path, size=[224, 224]):
    # Read segmask image
    x = cv2.cvtColor(cv2.imread(path), cv2.COLOR_RGB2BGR)
    if x.shape[0] != size[0] or x.shape[1] != size[1]:
        return cv2.resize(x, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    else:
        return x

def mult_overlay_on_img(X, I, ratio=[0.5, 0.5]):
    Y = np.zeros_like(X, dtype='uint8')
    for iter_image in range(I.shape[0]):
        T = ratio[0] * np.float32(X[iter_image]) + ratio[1] * np.float32(I[iter_image])
        Y[iter_image] = np.uint8(255 * T / np.max(T))
    return Y

def segmask_to_class_inds(segmask, colours):
    class_inds = []
    for iter_image in range(len(segmask)):
        cur_class_inds = [i for i in range(len(colours)) if np.any(np.all(segmask[iter_image] == colours[i], axis=2))]
        class_inds.append(cur_class_inds)
    return class_inds

def get_legends(class_inds, size, classes, colours):
    font_scale = 0.3
    scale_fac = size[1] / 224
    rect_offset_x = 100

    legends = np.zeros((len(class_inds), size[0], size[1], 3), dtype="uint8")
    for iter_image in range(len(class_inds)):
        cur_class_inds = class_inds[iter_image]
        cur_colours = [x for i, x in enumerate(colours) if i in cur_class_inds]
        row_height = size[0] // len(class_inds[iter_image])
        for iter_class_ind, class_ind in enumerate(cur_class_inds):
            class_string = classes[class_ind] + ' (' + str(class_ind) + ')'
            cv2.putText(legends[iter_image], class_string, (5, (iter_class_ind * row_height) + round(row_height * .75)),
                        cv2.FONT_HERSHEY_DUPLEX, font_scale * scale_fac, (255, 0, 0), 1)
            if iter_class_ind == len(cur_class_inds) - 1:
                cur_row_height = row_height + size[0] - (iter_class_ind + 1) * row_height
            else:
                cur_row_height = row_height
            colour = cur_colours[iter_class_ind]
            cv2.rectangle(legends[iter_image], (round(rect_offset_x * scale_fac), (iter_class_ind * row_height)),
                                                (size[1], (iter_class_ind * row_height) + cur_row_height),
                          thickness=-1, color=(int(colour[0]), int(colour[1]), int(colour[2])))
    return legends

def class_inds_serial_to_image_wise(class_inds, image_inds):
    unique_image_inds = list(set(image_inds))
    class_inds_image_wise = []
    for unique_image_ind in unique_image_inds:
        cur_image_class_inds = [x for i, x in enumerate(class_inds) if image_inds[i] == unique_image_ind]
        class_inds_image_wise.append(cur_image_class_inds)
    return class_inds_image_wise

def cs_gradcam_to_class_inds(cs_gradcam):
    is_class_present = np.any(np.any(cs_gradcam, axis=-2), axis=-1)
    class_inds = []
    for iter_input_image in range(is_class_present.shape[0]):
        class_inds.append(list(np.where(is_class_present[iter_input_image])[0]))
    return class_inds

def maxconf_class_as_colour(maxconf_crf, colours, size):
    num_input_images = maxconf_crf.shape[0]
    Y = np.zeros((num_input_images, size[0], size[1], 3), dtype='uint8')
    for iter_input_image in range(num_input_images):
        for iter_class in range(colours.shape[0]):
            Y[iter_input_image, maxconf_crf[iter_input_image] == iter_class] = np.array(colours[iter_class])
    return Y

def gradcam_as_continuous(gradcam, colours, size):
    num_input_images = gradcam.shape[0]
    maxconf_gradcam = np.argmax(gradcam, axis=1)
    Y = np.zeros((num_input_images, size[0], size[1], 3), dtype='uint8')
    for iter_input_image in range(num_input_images):
        for iter_class in range(colours.shape[0]):
            class_mask = np.array(np.ma.array(gradcam[iter_input_image, iter_class],
                                              mask=maxconf_gradcam[iter_input_image] != iter_class))
            Y[iter_input_image] += np.uint8(np.array(colours[iter_class]) * class_mask[:, :, None])
    return Y

def add_sidelabels(img, leftlabels, toplabels, addwidth, addheight, size):
    leftpanel = 255 * np.ones((img.shape[0], addwidth, 3), dtype='float32')
    toppanel = 255 * np.ones((addheight, img.shape[1], 3), dtype='float32')
    cornerpanel = 255 * np.ones((addheight, addwidth, 3), dtype='float32')

    font_scale = 0.3
    scale_fac = size[1] / 224

    for iter_leftlabel, leftlabel in enumerate(leftlabels):
        cv2.putText(leftpanel, leftlabel, (10, (iter_leftlabel * size[0]) + size[0] // 2),
                    cv2.FONT_HERSHEY_TRIPLEX, font_scale * scale_fac, (0, 0, 0), 1)
    for iter_toplabel, toplabel in enumerate(toplabels):
        cv2.putText(toppanel, toplabel, ((iter_toplabel * size[1]) + 10, addheight - 10),
                    cv2.FONT_HERSHEY_TRIPLEX, font_scale * scale_fac, (0, 0, 0), 1)
    img2 = np.concatenate((toppanel, img), axis=0)
    img3 = np.concatenate((np.concatenate((cornerpanel, leftpanel), axis=0), img2), axis=1)
    return img3

def concat_to_grid(filename, X1, X2, X3, X4, X5, X6, X7, out_dir, layout, htt_class):
    type_labels = ['Morphological HTT', 'Functional HTT']
    panel_labels = ['Original Image', 'Ground Truth Legend', 'Ground Truth Segmentation', 'Predicted Legend',
                         'Predicted Segmentation, Processed', 'Predicted Segmentation, Unprocessed',
                         'Predicted CSGC']
    size = X1.shape[:2]
    text_height = round(20 * size[0] / 224)
    text_width = round(200 * size[1] / 224)
    if layout == 'horizontal':
        left_labels = type_labels
        top_labels = panel_labels
        axis = 1
    elif layout == 'vertical':
        left_labels = panel_labels
        top_labels = type_labels
        axis = 0

    X = np.float32(np.concatenate((X1, X2, X3, X4, X5, X6, X7), axis=axis))
    X = add_sidelabels(X, left_labels, top_labels, text_width, text_height, size)
    cv2.imwrite(os.path.join(out_dir, htt_class, layout, filename + '.png'), cv2.cvtColor(X, cv2.COLOR_RGB2BGR))

def export_summary_image(input_files, input_images, out_dir, gt_legends, pred_legends, gt_segmask,
                         cs_gradcam_post_discrete, cs_gradcam_pre_discrete, cs_gradcam_pre_continuous, htt_class,
                         layouts=['horizontal', 'vertical']):
    gt_segmask_discrete_overlaid = mult_overlay_on_img(gt_segmask, input_images)
    cs_gradcam_post_discrete_overlaid = mult_overlay_on_img(cs_gradcam_post_discrete, input_images)
    cs_gradcam_pre_discrete_overlaid = mult_overlay_on_img(cs_gradcam_pre_discrete, input_images)
    cs_gradcam_pre_continuous_overlaid = mult_overlay_on_img(cs_gradcam_pre_continuous, input_images,
                                                             ratio=[0.75, 0.25])

    for layout in layouts:
        mkdir_if_nexist(os.path.join(out_dir, htt_class, layout))
        for iter_input_image in range(input_images.shape[0]):
            concat_to_grid(input_files[iter_input_image], input_images[iter_input_image], gt_legends[iter_input_image],
                           gt_segmask_discrete_overlaid[iter_input_image], pred_legends[iter_input_image],
                           cs_gradcam_post_discrete_overlaid[iter_input_image],
                           cs_gradcam_pre_discrete_overlaid[iter_input_image],
                           cs_gradcam_pre_continuous_overlaid[iter_input_image], out_dir, layout, htt_class)

def save_glas_bmps(input_files, pred, out_dir, htt_class, full_size):
    single_gland_out_dir = os.path.join(out_dir, htt_class, 'single_gland')
    mkdir_if_nexist(single_gland_out_dir)

    multi_gland_out_dir = os.path.join(out_dir, htt_class, 'multi_gland')
    mkdir_if_nexist(multi_gland_out_dir)

    for iter_input_image in range(len(input_files)):
        P = cv2.resize(pred[iter_input_image], (full_size[1], full_size[0]), interpolation=cv2.INTER_NEAREST)
        # Option 1: consider all predicted blobs as one gland object
        out_filename = input_files[iter_input_image].replace('.png', '.bmp')
        out_path = os.path.join(single_gland_out_dir, out_filename)
        cv2.imwrite(out_path, P)

        # Option 2: consider each predicted blob as an individual gland object
        im = filters.gaussian(P, sigma=.5)
        im = im > np.average(im)
        lbl = measure.label(im)
        out_path = os.path.join(multi_gland_out_dir, out_filename)
        cv2.imwrite(out_path, lbl)

def save_patchconfidence(image_inds, class_inds, scores, size, out_dir, out_names, classes):
    if len(image_inds) != len(class_inds) or len(image_inds) != len(scores):
        raise Exception('Number of images, classes, and scores must be equal!')
    for iter_patchconfidence in range(len(image_inds)):
        out_name = out_names[image_inds[iter_patchconfidence]]
        Y = 255 * scores[iter_patchconfidence] * np.ones((size[0], size[1], 3))
        patch_conf_name = os.path.splitext(out_name)[0] + '_h' + classes[class_inds[iter_patchconfidence]] + '.png'
        patch_conf_path = os.path.join(out_dir, patch_conf_name)
        cv2.imwrite(patch_conf_path, Y)
    a=1

def save_pred_segmasks(X, out_dir, out_names):
    if X.shape[0] != len(out_names):
        raise Exception('Number of files in segmasks must equal number of file names!')
    for iter_image in range(X.shape[0]):
        out_path = os.path.join(out_dir, out_names[iter_image])
        cv2.imwrite(out_path, cv2.cvtColor(X[iter_image], cv2.COLOR_RGB2BGR))
        a=1

def save_cs_gradcam(X, out_dir, out_names, classes):
    if X.shape[0] != len(out_names):
        raise Exception('Number of files in CS-Grad-CAMs must equal number of file names!')
    if X.shape[1] != len(classes):
        raise Exception('Number of classes in CS-Grad-CAMs must equal number of valid classes')
    for iter_image in range(X.shape[0]):
        for iter_class in range(X.shape[1]):
            if np.sum(X[iter_image, iter_class]) > 0:
                out_path = os.path.join(out_dir, os.path.splitext(out_names[iter_image])[0] + '_h' + classes[iter_class] + \
                           os.path.splitext(out_names[iter_image])[1])
                cv2.imwrite(out_path, 255 * X[iter_image, iter_class])