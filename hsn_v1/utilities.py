import numpy as np
import cv2
import keras
import os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from skimage import measure, filters
import math

def mkdir_if_nexist(pth):
    """Create a directory if the path does not already exist

    Parameters
    ----------
    pth : str
        Directory path to be created if it does not already exist
    """

    if not os.path.exists(pth):
        os.makedirs(pth)

def read_image(path):
    """Read single image from path

    Parameters
    ----------
    path : str
        Filepath to image to be read
    """

    # Read image in BGR format
    x = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    img = keras.preprocessing.image.load_img(path, target_size=(x.shape[0], x.shape[1]))
    y = keras.preprocessing.image.img_to_array(img)
    return y

def crop_into_patches(image, down_fac, out_size):
    """Crop input image into patches compatible with the classification CNN field of view

    Parameters
    ----------
    image : numpy 3D array (size: H x W x 3)
        The original input image
    down_fac : float
        The downsampling factor
    out_size : list (size: 2)
        The height and width of the patches to be cropped

    Returns
    -------
    patches : numpy 4D array (size: N x H x W x 3), where N = number of crops
        The extracted patches
    image : numpy 3D array (size: H x W x 3)
        The padded input image
    """

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

    # Determine the number of crops and the associated offsets
    num_crops = [math.ceil(downsampled_size[i] / out_size[i]) for i in range(2)]
    crop_offset = []
    for i in range(2):
        if num_crops[i] > 1:
            crop_offset.append(int(np.floor((num_crops[i] * out_size[i] - downsampled_size[i]) / (num_crops[i] - 1))))
        else:
            crop_offset.append(0)

    # Extract patches from the original image
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
    """Stitch the patch activations together for the final segmentation map

    Parameters
    ----------
    patch_activations : numpy 4D array (size: N x C x W x H), where N = number of crops, C = number of classes
        The patch activations
    down_fac : float
        The downsampling factor
    out_size : list (size: 2)
        The height and width of the original image (before cropping patches)

    Returns
    -------
    G : numpy 4D array (size: B x C x W x H), where B = batch size, C = number of classes
        The stitched patch activation (from patches), for a single image
    """

    num_classes = patch_activations.shape[1]
    input_size = patch_activations.shape[2:]
    upsampled_size = [round(x * down_fac) for x in input_size]

    # If upsampled image is larger than the original size, then remove padding
    pad_vert = math.ceil(max(input_size[0] * down_fac - out_size[0], 0) / 2)
    pad_horz = math.ceil(max(input_size[1] * down_fac - out_size[1], 0) / 2)
    out_size_padded = [out_size[0] + 2 * pad_vert, out_size[1] + 2 * pad_horz]

    # Determine the number of crops and the associated offsets
    num_crops = [math.ceil(out_size_padded[i] / upsampled_size[i]) for i in range(2)]
    crop_offset = []
    for i in range(2):
        if num_crops[i] > 1:
            crop_offset.append(int(np.floor((num_crops[i] * upsampled_size[i] - out_size_padded[i]) / (num_crops[i] - 1))))
        else:
            crop_offset.append(0)

    # Resize the patch activations and add to original image size, then divide by overlap count array
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
    """Read segmentation mask image from file

    Parameters
    ----------
    path : str
        Filepath to the segmentation mask image
    size : list (size: 2), optional
        Size of the segmentation mask image, for resizing

    Returns
    -------
    x : numpy 3D array (size: W x H x 3)
        The extracted segmentation mask image
    """

    # Read segmask image; resize if necessary
    x = cv2.cvtColor(cv2.imread(path), cv2.COLOR_RGB2BGR)
    if x.shape[0] != size[0] or x.shape[1] != size[1]:
        return cv2.resize(x, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    else:
        return x

def mult_overlay_on_img(X, I, ratio=[0.5, 0.5]):
    """Overlay multiple segmentation masks on top of images

    Parameters
    ----------
    X : numpy 4D array (size: B x W x H x 3)
        Segmentation masks
    I : numpy 4D array (size: B x W x H x 3)
        Original images
    ratio : list (size: 2), optional
        The scalar values to weight the segmentation mask and image intensities respectively in the final overlay

    Returns
    -------
    Y : numpy 4D array (size: B x W x H x 3)
        The overlaid segmentation masks and images
    """

    Y = np.zeros_like(X, dtype='uint8')
    for iter_image in range(I.shape[0]):
        T = ratio[0] * np.float32(X[iter_image]) + ratio[1] * np.float32(I[iter_image])
        Y[iter_image] = np.uint8(255 * T / np.max(T))
    return Y

def segmask_to_class_inds(segmask, colours):
    """Convert segmentation mask images into list of list of class indices present in each

    Parameters
    ----------
    segmask : numpy 4D array (size: B x W x H x 3), where B = batch size
        Segmentation mask images
    colours : numpy 2D array (size: N x 3), where N = number of colours
        Valid colours used in the segmentation mask images

    Returns
    -------
    class_inds : list (size: B) of list
        List of list of class indices present in segmentation mask images
    """

    class_inds = []
    for iter_image in range(len(segmask)):
        cur_class_inds = [i for i in range(len(colours)) if np.any(np.all(segmask[iter_image] == colours[i], axis=2))]
        class_inds.append(cur_class_inds)
    return class_inds

def get_legends(class_inds, size, classes, colours):
    """Get legends for displaying summary images

    Parameters
    ----------
    class_inds : list (size: B) of list, where B = batch size
        List of list of class indices present in segmentation mask images
    size : list (size: 2)
        Size of the image
    classes : list (size: B) of list, where B = batch size
        List of list of class names present in segmentation mask images
    colours : numpy 2D array (size: N x 3), where N = number of colours
        Valid colours used in the segmentation mask images

    Returns
    -------
    legends : numpy 4D array (size: B x W x H x 3), where B = batch size
        Legend images for the current batch
    """

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

def cs_gradcam_to_class_inds(cs_gradcam):
    """Convert class-specific Grad-CAMs to class indices

    Parameters
    ----------
    cs_gradcam : numpy 4D array (size: B x C x W x H), where B = batch size, C = number of classes
        The class-specific Grad-CAM

    Returns
    -------
    class_inds : list (size: B) of list, where B = batch size
        List of list of class names present in class-specific Grad-CAM
    """

    is_class_present = np.any(np.any(cs_gradcam, axis=-2), axis=-1)
    class_inds = []
    for iter_input_image in range(is_class_present.shape[0]):
        class_inds.append(list(np.where(is_class_present[iter_input_image])[0]))
    return class_inds

def maxconf_class_as_colour(maxconf_crf, colours, size):
    """Convert 3D discrete segmentation masks (indices) into 4D colour images, based on segmentation colour code

    Parameters
    ----------
    maxconf_crf : numpy 3D array (size: B x W x H), where B = batch size
        The maximum-confidence index array
    colours : numpy 2D array (size: N x 3), where N = number of colours
        Valid colours used in the segmentation mask images
    size : list (size: 2)
        Size of the image

    Returns
    -------
    Y : numpy 4D array (size: B x W x H x 3), where B = batch size
        The 4D outputted discrete segmentation mask image
    """

    num_input_images = maxconf_crf.shape[0]
    Y = np.zeros((num_input_images, size[0], size[1], 3), dtype='uint8')
    for iter_input_image in range(num_input_images):
        for iter_class in range(colours.shape[0]):
            Y[iter_input_image, maxconf_crf[iter_input_image] == iter_class] = np.array(colours[iter_class])
    return Y

def gradcam_as_continuous(gradcam, colours, size):
    """Convert 4D continuous Grad-CAM into 3D continuous Grad-CAM (continuous-valued max-confidence map)

    Parameters
    ----------
    gradcam : numpy 4D array (size: B x C x W x H), where B = batch size, C = number of classes
        The 4D continuous Grad-CAM
    colours : numpy 2D array (size: N x 3), where N = number of colours
        Valid colours used in the segmentation mask images
    size : list (size: 2)
        Size of the image

    Returns
    -------
    Y : numpy 4D array (size: B x W x H x 3), where B = batch size
        The 4D outputted continuous Grad-CAM
    """
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
    """Add side labels to the legend image

    Parameters
    ----------
    img : numpy 3D array (size: LH x LW x 3), where LH = legend height, LW = legend width
        The inputted legend image
    leftlabels : list of str
        The labels to be added to the left of the legend image
    toplabels : list of str
        The labels to be added to the top of the legend image
    addwidth : int
        The number of pixels to be added to the left of the legend image
    addheight : int
        The number of pixels to be added to the top of the legend image
    size : list (size: 2)
        Size of a single image (not the legend)

    Returns
    -------
    img3 : numpy 3D array (size: LH+addheight x LW+addwidth x 3), where LH = legend height, LW = legend width
        The legend image with the top and left labels added
    """

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
    """Concatenate summary images in a grid arrangement

    Parameters
    ----------
    filename : str
        The filename of the concatenated summary image
    X1, X2, X3, X4, X5, X6, X7 : numpy 3D arrays (size: H x W x 3)
        The summary images to be concatenated together
    out_dir : str
        The directory to save the concatenated summary image to
    layout : str
        The direction of concatenation, either 'horizontal' or 'vertical'
    htt_class : str
        The type of segmentation set to solve
    """

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
    """Export summary image of segmentation for debugging purposes

    Parameters
    ----------
    input_files : list of str
        List of input image filenames
    input_images : numpy 4D array (size: B x H x W x 3), where B = batch size
        The unnormalized input images in the current batch
    out_dir : str
        The directory to save the summary image to
    gt_legends : numpy 4D array (size: B x H x W x 3), where B = batch size
        The ground-truth annotation legend images in the current batch
    pred_legends : numpy 4D array (size: B x H x W x 3), where B = batch size
        The predicted segmentation legend images in the current batch
    gt_segmask : numpy 4D array (size: B x H x W x 3), where B = batch size
        The ground-truth annotation images in the current batch
    cs_gradcam_post_discrete : numpy 4D array (size: B x H x W x 3), where B = batch size
        The discrete predicted segmentation image, after dense CRF, in the current batch
    cs_gradcam_pre_discrete : numpy 4D array (size: B x H x W x 3), where B = batch size
        The discrete predicted segmentation image, before dense CRF, in the current batch
    cs_gradcam_pre_continuous : numpy 4D array (size: B x H x W x 3), where B = batch size
        The continuous predicted segmentation image, before dense CRF, in the current batch
    htt_class : str
        The type of segmentation set to solve
    layouts : list of str, optional
        The directions of concatenation, elements must be either 'horizontal' or 'vertical'
    """

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
    """Save GlaS segmentations as images

    Parameters
    ----------
    input_files : list of str
        List of input image filenames
    pred : numpy 3D array (size: B x H x W), B = batch size
        The predicted binary gland segmentation masks for the current batch
    out_dir : str
        The directory to save the GlaS segmentation images to
    htt_class : str
        The type of segmentation set to solve
    full_size : tuple of int
        Original size of the GlaS input image
    """

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
    """Save patch confidence scores as single-valued patch images

    Parameters
    ----------
    image_inds : numpy 1D array (size: P), where P = number of predicted classes in batch
        The image indices of the predicted classes in batch
    class_inds : numpy 1D array (size: P), where P = number of predicted classes in batch
        The class indices of the predicted classes in batch
    scores : numpy 1D array (size: P), where P = number of predicted classes in batch
        The class confidence scores of the predicted classes in batch
    size : list of int (size: 2)
        The patch size
    out_dir : str
        The directory to save the patch confidence images to
    out_names : list of str (size: B), where B = batch size
        The name of the original patch images in the batch
    classes : list of str (size: C), where C = number of classes
        The names of the segmentation classes
    """

    if len(image_inds) != len(class_inds) or len(image_inds) != len(scores):
        raise Exception('Number of images, classes, and scores must be equal!')
    for iter_patchconfidence in range(len(image_inds)):
        out_name = out_names[image_inds[iter_patchconfidence]]
        Y = 255 * scores[iter_patchconfidence] * np.ones((size[0], size[1], 3))
        patch_conf_name = os.path.splitext(out_name)[0] + '_h' + classes[class_inds[iter_patchconfidence]] + '.png'
        patch_conf_path = os.path.join(out_dir, patch_conf_name)
        cv2.imwrite(patch_conf_path, Y)

def save_pred_segmasks(X, out_dir, out_names):
    """Save the predicted segmentation masks in the current batch to file

    Parameters
    ----------
    X : numpy 4D array (size: B x H x W x 3), where B = batch size
        The predicted segmentation masks in the current batch to save
    out_dir : str
         The directory to save the predicted segmentation masks to
    out_names : list of str (size: B), where B = batch size
        The name of the original patch images in the batch
    """

    if X.shape[0] != len(out_names):
        raise Exception('Number of files in segmasks must equal number of file names!')
    for iter_image in range(X.shape[0]):
        out_path = os.path.join(out_dir, out_names[iter_image])
        cv2.imwrite(out_path, cv2.cvtColor(X[iter_image].astype('uint8'), cv2.COLOR_RGB2BGR))

def save_cs_gradcam(X, out_dir, out_names, classes):
    """Save the predicted HTT-adjusted segmentation masks in the current batch to file

    Parameters
    ----------
    X : numpy 4D array (size: B x H x W x 3), where B = batch size
        The predicted HTT-adjusted segmentation masks in the current batch to save
    out_dir : str
         The directory to save the predicted segmentation masks to
    out_names : list of str (size: B), where B = batch size
        The name of the original patch images in the batch
    classes : list of str (size: C), where C = number of classes
        The names of the segmentation classes
    """

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

def show_values(pc, fmt="%.2f", **kw):
    '''
    Heatmap with text in each cell with matplotlib's pyplot
    Source: http://stackoverflow.com/a/25074150/395857
    By HYRY
    '''
    pc.update_scalarmappable()
    ax = pc.axes
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)

def cm2inch(*tupl):
    '''
    Specify figure size in centimeter in matplotlib
    Source: http://stackoverflow.com/a/22787457/395857
    By gns-ank
    '''
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, rot_angle=0):
    '''
    Inspired by:
    - http://stackoverflow.com/a/16124677/395857
    - http://stackoverflow.com/a/25074150/395857
    '''

    # Plot it out
    fig, ax = plt.subplots()
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='afmhot', vmin=0.0, vmax=1.0)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    # ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)
    if rot_angle == 45:
        ax.set_xticks(np.arange(AUC.shape[1]), minor=False)
        ax.set_xticklabels(xticklabels, minor=False, rotation=rot_angle, horizontalalignment='left')
    else:
        ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)
        ax.set_xticklabels(xticklabels, minor=False, rotation=rot_angle)
    # set tick labels
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    # plt.title(title, y=1.08, fontsize=14)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    axis_offset = -0.012*AUC.shape[0] + 1.436
    ax.xaxis.set_label_coords(.5, axis_offset)

    # Turn off all the ticks
    ax = plt.gca()
    ax.axis('equal')
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add text in each cell
    cell_font = 10 # math.ceil(AUC.shape[1] * 10 / 28)
    show_values(c, fontsize=cell_font)

    # Proper orientation (origin at the top left instead of bottom left)
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    # resize
    fig = plt.gcf()
    ax.axis('tight')
    fig_len = 40 / 28 * AUC.shape[0]

    fig.set_size_inches(cm2inch(fig_len, fig_len))