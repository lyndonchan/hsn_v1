import hsn_v1
import time

# 0. User-defined Settings (maybe take these from command-line some day)
MODEL_NAME = 'histonet_X1.7_clrdecay_5'
# PYRAMID_IDS = ['123_R01', 'P8', 'P14', 'PP1', 'PP5', 'PP7', 'PP19', 'PP49', 'PP51', 'PP53', 'PP59', 'PP60']
PYRAMID_IDS = ['PP7', 'PP51', 'PP53', 'PP59', 'PP60']
INPUT_NAMES = []
for pyramid_id in PYRAMID_IDS:
    INPUT_NAMES.append('03_wsi_' + pyramid_id + '_overlap')
INPUT_MODE = 'wsi'                      # {'patch', 'wsi'}
INPUT_SIZE = [224, 224]                 # [<int>, <int>] > 0
HTT_MODE = 'both'                      # {'both', 'morph', 'func', 'glas'}
BATCH_SIZE = 16                         # int > 0
GT_MODE = 'off'                         # {'on', 'off'}
RUN_LEVEL = 1                           # {1: HTT confidence scores, 2: Grad-CAMs, 3: Segmentation masks}
SAVE_TYPES = [1, 0, 0, 0]               # {HTT confidence scores, Grad-CAMs, Segmentation masks, Summary images}
VERBOSITY = 'NORMAL'                    # {'NORMAL', 'QUIET'}

for input_name in INPUT_NAMES:
    print('Running for input file ' + input_name)
    # Settings for image set
    # IN_PX_RESOL = 0.620
    # OUT_PX_RESOL = 0.25 * 1088 / 224    # 1.21428571429
    # DOWNSAMPLE_FACTOR = OUT_PX_RESOL / IN_PX_RESOL
    DOWNSAMPLE_FACTOR = 1

    # 1. Setup HistoSegNetV1
    hsn = hsn_v1.HistoSegNetV1(params={'input_name': input_name, 'input_size': INPUT_SIZE, 'input_mode': INPUT_MODE,
                                       'down_fac': DOWNSAMPLE_FACTOR, 'batch_size': BATCH_SIZE, 'htt_mode': HTT_MODE,
                                       'gt_mode': GT_MODE, 'run_level': RUN_LEVEL, 'save_types': SAVE_TYPES,
                                       'verbosity': VERBOSITY})
    # 2. Find image(s)
    hsn.find_img()
    hsn.analyze_img()

    # 3. Loading HistoNet
    hsn.load_histonet(params={'model_name': MODEL_NAME})

    # 3. Batch-wise operation
    hsn.run_batch()