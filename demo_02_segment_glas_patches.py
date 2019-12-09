import hsn_v1
import time

# 0. User-defined Settings (maybe take these from command-line some day)
MODEL_NAME = 'histonet_X1.7_clrdecay_5'
INPUT_NAME = '02_glas_full'
INPUT_MODE = 'patch'                    # {'patch', 'wsi'}
INPUT_SIZE = [224, 224]                 # [<int>, <int>] > 0
HTT_MODE = 'glas'                       # {'both', 'morph', 'func', 'glas'}
BATCH_SIZE = 1                          # int > 0
GT_MODE = 'on'                         # {'on', 'off'}
RUN_LEVEL = 3                           # {1: HTT confidence scores, 2: Grad-CAMs, 3: Segmentation masks}
SAVE_TYPES = [1, 1, 1, 0]               # {HTT confidence scores, Grad-CAMs, Segmentation masks, Summary images}
VERBOSITY = 'NORMAL'                    # {'NORMAL', 'QUIET'}

# Settings for image set
IN_PX_RESOL = 0.620
OUT_PX_RESOL = 0.25 * 1088 / 224    # 1.21428571429
DOWNSAMPLE_FACTOR = OUT_PX_RESOL / IN_PX_RESOL
# DOWNSAMPLE_FACTOR = 1

# 1. Setup HistoSegNetV1
hsn = hsn_v1.HistoSegNetV1(params={'input_name': INPUT_NAME, 'input_size': INPUT_SIZE, 'input_mode': INPUT_MODE,
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