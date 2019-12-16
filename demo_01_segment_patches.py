import hsn_v1

# User-defined Settings
MODEL_NAME = 'histonet_X1.7_clrdecay_5'
INPUT_NAME = '01_tuning_patch'
INPUT_MODE = 'patch'                    # {'patch', 'wsi'}
INPUT_SIZE = [224, 224]                 # [<int>, <int>] > 0
HTT_MODE = 'both'                       # {'both', 'morph', 'func', 'glas'}
BATCH_SIZE = 16                         # int > 0
GT_MODE = 'on'                          # {'on', 'off'}
RUN_LEVEL = 3                           # {1: HTT confidence scores, 2: Grad-CAMs, 3: Segmentation masks}
SAVE_TYPES = [1, 1, 1, 1]               # {HTT confidence scores, Grad-CAMs, Segmentation masks, Summary images}
VERBOSITY = 'NORMAL'                    # {'NORMAL', 'QUIET'}
DOWNSAMPLE_FACTOR = 1

# Setup HistoSegNetV1
hsn = hsn_v1.HistoSegNetV1(params={'input_name': INPUT_NAME, 'input_size': INPUT_SIZE, 'input_mode': INPUT_MODE,
                                   'down_fac': DOWNSAMPLE_FACTOR, 'batch_size': BATCH_SIZE, 'htt_mode': HTT_MODE,
                                   'gt_mode': GT_MODE, 'run_level': RUN_LEVEL, 'save_types': SAVE_TYPES,
                                   'verbosity': VERBOSITY})

# Find image(s)
hsn.find_img()
hsn.analyze_img()

# Loading HistoNet
hsn.load_histonet(params={'model_name': MODEL_NAME})

# Batch-wise operation
hsn.run_batch()