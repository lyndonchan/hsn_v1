import numpy as np

class Atlas:
    def __init__(self):
        self.level1 = ['E', 'C', 'H', 'S', 'A', 'M', 'N', 'G', 'T']
        self.level2 = ['E.M', 'E.T', 'E.P', 'C.D', 'C.L', 'H.E', 'H.K', 'H.Y', 'S.M', 'S.E', 'S.C', 'S.R', 'A.W',
                       'A.B', 'A.M', 'M.M', 'M.K', 'N.P', 'N.R', 'N.G', 'G.O', 'G.N', 'T']
        self.level3 = ['E.M.S', 'E.M.U', 'E.M.O', 'E.T.S', 'E.T.U', 'E.T.O', 'E.P', 'C.D.I', 'C.D.R', 'C.L', 'H.E',
                       'H.K', 'H.Y', 'S.M.C', 'S.M.S', 'S.E', 'S.C.H', 'S.R', 'A.W', 'A.B', 'A.M', 'M.M', 'M.K',
                       'N.P', 'N.R.B', 'N.R.A', 'N.G.M', 'N.G.A', 'N.G.O', 'N.G.E', 'N.G.R', 'N.G.W', 'N.G.T',
                       'G.O', 'G.N', 'T']
        self.level4 = ['E', 'E.M', 'E.T', 'E.P', 'C', 'C.D', 'C.L', 'H', 'H.E', 'H.K', 'H.Y', 'S', 'S.M', 'S.E',
                       'S.C', 'S.R', 'A', 'A.W', 'A.B', 'A.M', 'M', 'M.M', 'M.K', 'N', 'N.P', 'N.R', 'N.G', 'G',
                       'G.O', 'G.N', 'T']
        self.level5 = ['E', 'E.M', 'E.M.S', 'E.M.U', 'E.M.O', 'E.T', 'E.T.S', 'E.T.U', 'E.T.O', 'E.P', 'C', 'C.D',
                       'C.D.I', 'C.D.R', 'C.L', 'H', 'H.E', 'H.K', 'H.Y', 'S', 'S.M', 'S.M.C', 'S.M.S', 'S.E',
                       'S.C', 'S.C.H', 'S.R', 'A', 'A.W', 'A.B', 'A.M', 'M', 'M.M', 'M.K', 'N', 'N.P', 'N.R',
                       'N.R.B', 'N.R.A', 'N.G', 'N.G.M', 'N.G.A', 'N.G.O', 'N.G.E', 'N.G.R', 'N.G.W', 'N.G.T',
                       'G', 'G.O', 'G.N', 'T']

        self.level5_inds_in_level3 = [i for i,x in enumerate(self.level5) if np.isin(x, self.level3)]
        self.level5_in_level3 = [x for i,x in enumerate(self.level5) if np.isin(x, self.level3)]

        self.morph_classes = ['Background', 'E.M.S', 'E.M.U', 'E.M.O', 'E.T.S', 'E.T.U', 'E.T.O', 'E.T.X',
                             'E.P', 'C.D.I', 'C.D.R', 'C.L', 'C.X', 'H.E', 'H.K', 'H.Y', 'H.X', 'S.M.C',
                             'S.M.S', 'S.E', 'S.C.H', 'S.C.X', 'S.R', 'A.W', 'A.B', 'A.M', 'M.M', 'M.K',
                             'N.P', 'N.R.B', 'N.R.A', 'N.G.M', 'N.G.W', 'N.G.X']
        self.morph_colours = np.array([[255, 255, 255], [0, 0, 128], [0, 128, 0], [255, 165, 0], [255, 192, 203],
                                      [255, 0, 0], [173, 20, 87], [0, 204, 184], [176, 141, 105], [3, 155, 229],
                                      [158, 105, 175], [216, 27, 96], [131, 81, 63], [244, 81, 30], [124, 179, 66],
                                      [142, 36, 255], [230, 124, 115], [240, 147, 0], [204, 25, 165], [121, 85, 72],
                                      [142, 36, 170], [249, 127, 57], [179, 157, 219], [121, 134, 203], [97, 97, 97],
                                      [167, 155, 142], [228, 196, 136], [213, 0, 0], [4, 58, 236], [0, 150, 136],
                                      [228, 196, 65], [239, 108, 0], [74, 21, 209], [148, 0, 0]])
        self.func_classes = ['Background', 'Other', 'G.O', 'G.N', 'G.X', 'T']
        self.func_colours = np.array([[255, 255, 255], [3, 155, 229], [0, 0, 128], [0, 128, 0], [255, 165, 0],
                                      [173, 20, 87]])
        self.glas_valid_classes = ['Other', 'G.O']
        self.glas_valid_colours = np.array([[3, 155, 229], [0, 0, 128]])

        # Checks
        if len(np.unique(self.morph_classes, axis=0)) != len(self.morph_classes):
            raise Exception('You have duplicate classes for morphological HTTs')
        if len(np.unique(self.func_classes, axis=0)) != len(self.func_classes):
            raise Exception('You have duplicate classes for functional HTTs')
        if len(np.unique(self.morph_colours, axis=0)) != len(self.morph_colours):
            raise Exception('You have duplicate colours for morphological HTTs')
        if len(np.unique(self.func_colours, axis=0)) != len(self.func_colours):
            raise Exception('You have duplicate colours for functional HTTs')

        morph_valid_class_inds = [i for i, x in enumerate(self.morph_classes) if '.X' not in x]
        func_valid_class_inds = [i for i, x in enumerate(self.func_classes) if '.X' not in x]
        self.morph_valid_classes = [self.morph_classes[i] for i in morph_valid_class_inds]
        self.func_valid_classes = [self.func_classes[i] for i in func_valid_class_inds]
        self.morph_valid_colours = self.morph_colours[morph_valid_class_inds]
        self.func_valid_colours = self.func_colours[func_valid_class_inds]

        self.level3_valid_inds = [i for i, x in enumerate(self.level5) if np.isin(x,self.level3) and
                                  (np.isin(x, self.morph_valid_classes) or np.isin(x, self.func_valid_classes))]
        a=1

    def convert_class_inds(self, class_inds_in, classes_in, classes_out):
        classes_out = np.array([classes_out.index(classes_in[x]) for x in class_inds_in])
        return classes_out

    def onehot_to_class_inds(self, onehot_x):
        offset_morph = 1
        offset_func = 2

        num_morph = len(self.morph_classes) - offset_morph
        num_func = len(self.func_classes) - offset_func

        cur_gt_morph_inds = onehot_x[:num_morph]
        cur_gt_func_inds = onehot_x[num_morph:num_morph + num_func]
        cur_gt_morph_class_inds = [i+offset_morph for i, x in enumerate(cur_gt_morph_inds) if x]
        cur_gt_func_class_inds = [i+offset_func for i, x in enumerate(cur_gt_func_inds) if x]

        cur_gt_morph_class_inds = [i for i in cur_gt_morph_class_inds if '.X' not in self.morph_classes[i]]
        cur_gt_func_class_inds = [i for i in cur_gt_func_class_inds if '.X' not in self.func_classes[i]]

        return cur_gt_morph_class_inds, cur_gt_func_class_inds