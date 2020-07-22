import numpy as np

import tensorflow as tf
import tf.keras as K


from RAN import RAN
from config import cfg


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

import pickle
import cv2
import os
import math
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import math as ma
import time

from predict_next_frame import *
from qt_orient import *
from qt_segment import *
from qt_sets import *
from qt_squares_bboxes import *
from qt_viterbi_segm import *
from qt_tree2indim import *
from qt_indim2im import *
from qt_correct import *
from qt_create_clip import *
from qt_reconstruct_b import *
from qt_values import *
from qt_plot_correction import *
from sort import *
import tensorflow as tf
from keras import backend as K
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout

from keras.models import Model
from Faster_RCNN import *

# Actual ReImagine Code Starts


# -------------------- Parse input arguments --------------------
if len(sys.argv) < 4:
    print("Not enough arguments", len(sys.argv))
    print(sys.argv)
    exit()
lamb = int(sys.argv[1])
videoFile = sys.argv[2]
outputDir = sys.argv[3]
#  -------------------- END Parse input arguments  --------------------





# ------------------------------- Start Initializing Faster R-CNN ------------------------------------

# config = tf.ConfigProto()
config = tf.ConfigProto(device_count={'GPU': 0})  #
# config.gpu_options.per_process_gpu_memory_fraction = 0.25
session = tf.Session(config=config)
K.set_session(session)

# Start Initializing Faster R-CNN
base_path = '/media/srutarshi/DATAPART1/Srutarshi/GenDistData_npDist_binn_v2/Zeus_npdist_binn_v2'
config_output_filename = os.path.join(base_path, 'model_vgg_config.pickle')

with open(config_output_filename, 'rb') as f_in:
    C = pickle.load(f_in)

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

num_features = 512
input_shape_img = (None, None, 3)
input_shape_features = (None, None, num_features)
img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)
# define the base network (VGG here, can be Resnet50, Inception, etc)
shared_layers = nn_base(img_input, trainable=True)
# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = rpn_layer(shared_layers, num_anchors)
classifier = classifier_layer(feature_map_input, roi_input, C.num_rois, nb_classes=len(C.class_mapping))
model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)
model_classifier = Model([feature_map_input, roi_input], classifier)
print('Loading weights from {}'.format(C.model_path))
model_rpn.load_weights(C.model_path, by_name=True)
model_classifier.load_weights(C.model_path, by_name=True)
model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')
# Returns "model_rpn, model_

# Switch key value for class mapping
class_mapping = C.class_mapping
class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}

# If the box classification value is less than this, we ignore this box
bbox_threshold = 0.7

print("[INFO] loading model...")

# ------------------------------- End Initializing Faster R-CNN ------------------------------------


# ------------------------------- TEST VIDEO PREPROCESSING STARTS ----------------------------------
# Directory where video files are accessed
file_p = videoFile

# file_p = sys.argv[1]
ff = os.path.basename(file_p)



# Directory where distorted video files are stored
rootdir_det = outputDir + str(lamb)

# Make directory if ff doesn't exist, make new directory
if not os.path.exists(rootdir_det):
    os.mkdir(rootdir_det)

# construct the argument parse and parse the arguments
COLORS = np.random.uniform(0, 255, size=(21, 3))
new_rows = 512  # rows in new image
new_cols = 512  # cols in new image

# file_p = os.path.join(rootdir, ff)
print('file=', file_p)
vs = cv2.VideoCapture(file_p)  # CAPTURE VIDEO FILE
time.sleep(2.0)

rootdir_d_fi = os.path.splitext(ff)[0]
print(rootdir_d_fi)
rootd_det = os.path.join(rootdir_det, rootdir_d_fi)

NF = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))  # this depends on the OpenCV version and the video codecs installed
print('NF=', NF)
ori_width = vs.get(3)
ori_height = vs.get(4)
print('ori_width=', ori_width)
print('ori_height=', ori_height)

out_rate = np.zeros((1, 3))
vs.set(cv2.CAP_PROP_POS_FRAMES, 0)  # captures 1st frame
[gr1, f1] = vs.read()
f1 = cv2.resize(f1, (new_rows, new_cols), interpolation=cv2.INTER_AREA)
f1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)

vs.set(cv2.CAP_PROP_POS_FRAMES, 1)  # captures 2nd frame
[gr2, f2] = vs.read()
f2 = cv2.resize(f2, (new_rows, new_cols), interpolation=cv2.INTER_AREA)
f2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

# ------------------------------- TEST VIDEO PREPROCESSING ENDS ----------------------------------


# ------------------------------- Configuring Frame Count ---------------------------------
nf = min(NF, NF)
# nf = NF  # nf < NF for partial experiments, nf = NF for full experiments
# nf = np.clip(nf,0,900) # number of frames clipped to 900 frames
visualize = 1  # Take this as the input from user at a later stage

N = int(ma.log(512, 2))
n0 = 0
Nl = N - n0 + 1
Max_Rate = 512 * 512 * 8
# ------------------------------- END Configuring Frame Count ---------------------------------


# Create Object Tracker - instance of SORT created
mot_tracker = Sort()

# ------------------------------- Quadtree ---------------------------------
# end of line 41 in MATLAB code
print('Creating qt-related global variables...')
o = qt_orient(Nl)
a = qt_segment(Nl)
[f, t] = qt_sets(Nl)
sq = qt_squares_bboxes(Nl, o)

# end of line 55 in MATLAB code
f_rec = np.zeros((nf, new_rows, new_cols), dtype='uint8')
f_rec[0, :, :] = f1.copy()
f_rec[1, :, :] = f2.copy()

rate = np.zeros((nf - 2, 1))
flow = np.zeros((nf - 2, 3, 512, 512))
qtsq = np.zeros((nf - 2, 3, 512, 512))  # save quadtree S & Q for video
segment = np.ones((nf - 2, 512, 512))
segm_track = np.ones((nf - 2, 512, 512))
# ----------------------------------------------------------------


# -------------------------- Start Preparing Output --------------------------

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
rootdir_wp = os.path.join(rootdir_det, rootdir_d_fi + '.mp4')
out = cv2.VideoWriter(rootdir_wp, fourcc, 30.0, (int(new_rows + 0.5), int(new_cols + 0.5)))

outfile_tracked = os.path.join(rootdir_det, rootdir_d_fi + '_tracked.txt')
outfile_dets = os.path.join(rootdir_det, rootdir_d_fi + '_detections.txt')

fil = open(outfile_tracked, "a+")
fil1 = open(outfile_dets, "a+")

# --------------------------- END Preparing Output ----------------------------


print('\n Starting simulation with NF = ', NF)
for k in range(3, nf + 1):
    print('Trial: ', k)
    f1 = f_rec[k - 3, :, :]
    f2 = f_rec[k - 2, :, :]
    # obtain new frame on Chip
    vs.set(cv2.CAP_PROP_POS_FRAMES, k - 1)
    [gr3, f3] = vs.read()
    f3 = cv2.resize(f3, (new_rows, new_cols), interpolation=cv2.INTER_AREA)
    f3 = cv2.cvtColor(f3, cv2.COLOR_BGR2GRAY)
    predBboxes = []
    predClass = []
    predProb = []
    area = 0
    if k > 3:
        [trks, to_del, trks_cls, trks_prob] = mot_tracker.predict()
        trks[trks < 0] = 0
        trks[trks > 511] = 511

        predBboxes = [trks[i, :] for i in range(trks.shape[0])]
        predClass = [C.class_mapping[i] for i in trks_cls]
        predProb = trks_prob[:]

        for i in range(0, trks.shape[0]):
            segm_track[k - 3, int(trks[i, 1]):int(trks[i, 3]), int(trks[i, 0]):int(trks[i, 2])] = 2
            area = area + np.multiply((trks[i, 3] - trks[i, 1]), (trks[i, 2] - trks[i, 0]))
            #print('trk co-ords', trks[i, 3], trks[i, 1], trks[i, 2], trks[i, 0])
        # print ('area',area)

    segm_track_p = segm_track[k - 3, :, :]
    area = np.clip(area, 0, 262144)

    # Run Viterbi on Host (if f3 is used - Viterbi is run on chip. If f3_rec is used - Viterbi is run on host)
    priorities = [0, 2]  # this is not used in MATLAB code (useless)
    [tree, gxf_min, total_rate] = qt_viterbi_segm(f2, f3, N, n0, o, a, f, t, segm_track_p, priorities, area, float(lamb))
    a1 = (total_rate / Max_Rate) * 100
    print('total rate after optimization: ', total_rate)
    print('% rate after optimization: ', a1)
    # a123 = np.asarray([k,total_rate,a1])
    # out_rate = np.vstack((out_rate,a123))

    # Chip receives tree (S,Q) and performs reconstruction
    [tree_rec, f3_rec, e_rec] = qt_reconstruct_b(tree, sq, f2, f3)
    dist_rec = np.sum(abs(f3 - f3_rec))

    f3_corr = f3_rec
    aa11 = (rate[k - 3] / Max_Rate) * 100
    # print('rate after correction: ',rate[k-3],aa11)

    # send corrected state and data back to host
    f_rec[k - 1, :, :] = np.copy(f3_corr)

    # OpenCV does'nt save 2 dim frames - it has to be converted to 3 dim.
    f3_save = cv2.cvtColor(f3_corr, cv2.COLOR_GRAY2RGB)
    # f3_save = np.copy(f3_corr)

    # R-CNN does the object detection per frame basis
    img = np.copy(f3_save)
    X, ratio = format_img(img, C)
    X = np.transpose(X, (0, 2, 3, 1))

    # get output layer Y1, Y2 from the RPN and the feature maps F
    # Y1: y_rpn_cls
    # Y2: y_rpn_regr
    [Y1, Y2, F] = model_rpn.predict(X)

    # Get bboxes by applying NMS 
    # R.shape = (300, 4)
    R = rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)

    # convert from (x1,y1,x2,y2) to (x,y,w,h)
    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]

    # apply the spatial pyramid pooling to the proposed regions
    bboxes = {}
    probs = {}

    def calcIoU(bbox1, bbox2):
        # Each bbox is a 4 element list - in the order x1, y1, x2, y2;
        xx1_int = np.maximum(bbox1[0], bbox2[0])
        yy1_int = np.maximum(bbox1[1], bbox2[1])
        xx2_int = np.minimum(bbox1[2], bbox2[2])
        yy2_int = np.minimum(bbox1[3], bbox2[3])
        ww_int = np.maximum(0, xx2_int - xx1_int)
        hh_int = np.maximum(0, yy2_int - yy1_int)
        area_int = ww_int * hh_int
        area_new = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area_predict = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        if area_new == 0.0 or area_predict == 0.0:
            return 0

        area_union = area_new + area_predict - area_int
        newIoU = area_int / (area_union + 1e-6) if abs(area_new - area_predict) / area_new < 0.25 else 0.1

        if newIoU < 0.2:
            return 0.0

        return newIoU

    for jk in range(R.shape[0] // C.num_rois + 1):
        ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
        if ROIs.shape[1] == 0:
            break

        if jk == R.shape[0] // C.num_rois:
            # pad R
            curr_shape = ROIs.shape
            target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
            ROIs_padded[:, :curr_shape[1], :] = ROIs
            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
            ROIs = ROIs_padded

        [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

        # Calculate bboxes coordinates on resized image
        for ii in range(P_cls.shape[1]):

            (x, y, w, h) = ROIs[0, ii, :]

            def bestIoU(class_num, box, anchorBox):
                # calculate the "best IoU"
                x_in = anchorBox[0]
                y_in = anchorBox[1]
                w_in = anchorBox[2]
                h_in = anchorBox[3]
                (tx, ty, tw, th) = P_regr[0, ii, 4 * class_num:4 * (class_num + 1)]
                tx /= C.classifier_regr_std[0]
                ty /= C.classifier_regr_std[1]
                tw /= C.classifier_regr_std[2]
                th /= C.classifier_regr_std[3]
                x_in, y_in, w_in, h_in = apply_regr(x_in, y_in, w_in, h_in, tx, ty, tw, th)
                x1, y1, x2, y2 = get_real_coordinates(ratio, C.rpn_stride * x_in, C.rpn_stride * y_in,
                                                      C.rpn_stride * (x_in + w_in), C.rpn_stride * (y_in + h_in))
                return calcIoU([x1, y1, x2, y2], box)


            if len(predBboxes) != 0 and len(predClass) != 0:
                # calculate the amount of IoU between the boxs
                bboxInfluence = [bestIoU(cls, box, [x, y, w, h]) * prob for cls, box, prob in zip(predClass, predBboxes, predProb)]
                # pick a predicted bounding box to match itself with
                bestBbox = np.argmax(bboxInfluence)
                targetClass = predClass[bestBbox]

                weight1 = 1.0
                weight2 = 0.7

                P_cls[0, ii, targetClass] = math.sqrt(weight1 * P_cls[0, ii, targetClass]**2 + weight2 * bboxInfluence[bestBbox]**2)


            if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                continue

            cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]  # get the maximum class here...
            cls_num = np.argmax(P_cls[0, ii, :])
            try:
                (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                tx /= C.classifier_regr_std[0]
                ty /= C.classifier_regr_std[1]
                tw /= C.classifier_regr_std[2]
                th /= C.classifier_regr_std[3]
                x, y, w, h = apply_regr(x, y, w, h, tx, ty, tw, th)
            except:
                pass

            if cls_name not in bboxes:
                bboxes[cls_name] = []
                probs[cls_name] = []

            bboxes[cls_name].append(
                [C.rpn_stride * x, C.rpn_stride * y, C.rpn_stride * (x + w), C.rpn_stride * (y + h)])
            probs[cls_name].append(np.max(P_cls[0, ii, :]))

    all_dets = []
    BB = []
    bb = np.zeros(4)
    key_list = []
    new_probs_up = []

    for key in bboxes:
        bbox = np.array(bboxes[key])

        new_boxes, new_probs = non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.2)
        new_probs_up = np.hstack((new_probs_up, new_probs))

        for jk in range(new_boxes.shape[0]):
            (x1, y1, x2, y2) = new_boxes[jk, :]

            # Calculate real coordinates on original image
            (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

            bb[0] = real_x1
            bb[1] = real_y1
            bb[2] = real_x2
            bb[3] = real_y2
            BB.append(bb.copy())

            # textLabel = '{}'.format(key)
            all_dets.append(key)
            # (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
            # textOrg = (real_x1, real_y1-0)

            fil1.write(str(k - 3) + ',' + str(real_x1) + ',' + str(real_y1) + ',' + str(real_x2) + ',' + str(
                real_y2) + ',' + key)
            fil1.write("\n")

    BB = np.asarray(BB)
    detectionss1 = np.copy(BB)
    print('detectionss1=', detectionss1)
    print('all_dets=', all_dets)
    #print('new_probs_up', new_probs_up)

    if k == 3:
        trks = []
        to_del = []

    rett0 = mot_tracker.update(detectionss1, all_dets, new_probs_up, trks, to_del)
    print('updated tracks ', rett0)

    for d in rett0:
        fil.write(str(k - 1) + ',' + str(int(d[4])) + ',' + str(d[0]) + ',' + str(d[1]) + ',' + str(d[2]) + ',' + str(
            d[3]) + ',' + str(d[5]) + ',' + str(d[6]) + ',' + str(total_rate))
        fil.write("\n")
        cv2.rectangle(f3_save, (int(float(d[0])), int(float(d[1]))), (int(float(d[2])), int(float(d[3]))),
                      COLORS[np.int(d[4]) % 21], 2)
    out.write(f3_save)

out.release()
vs.release()



