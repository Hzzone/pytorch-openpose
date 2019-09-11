import math

import cv2
import numpy as np

MPII_TO_OPENPOSE_IDX = {
    0: 10,  # rank
    1: 9,  # rkne
    2: 8,  # rhip
    3: 11,  # lhip
    4: 12,  # lkne
    5: 13,  # lank
    # 6: ,  # pelvis
    # 7:,  # thorax
    8: 1,  # neck
    9: 0,  # head
    10: 4,  # rwri
    11: 3,  # relb
    12: 2,  # rsho
    13: 5,  # lsho
    14: 6,  # lelb
    15: 7,  # lwri
}


def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride) # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad


# transfer caffe model to pytorch which will match the layer name
def transfer(model, model_weights):
    transfered_model_weights = {}
    for weights_name in model.state_dict().keys():
        transfered_model_weights[weights_name] = model_weights['.'.join(weights_name.split('.')[1:])]
    return transfered_model_weights


# draw the body keypoint and lims
def draw_bodypose(canvas, candidate, subset):
    stickwidth = 4
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    # plt.imsave("preview.jpg", canvas[:, :, [2, 1, 0]])
    # plt.imshow(canvas[:, :, [2, 1, 0]])
    return canvas


# get max index of 2d array
def npmax(array):
    arrayindex = array.argmax(1)
    arrayvalue = array.max(1)
    i = arrayvalue.argmax()
    j = arrayindex[i]
    return i, j


def get_openpose_annotation(mpii_annotation):
    """
    Converts MPII annotations to OpenPose formatted joint annotations

    Inputs
    ------
    mpii_annotations: dict
        Dictionary of mpii annotation values for a single image. Keys include 'joint_self', etc.

    Returns
    -------
    openpose_joints: array-like [N_people, 15, 2]
        Joints of people in the image in OpenPose ordering
    """
    # Get joints in OpenPose format (excluding center point for now)
    people = np.expand_dims(np.array(mpii_annotation['joint_self'])[:, :2], axis=0)
    n_people = mpii_annotation['numOtherPeople'] + 1
    if n_people > 1:
        joint_others = np.array(mpii_annotation['joint_others'])

        if n_people == 2:
            joint_others = np.expand_dims(joint_others, axis=0)[:, :, :2]
        people = np.vstack((people, joint_others))

    openpose_joints = np.zeros((len(people), 14, 2))
    for mpii_idx, openpose_idx in MPII_TO_OPENPOSE_IDX.items():
        openpose_joints[:, openpose_idx, :] = people[:, mpii_idx, :]

    # Add center points
    center_points = np.expand_dims(mpii_annotation['objpos'], axis=0)
    if n_people > 1:
        cp_other = mpii_annotation['objpos_other']
        if n_people == 2:
            cp_other = np.expand_dims(cp_other, axis=0)
        center_points = np.vstack((center_points, cp_other))
        center_points = np.expand_dims(center_points, axis=1)

    # Final joints
    openpose_joints = np.hstack((openpose_joints, center_points))
    return openpose_joints


def get_pafs(people, image_resolution, stride=8, threshold=1):
    """
    Compute PAFs for given joint annotations.

    Ported from https://github.com/CMU-Perceptual-Computing-Lab/caffe_train/blob/
    76dd9563fb24cb1702d0245cda7cc36ec2aed43b/src/caffe/cpm_data_transformer.cpp

    Inputs
    ------
    people: array-like [Nx15x2]
        Joints of people in image in OpenPose ordering

    image_resolution: array-like (2,)

    stride: int

    threshold: int

    Returns
    -------
    pafs: array-like [height / stride, width / stride, 28]
        Part affinity fields.
    """
    mapIdx = np.array(
        [[16, 17], [18, 19], [20, 21], [22, 23], [24, 25], [26, 27], [28, 29],
         [30, 31], [38, 39], [40, 41], [42, 43], [32, 33], [34, 35], [36, 37]]) - 16
    limbSeq = np.array(
        [[1, 2], [2, 3], [3, 4], [4, 5], [2, 6], [6, 7], [7, 8], [2, 15],
         [15, 12], [12, 13], [13, 14], [15, 9], [9, 10], [10, 11]]) - 1

    paf_resolution = np.array(image_resolution)[:2] // stride
    paf_channels = len(limbSeq) * 2

    pafs = np.zeros((paf_resolution[0], paf_resolution[1], paf_channels))
    count = np.zeros((paf_resolution[0], paf_resolution[1], len(limbSeq)))

    assert people.shape[1:] == (15, 2)

    for person in people:
        for i, limb in enumerate(limbSeq):
            x_j1, x_j2 = person[limb[0]], person[limb[1]]
            x_j1 = x_j1 * 1.0 / stride
            x_j2 = x_j2 * 1.0 / stride

            height, width = pafs.shape[:2]
            min_x = max(int(round(min(x_j1[0], x_j2[0]) - threshold)), 0)
            max_x = min(int(round(max(x_j1[0], x_j2[0]) + threshold)), width)
            min_y = max(int(round(min(x_j1[1], x_j2[1]) - threshold)), 0)
            max_y = min(int(round(max(x_j1[1], x_j2[1]) + threshold)), height)

            v = (x_j2 - x_j1) / np.linalg.norm(x_j2 - x_j1, ord=2)
            paf_index = mapIdx[i]

            for r in range(min_y, max_y):
                for c in range(min_x, max_x):
                    px = c - x_j1[0]
                    py = r - x_j1[1]

                    distance = abs(v[1] * px - v[0] * py)
                    if distance <= threshold:
                        pafs[r, c, paf_index[0]] = (pafs[r, c, paf_index[0]] * count[r, c, i] + v[0]) / (
                                    count[r, c, i] + 1)
                        pafs[r, c, paf_index[1]] = (pafs[r, c, paf_index[1]] * count[r, c, i] + v[1]) / (
                                    count[r, c, i] + 1)
                        count[r, c, i] += 1

    return pafs


def get_gaussian_maps(people, image_resolution, stride=8, sigma=7):
    """
    Generate ground truth Gaussian Map given mpii annotations.

    Ported from https://github.com/CMU-Perceptual-Computing-Lab/caffe_train/blob/
    76dd9563fb24cb1702d0245cda7cc36ec2aed43b/src/caffe/cpm_data_transformer.cpp#L1021
    """
    grid_y, grid_x = np.array(image_resolution[:2]) // stride
    all_heatmaps = []
    start = stride / 2 - 0.5

    # Generate gaussian map for every person, then combine using their maximums
    for person in people:
        person_heatmap = np.zeros((grid_y, grid_x, len(people[0])))

        for i, joint in enumerate(person):

            # Loop through each point in the heatmap for this joint and compute accordingly
            for g_y in range(grid_y):
                for g_x in range(grid_x):
                    x = start + g_x * stride
                    y = start + g_y * stride

                    d2 = (x - joint[0]) ** 2 + (y - joint[1]) ** 2
                    exponent = d2 / 2 / sigma / sigma
                    if exponent > 4.60523:
                        continue

                    person_heatmap[g_y, g_x, i] += np.exp(-exponent)

        all_heatmaps.append(person_heatmap)

    # Get maximum of the heatmaps and return
    return np.max(all_heatmaps, axis=0)
