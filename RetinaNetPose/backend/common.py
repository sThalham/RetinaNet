"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import keras.backend
from .dynamic import meshgrid


def bbox_transform_inv(boxes, deltas, mean=None, std=None):
    if mean is None:
        mean = [0, 0, 0, 0]
    if std is None:
        std = [0.2, 0.2, 0.2, 0.2]

    width  = boxes[:, :, 2] - boxes[:, :, 0]
    height = boxes[:, :, 3] - boxes[:, :, 1]

    x1 = boxes[:, :, 0] + (deltas[:, :, 0] * std[0] + mean[0]) * width
    y1 = boxes[:, :, 1] + (deltas[:, :, 1] * std[1] + mean[1]) * height
    x2 = boxes[:, :, 2] + (deltas[:, :, 2] * std[2] + mean[2]) * width
    y2 = boxes[:, :, 3] + (deltas[:, :, 3] * std[3] + mean[3]) * height

    pred_boxes = keras.backend.stack([x1, y1, x2, y2], axis=2)

    return pred_boxes


def xy_transform_inv(poses, deltas, mean=None, std=None):
    if mean is None:
       mean = [0.0, 0.0]
    if std is None:
        std = [0.2, 0.2]

    width = poses[:, :, 2] - poses[:, :, 0]
    height = poses[:, :, 3] - poses[:, :, 1]

    x = poses[:, :, 0] + (deltas[:, :, 0] * std[0] + mean[0]) * width
    y = poses[:, :, 1] + (deltas[:, :, 1] * std[1] + mean[1]) * height

    pred_pose = keras.backend.stack([x, y], axis=2)

    return pred_pose


def depth_transform_inv(poses, deltas, mean=None, std=None):
    if mean is None:
        mean = [1.0]
    if std is None:
        std = [1.0]

    z = deltas[:, :, 0] * std[0] + mean[0]

    pred_pose = keras.backend.stack([z], axis=2)

    return pred_pose


def rotation_transform_inv(poses, deltas, mean=None, std=None):
    if mean is None:
        mean = [0.0, 0.0, 0.0, 0.0]
    if std is None:
        std = [1.0, 1.0, 1.0, 1.0]

    subTensors = []
    for i in range(0, keras.backend.int_shape(deltas)[3]):
        rx = deltas[:, :, 0, i] * std[0] + mean[0]
        ry = deltas[:, :, 1, i] * std[1] + mean[1]
        rz = deltas[:, :, 2, i] * std[2] + mean[2]
        rw = deltas[:, :, 3, i] * std[3] + mean[3]

        pred_pose = keras.backend.stack([rx, ry, rz, rw], axis=2)
        pred_pose = keras.backend.expand_dims(pred_pose, axis=3)
        subTensors.append(pred_pose)
    pose_cls = keras.backend.concatenate(subTensors, axis=3)
    print(pose_cls)

    return pose_cls


def shift(shape, stride, anchors):
    # shifts anchors
    shift_x = (keras.backend.arange(0, shape[1], dtype=keras.backend.floatx()) + keras.backend.constant(0.5, dtype=keras.backend.floatx())) * stride
    shift_y = (keras.backend.arange(0, shape[0], dtype=keras.backend.floatx()) + keras.backend.constant(0.5, dtype=keras.backend.floatx())) * stride

    shift_x, shift_y = meshgrid(shift_x, shift_y)
    shift_x = keras.backend.reshape(shift_x, [-1])
    shift_y = keras.backend.reshape(shift_y, [-1])

    shifts = keras.backend.stack([
        shift_x,
        shift_y,
        shift_x,
        shift_y
    ], axis=0)

    shifts = keras.backend.transpose(shifts)
    number_of_anchors = keras.backend.shape(anchors)[0]

    k = keras.backend.shape(shifts)[0]  # number of base points = feat_h * feat_w

    shifted_anchors = keras.backend.reshape(anchors, [1, number_of_anchors, 4]) + keras.backend.cast(keras.backend.reshape(shifts, [k, 1, 4]), keras.backend.floatx())
    shifted_anchors = keras.backend.reshape(shifted_anchors, [k * number_of_anchors, 4])

    return shifted_anchors
