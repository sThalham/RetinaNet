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

import keras
from .. import backend


def filter_detections(
    boxes,
    #xy,
    #depths,
    rotations,
    classification,
    other                 = [],
    class_specific_filter = True,
    nms                   = True,
    score_threshold       = 0.05,
    max_detections        = 300,
    nms_threshold         = 0.5
):

    def _filter_detections(scores, labels):
        indices = backend.where(keras.backend.greater(scores, score_threshold))

        if nms:
            filtered_boxes  = backend.gather_nd(boxes, indices)
            filtered_scores = keras.backend.gather(scores, indices)[:, 0]
            nms_indices = backend.non_max_suppression(filtered_boxes, filtered_scores, max_output_size=max_detections, iou_threshold=nms_threshold)
            indices = keras.backend.gather(indices, nms_indices)

        labels = backend.gather_nd(labels, indices)
        indices = keras.backend.stack([indices[:, 0], labels], axis=1)

        return indices

    if class_specific_filter:
        all_indices = []

        for c in range(int(classification.shape[1])):
            scores = classification[:, c]
            labels = c * backend.ones((keras.backend.shape(scores)[0],), dtype='int64')
            all_indices.append(_filter_detections(scores, labels))

        indices = keras.backend.concatenate(all_indices, axis=0)
    else:
        scores  = keras.backend.max(classification, axis    = 1)
        labels  = keras.backend.argmax(classification, axis = 1)
        indices = _filter_detections(scores, labels)


    scores              = backend.gather_nd(classification, indices)
    labels              = indices[:, 1]
    # labels = along axis=0 cls
    scores, top_indices = backend.top_k(scores, k=keras.backend.minimum(max_detections, keras.backend.shape(scores)[0]))

    indices             = keras.backend.gather(indices[:, 0], top_indices)
    boxes               = keras.backend.gather(boxes, indices)
    #xy                  = keras.backend.gather(xy, indices)
    #depths           = keras.backend.gather(depths, indices)
    #rotations = rotations[:, :, labels[0]]
    rotations           = keras.backend.gather(rotations, indices)
    labels              = keras.backend.gather(labels, top_indices)
    rotations = rotations[:, :, labels[0]]
    other_              = [keras.backend.gather(o, indices) for o in other]

    # zero padding
    pad_size  = keras.backend.maximum(0, max_detections - keras.backend.shape(scores)[0])
    boxes     = backend.pad(boxes, [[0, pad_size], [0, 0]], constant_values=-1)
    #xy        = backend.pad(xy, [[0, pad_size], [0, 0]], constant_values=-1)
    #depths    = backend.pad(depths, [[0, pad_size], [0, 0]], constant_values=-1)
    #rotations = backend.pad(rotations, [[0, pad_size], [0, 0], [0, 0]], constant_values=-1)
    rotations = backend.pad(rotations, [[0, pad_size], [0, 0]], constant_values=-1)
    scores    = backend.pad(scores, [[0, pad_size]], constant_values=-1)
    labels    = backend.pad(labels, [[0, pad_size]], constant_values=-1)
    labels    = keras.backend.cast(labels, 'int32')
    other_    = [backend.pad(o, [[0, pad_size]] + [[0, 0] for _ in range(1, len(o.shape))], constant_values=-1) for o in other_]

    boxes.set_shape([max_detections, 4])
    #xy.set_shape([max_detections, 2])
    #depths.set_shape([max_detections, 1])
    rotations.set_shape([max_detections, 4])
    scores.set_shape([max_detections])
    labels.set_shape([max_detections])
    for o, s in zip(other_, [list(keras.backend.int_shape(o)) for o in other]):
        o.set_shape([max_detections] + s[1:])

    return [boxes, rotations, scores, labels] + other_


class FilterDetections(keras.layers.Layer):

    def __init__(
        self,
        nms                   = True,
        class_specific_filter = True,
        nms_threshold         = 0.5,
        score_threshold       = 0.05,
        max_detections        = 300,
        parallel_iterations   = 32,
        **kwargs
    ):
        self.nms                   = nms
        self.class_specific_filter = class_specific_filter
        self.nms_threshold         = nms_threshold
        self.score_threshold       = score_threshold
        self.max_detections        = max_detections
        self.parallel_iterations   = parallel_iterations
        super(FilterDetections, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        boxes          = inputs[0]
        #xy             = inputs[1]
        #depths         = inputs[2]
        rotations      = inputs[1]
        classification = inputs[2]
        other          = inputs[3:]

        def _filter_detections(args):
            boxes          = args[0]
            #xy             = args[1]
            #depths         = args[2]
            rotations      = args[1]
            classification = args[2]
            other          = args[3]

            return filter_detections(
                boxes,
                #xy,
                #depths,
                rotations,
                classification,
                other,
                nms                   = self.nms,
                class_specific_filter = self.class_specific_filter,
                score_threshold       = self.score_threshold,
                max_detections        = self.max_detections,
                nms_threshold         = self.nms_threshold,
            )

        # call filter_detections on each batch
        outputs = backend.map_fn(
            _filter_detections,
            elems=[boxes, rotations, classification, other],
            dtype=[keras.backend.floatx(), keras.backend.floatx(), keras.backend.floatx(), 'int32'] + [o.dtype for o in other],
            parallel_iterations=self.parallel_iterations
        )

        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (input_shape[0][0], self.max_detections, 4),
            #(input_shape[1][0], self.max_detections, 2),
            #(input_shape[2][0], self.max_detections, 1),
            (input_shape[1][0], self.max_detections, 4),
            (input_shape[2][0], self.max_detections),
            (input_shape[2][0], self.max_detections),
        ] + [
            tuple([input_shape[i][0], self.max_detections] + list(input_shape[i][3:])) for i in range(3, len(input_shape))
        ]

    def compute_mask(self, inputs, mask=None):
        # Keras requires this if more than one output
        return (len(inputs) + 1) * [None]

    def get_config(self):
        config = super(FilterDetections, self).get_config()
        config.update({
            'nms'                   : self.nms,
            'class_specific_filter' : self.class_specific_filter,
            'nms_threshold'         : self.nms_threshold,
            'score_threshold'       : self.score_threshold,
            'max_detections'        : self.max_detections,
            'parallel_iterations'   : self.parallel_iterations,
        })

        return config
