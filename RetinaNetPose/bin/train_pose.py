#!/usr/bin/env python

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

import argparse
import os
import sys
import warnings

import keras
import keras.preprocessing.image
import tensorflow as tf

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import RetinaNetPose.bin  # noqa: F401
    __package__ = "RetinaNetPose.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from .. import layers  # noqa: F401
from .. import losses
from .. import models
from ..callbacks import RedirectModel
from ..callbacks.eval import Evaluate
from ..models.retinanet import retinanet_bbox
from ..utils.config import read_config_file, parse_anchor_parameters
from ..utils.keras_version import check_keras_version
from ..utils.model import freeze as freeze_model
from ..utils.transform import random_transform_generator


def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def get_session():

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def model_with_weights(model, weights, skip_mismatch):

    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model


def create_models(backbone_retinanet, num_classes, weights, multi_gpu=0, freeze_backbone=False, config=None):

    modifier = freeze_model if freeze_backbone else None

    # load anchor parameters, or pass None (so that defaults will be used)
    anchor_params = None
    num_anchors   = None
    if config and 'anchor_parameters' in config:
        anchor_params = parse_anchor_parameters(config)
        num_anchors   = anchor_params.num_anchors()

    model = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier), weights=weights, skip_mismatch=True)
    training_model = model

    # make prediction model
    prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params)

    # compile model
    training_model.compile(
        loss={
            'bbox' : losses.smooth_l1(),
            #'xy_reg': losses.smooth_l1(),
            #'dep_est': losses.smooth_l1(),
            'rotation': losses.weighted_MSE(),
            'cls': losses.focal()
        },
        optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
    )

    return model, training_model, prediction_model


def create_callbacks(model, training_model, prediction_model, validation_generator, args):

    callbacks = []

    tensorboard_callback = None

    if args.tensorboard_dir:
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir                = args.tensorboard_dir,
            histogram_freq         = 0,
            batch_size             = args.batch_size,
            write_graph            = True,
            write_grads            = False,
            write_images           = False,
            embeddings_freq        = 0,
            embeddings_layer_names = None,
            embeddings_metadata    = None
        )
        callbacks.append(tensorboard_callback)

    if args.evaluation and validation_generator:
        if args.dataset_type == 'coco':
            from ..callbacks.coco import CocoEval

            evaluation = CocoEval(validation_generator, tensorboard=tensorboard_callback)
        else:
            evaluation = Evaluate(validation_generator, tensorboard=tensorboard_callback, weighted_average=args.weighted_average)
        evaluation = RedirectModel(evaluation, prediction_model)
        callbacks.append(evaluation)

    if args.snapshots:
        makedirs(args.snapshot_path)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                args.snapshot_path,
                '{backbone}_{dataset_type}_{{epoch:02d}}.h5'.format(backbone=args.backbone, dataset_type=args.dataset_type)
            ),
            verbose=1,

        )
        checkpoint = RedirectModel(checkpoint, model)
        callbacks.append(checkpoint)

    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor    = 'loss',
        factor     = 0.1,
        patience   = 2,
        verbose    = 1,
        mode       = 'auto',
        min_delta  = 0.0001,
        cooldown   = 0,
        min_lr     = 0
    ))

    return callbacks


def create_generators(args, preprocess_image):

    common_args = {
        'batch_size'       : args.batch_size,
        'config'           : args.config,
        'image_min_side'   : args.image_min_side,
        'image_max_side'   : args.image_max_side,
        'preprocess_image' : preprocess_image,
    }

    args.random_transform = False
    if args.random_transform:
        transform_generator = random_transform_generator(
            min_rotation=-0.1,
            max_rotation=0.1,
            min_translation=(-0.1, -0.1),
            max_translation=(0.1, 0.1),
            min_shear=-0.1,
            max_shear=0.1,
            min_scaling=(0.9, 0.9),
            max_scaling=(1.1, 1.1),
            flip_x_chance=0.0,  # flipping image might lead to inconsistencies in regression (test that at some point)
            flip_y_chance=0.0,  # flipping image might lead to inconsistencies in regression (test that at some point)
        )
    else:
        #transform_generator = random_transform_generator(flip_x_chance=0.5)
        transform_generator = random_transform_generator(flip_x_chance=0.0)

    if args.dataset_type == 'linemod':
        # import here to prevent unnecessary dependency on cocoapi
        from ..preprocessing.linemod import LinemodGenerator

        train_generator = LinemodGenerator(
            args.linemod_path,
            'train',
            transform_generator=transform_generator,
            **common_args
        )

        validation_generator = LinemodGenerator(
            args.linemod_path,
            'val',
            **common_args
        )
        train_iterations = len(os.listdir(os.path.join(args.linemod_path, 'images/train')))
    elif args.dataset_type == 'tless':
        # import here to prevent unnecessary dependency on cocoapi
        from ..preprocessing.tless import TlessGenerator

        train_generator = TlessGenerator(
            args.tless_path,
            'train',
            transform_generator=transform_generator,
            **common_args
        )

        validation_generator = TlessGenerator(
            args.tless_path,
            'test',
            **common_args
        )
        train_iterations = len(os.listdir(os.path.join(args.tless_path, 'images/train')))
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return train_generator, validation_generator, train_iterations


def parse_args(args):
    parser = argparse.ArgumentParser(description='RetinaNet with pose estimation.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    linemod_parser = subparsers.add_parser('linemod')
    linemod_parser.add_argument('linemod_path', help='Path to dataset directory (ie. /tmp/LineMOD).')

    tless_parser = subparsers.add_parser('tless')
    tless_parser.add_argument('tless_path', help='Path to dataset directory (ie. /tmp/Tless).')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--snapshot',          help='Resume training from a snapshot.')
    group.add_argument('--imagenet-weights',  help='Initialize the model with pretrained imagenet weights. This is the default behaviour.', action='store_const', const=True, default=True)
    group.add_argument('--weights',           help='Initialize the model with weights from a file.')
    group.add_argument('--no-weights',        help='Don\'t initialize the model with any weights.', dest='imagenet_weights', action='store_const', const=False)

    parser.add_argument('--backbone',         help='Backbone model used by retinanet.', default='resnet50', type=str)
    parser.add_argument('--batch-size',       help='Size of the batches.', default=1, type=int)
    parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).')
    #parser.add_argument('--multi-gpu',        help='Number of GPUs to use for parallel processing.', type=int, default=0)
    #parser.add_argument('--multi-gpu-force',  help='Extra flag needed to enable (experimental) multi-gpu support.', action='store_true')
    parser.add_argument('--epochs',           help='Number of epochs to train.', type=int, default=15)
    #parser.add_argument('--steps',            help='Number of steps per epoch.', type=int, default=10000)
    parser.add_argument('--snapshot-path',    help='Path to store snapshots of models during training (defaults to \'./data\')', default='./data')
    parser.add_argument('--tensorboard-dir',  help='Log directory for Tensorboard output', default='./logs')
    parser.add_argument('--no-snapshots',     help='Disable saving snapshots.', dest='snapshots', action='store_false')
    parser.add_argument('--no-evaluation',    help='Disable per epoch evaluation.', dest='evaluation', action='store_true')
    parser.add_argument('--freeze-backbone',  help='Freeze training of backbone layers.', action='store_true')
    parser.add_argument('--random-transform', help='Randomly transform image and annotations.', action='store_true')
    parser.add_argument('--image-min-side',   help='Rescale the image so the smallest side is min_side.', type=int, default=480)
    parser.add_argument('--image-max-side',   help='Rescale the image if the largest side is larger than max_side.', type=int, default=640)
    parser.add_argument('--config',           help='Path to a configuration parameters .ini file.')
    parser.add_argument('--weighted-average', help='Compute the mAP using the weighted average of precisions among classes.', action='store_true')

    return parser.parse_args(args)


def main(args=None):

    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    backbone = models.backbone(args.backbone)

    check_keras_version()

    if args.gpu:
        print('specified GPU for training: ', args.gpu)
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    if args.config:
        args.config = read_config_file(args.config)

    train_generator, validation_generator, train_iterations = create_generators(args, backbone.preprocess_image)

    if args.snapshot is not None:
        print('Loading model, this may take a second...')
        model = models.load_model(args.snapshot, backbone_name=args.backbone)
        training_model = model
        anchor_params = None
        if args.config and 'anchor_parameters' in args.config:
            anchor_params = parse_anchor_parameters(args.config)
        prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params)
    else:
        weights = args.weights

        if weights is None and args.imagenet_weights:
            weights = backbone.download_imagenet()

        print('Creating model, this may take a second...')
        model, training_model, prediction_model = create_models(
            backbone_retinanet=backbone.retinanet,
            num_classes=train_generator.num_classes(),
            weights=weights,
            multi_gpu=0,
            freeze_backbone=args.freeze_backbone,
            config=args.config
        )

    #print(model.summary())

    callbacks = create_callbacks(
        model,
        training_model,
        prediction_model,
        validation_generator,
        args,
    )

    # start training
    training_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=train_iterations,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
        use_multiprocessing=True,
        workers=4
    )


if __name__ == '__main__':
    main()
