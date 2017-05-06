#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import time
import sys
import helper
import vgg16


args = helper.get_args(description='Extract VGG16 features')
helper.setup(args)

batch_size = args.batch_size

with tf.Session() as sess:
    vgg = vgg16.Vgg16()

    images = tf.placeholder("float", [None, 224, 224, 3])
    vgg.build(images)

    LAYERS_TO_EXTRACT = helper.get_vgg_layers_to_be_extracted(vgg, args.extract_layers)

    for img_paths, imgs in helper.next_img_batch(
            count=batch_size,
            done_file=args.done_file,
            images_file=args.images_list_file,
            prepend_image_path=args.images_path
        ):

        time_start = time.time()
        features = sess.run(LAYERS_TO_EXTRACT, feed_dict={images: imgs})
        helper.save_features(img_paths, features, features_file=args.features_file)

        seconds_passed = (time.time() - time_start)
        print('{:.3f} seconds/image'.format(seconds_passed / batch_size))
        sys.stdout.flush()
