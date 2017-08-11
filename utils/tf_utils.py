import tensorflow as tf
import utils.cpm_utils as cpm_utils


def read_and_decode_cpm(tfr_queue, img_size, num_joints, center_radius):
    tfr_reader = tf.TFRecordReader()
    _, serialized_example = tfr_reader.read(tfr_queue)

    queue_images = []
    queue_center_maps = []
    queue_labels = []
    queue_orig_images = []

    for i in range(2):
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'image': tf.FixedLenFeature([], tf.string),
                                               'heatmaps': tf.FixedLenFeature(
                                                   [int(img_size * img_size * (num_joints + 1))], tf.float32)
                                           })

        # img_size = 128
        # center_radius = 11
        img = tf.decode_raw(features['image'], tf.uint8)
        img = tf.reshape(img, [img_size, img_size, 3])
        img = tf.cast(img, tf.float32)

        img = img[..., ::-1]
        img = tf.image.random_contrast(img, 0.7, 1)
        img = tf.image.random_brightness(img, max_delta=0.9)
        img = tf.image.random_hue(img, 0.05)
        img = tf.image.random_saturation(img, 0.7, 1.1)
        img = img[..., ::-1]

        # heatmap = tf.decode_raw(features['heatmaps'], tf.float32)
        heatmap = tf.reshape(features['heatmaps'], [img_size, img_size, (num_joints + 1)])

        # create centermap
        center_map = tf.constant((cpm_utils.make_gaussian(img_size, center_radius,
                                                          [int(img_size / 2), int(img_size / 2)])).reshape(
            (img_size, img_size, 1)), name='center_map')
        center_map = tf.cast(center_map, tf.float32)

        # merge img + centermap + heatmap
        merged_img_heatmap = tf.concat([img, center_map, heatmap], axis=2)

        # subtract mean before pad
        mean_volume = tf.concat((128 * tf.ones(shape=(img_size, img_size, 3)),
                                 tf.zeros(shape=(img_size, img_size, (num_joints + 1))),
                                 tf.ones(shape=(img_size, img_size, 1))), axis=2)

        merged_img_heatmap -= mean_volume

        # preprocessing
        preprocessed_merged_img_c_heatmap, _, _ = preprocess(merged_img_heatmap,
                                                             label=None,
                                                             crop_off_ratio=0.05,
                                                             rotation_angle=0.8,
                                                             has_bbox=False,
                                                             do_flip_lr=True,
                                                             do_flip_ud=False,
                                                             low_sat=None,
                                                             high_sat=None,
                                                             max_bright_delta=None,
                                                             max_hue_delta=None)

        padded_img_size = img_size  # * (1 + tf.random_uniform([], minval=0.0, maxval=0.3))
        padded_img_size = tf.cast(padded_img_size, tf.int32)

        # resize pad
        preprocessed_merged_img_c_heatmap = tf.image.resize_image_with_crop_or_pad(preprocessed_merged_img_c_heatmap,
                                                                                   padded_img_size, padded_img_size)
        preprocessed_merged_img_c_heatmap += tf.concat((128 * tf.ones(shape=(padded_img_size, padded_img_size, 3)),
                                                        tf.zeros(
                                                            shape=(padded_img_size, padded_img_size, (num_joints + 1))),
                                                        tf.ones(shape=(padded_img_size, padded_img_size, 1))), axis=2)
        preprocessed_merged_img_c_heatmap = tf.image.resize_images(preprocessed_merged_img_c_heatmap,
                                                                   size=[img_size, img_size])

        with tf.control_dependencies([preprocessed_merged_img_c_heatmap]):
            # preprocessed_img = tf.slice(preprocessed_merged_img_c_heatmap, [0,0,0], [368,368,3])
            # preprocessed_center_maps = tf.slice(preprocessed_merged_img_c_heatmap, [0,0,3], [368,368,1])
            # preprocessed_heatmaps = tf.slice(preprocessed_merged_img_c_heatmap, [0,0,4], [368,368,13])

            preprocessed_img, preprocessed_center_maps, preprocessed_heatmaps = tf.split(
                preprocessed_merged_img_c_heatmap, [3, 1, (num_joints + 1)], axis=2)

            # Normalize image value
            preprocessed_img /= 256
            preprocessed_img -= 0.5

            queue_images.append(preprocessed_img)
            queue_center_maps.append(preprocessed_center_maps)
            queue_labels.append(preprocessed_heatmaps)
            queue_orig_images.append(img)

    return queue_images, queue_center_maps, queue_labels, queue_orig_images
    # return preprocessed_img, preprocessed_center_maps, preprocessed_heatmaps, img


def read_batch_cpm(tfr_path, img_size, hmap_size, num_joints, center_radius, batch_size=16, num_epochs=None):
    """Read batch images as the input to the network

        tfr_path: path to tfrecord file
        num_epochs: None=iteratively read forever
                    other number=iterate whole tfr_file how many times
        """

    with tf.name_scope('Batch_Inputs'):
        tfr_queue = tf.train.string_input_producer(tfr_path, num_epochs=num_epochs, shuffle=True)

        # images, centers, labels, image_orig = read_and_decode_cpm(tfr_queue, img_size, num_joints, center_radius)

        data_list = [read_and_decode_cpm(tfr_queue, img_size, num_joints, center_radius) for _ in
                     range(2 * len(tfr_path))]

        batch_images, batch_centers, batch_labels, batch_images_orig = tf.train.shuffle_batch_join(data_list,
                                                                                                   batch_size=batch_size,
                                                                                                   capacity=100 + 6 * batch_size,
                                                                                                   min_after_dequeue=100,
                                                                                                   enqueue_many=True,
                                                                                                   name='batch_data_read')

        # batch_labels = tf.image.resize_bilinear(batch_labels, size=tf.constant((hmap_size,hmap_size), name='shape'))

    return batch_images, batch_centers, batch_labels, batch_images_orig


def rotate_points(orig_points, angle, w, h):
    """Return rotated points

    Args:
        orig_points: 'Tensor' with shape [N,2], each entry is point (x,y)
        angle: rotate radians

    Returns:
        'Tensor' with shape [N,2], with rotated points
    """

    # rotation
    rotate_mat = tf.stack([[tf.cos(angle) / w, tf.sin(angle) / h],
                           [-tf.sin(angle) / w, tf.cos(angle) / h]])

    # shift coord
    orig_points = tf.subtract(orig_points, 0.5)

    orig_points = tf.stack([orig_points[:, 0] * w,
                            orig_points[:, 1] * h], axis=1)
    print(orig_points)
    rotated_points = tf.matmul(orig_points, rotate_mat) + 0.5

    return rotated_points


def preprocess(image,
               label,
               has_bbox=True,
               rotation_angle=1.5,
               crop_off_ratio=0.2,
               do_flip_lr=True,
               do_flip_ud=True,
               max_hue_delta=0.15,
               low_sat=0.5,
               high_sat=2.0,
               max_bright_delta=0.3):
    """Do some processes for input image

    Args:
        image: A 'Tensor' of RGB image
        label: vector of floats with even length (be pair of (x,y))
        has_bbox: if 'True', Assume first 4 numbers of 'label' are [top-left, bot-right] coords
        rotation_angle: maximum allowed rotation radians
        crop_off_ratio: maximum cropping offset of top-left corner
                        1-crop_off_ratio be maximum cropping offset of cropped bot-right corner
        do_flip_lr: with half chance flip the image left right
        do_flip_ud: with half chance flip the image upper down
        max_hue_delta: allowed random adjust hue range
        low_sat: lowest range of saturation
        high_sat: highest range of saturation
        max_bright_delta: allowed random adjust brightness range

    Returns:
        image: processed image 'Tensor'
        new_bbox: 'Tensor' of processed bbox coords if 'has_bbox' == True
        total_points: 'Tensor' of processed points coords
    """

    new_bbox = []
    total_points = []

    # [height, width, channel] of input image
    img_shape_list = image.get_shape().as_list()

    if max_hue_delta is not None:
        # random hue
        image = tf.image.random_hue(image, max_delta=max_hue_delta)

    if low_sat is not None and high_sat is not None:
        # random saturation
        image = tf.image.random_saturation(image, lower=low_sat, upper=high_sat)

    if max_bright_delta is not None:
        # random brightness
        image = tf.image.random_brightness(image, max_delta=max_bright_delta)

    if label is not None:
        total_points = tf.stack([label[i] for i in range(label.shape[0])])

    # crop image
    new_top_left_x = crop_off_ratio * tf.random_uniform([], minval=-1.0, maxval=1.0)
    off_w_ratio = tf.cond(tf.less(new_top_left_x, 0), lambda: tf.zeros([]), lambda: new_top_left_x)

    new_top_left_y = crop_off_ratio * tf.random_uniform([], minval=-1.0, maxval=1.0)
    off_h_ratio = tf.cond(tf.less(new_top_left_y, 0), lambda: tf.zeros([]), lambda: new_top_left_y)

    new_bot_right_x = crop_off_ratio * tf.random_uniform([], minval=-1.0, maxval=1.0)
    tar_w_ratio = tf.cond(tf.less(new_bot_right_x, 0), lambda: tf.ones([]) - off_w_ratio,
                          lambda: 1 - new_bot_right_x - off_w_ratio)

    new_bot_right_y = crop_off_ratio * tf.random_uniform([], minval=-1.0, maxval=1.0)
    tar_h_ratio = tf.cond(tf.less(new_bot_right_y, 0), lambda: tf.ones([]) - off_h_ratio,
                          lambda: 1 - new_bot_right_y - off_h_ratio)

    pad_image_height = (1 - new_top_left_y - new_bot_right_y) * img_shape_list[0]
    pad_image_width = (1 - new_top_left_x - new_bot_right_x) * img_shape_list[1]
    cropped_image = tf.image.crop_to_bounding_box(image,
                                                  offset_width=tf.cast(off_w_ratio * img_shape_list[1], tf.int32),
                                                  offset_height=tf.cast(off_h_ratio * img_shape_list[0], tf.int32),
                                                  target_height=tf.cast(tar_h_ratio * img_shape_list[0], tf.int32),
                                                  target_width=tf.cast(tar_w_ratio * img_shape_list[1], tf.int32))

    image = tf.image.pad_to_bounding_box(cropped_image,
                                         offset_width=tf.cast((off_w_ratio - new_top_left_x) * img_shape_list[1],
                                                              tf.int32),
                                         offset_height=tf.cast((off_h_ratio - new_top_left_y) * img_shape_list[0],
                                                               tf.int32),
                                         target_height=tf.cast(pad_image_height, tf.int32),
                                         target_width=tf.cast(pad_image_width, tf.int32))

    # random rotation angle
    angle = rotation_angle * tf.random_uniform([])

    # rotate image
    image = tf.contrib.image.rotate(image, -angle, interpolation='BILINEAR')

    if label is not None:
        if has_bbox:
            # include 4 bbox points
            bbox_points = tf.stack([[total_points[0][0], total_points[0][1]],
                                    [total_points[1][0], total_points[0][1]],
                                    [total_points[0][0], total_points[1][1]],
                                    [total_points[1][0], total_points[1][1]]], axis=0)
            if label.shape[0] == 4:
                total_points = bbox_points
            else:
                total_points = tf.concat([bbox_points, total_points[2:]], axis=0)

        # rotate points
        total_points = rotate_points(total_points, angle, pad_image_width, pad_image_height)

        if has_bbox:
            # new bbox [top_left, bot_right]
            new_bbox = tf.stack([[total_points[2][0], total_points[0][1]],
                                 [total_points[1][0], total_points[3][1]]], axis=0)
            total_points = tf.concat([new_bbox, total_points[4:]], axis=0)

    if label is not None:
        # adjust points' coords for cropped image
        total_points = tf.reshape(total_points[:], shape=[-1, 2])
        total_points = tf.stack([(total_points[:, 0] - new_top_left_x) / (1 - new_top_left_x - new_bot_right_x),
                                 (total_points[:, 1] - new_top_left_y) / (1 - new_top_left_y - new_bot_right_y)],
                                axis=1)

    if label is not None:
        # chance flip left right
        def flip_lr():
            i = tf.image.flip_left_right(image)
            l = tf.stack([1 - total_points[:, 0],
                          total_points[:, 1]], axis=1)
            return i, l

        def no_flip_lr():
            i = image
            l = total_points
            return i, l

        if do_flip_lr:
            image, total_points = tf.cond(tf.greater(tf.random_uniform([]), 0.5), flip_lr, no_flip_lr)

        # chance flip upside down
        def flip_ud():
            i = tf.image.flip_up_down(image)
            l = tf.stack([total_points[:, 0],
                          1 - total_points[:, 1]], axis=1)
            return i, l

        def no_flip_ud():
            i = image
            l = total_points
            return i, l

        if do_flip_ud:
            image, total_points = tf.cond(tf.greater(tf.random_uniform([]), 0.5), flip_ud, no_flip_ud)

        if has_bbox:
            new_bbox = tf.stack([(total_points[0, 0] + total_points[1, 0]) / 2,
                                 (total_points[0, 1] + total_points[1, 1]) / 2,
                                 tf.abs(total_points[1, 0] - total_points[0, 0]),
                                 tf.abs(total_points[1, 1] - total_points[0, 1])], axis=0)

        total_points = tf.reshape(total_points, shape=[-1, ])

    else:
        # chance flip left right
        def flip_lr():
            i = tf.image.flip_left_right(image)
            return i

        def no_flip_lr():
            i = image
            return i

        if do_flip_lr:
            image = tf.cond(tf.greater(tf.random_uniform([]), 0.5), flip_lr, no_flip_lr)

        # chance flip upside down
        def flip_ud():
            i = tf.image.flip_up_down(image)
            return i

        def no_flip_ud():
            i = image
            return i

        if do_flip_ud:
            image = tf.cond(tf.greater(tf.random_uniform([]), 0.5), flip_ud, no_flip_ud)

    return image, new_bbox, total_points
