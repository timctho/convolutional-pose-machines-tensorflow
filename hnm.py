import tensorflow as tf
import numpy as np
from utils import cpm_utils
import cv2
import time
import math
import importlib
import sys
import os
import json
from Tim_utils import utils
import Ensemble_data_generator


from config import FLAGS
cpm_model = importlib.import_module('models.nets.'+FLAGS.network_def)

if sys.version_info.major == 3:
    PYTHON_VERSION = 3
else:
    PYTHON_VERSION = 2


def scale_square_data(img, points, box_size):
    # Resize and pad image to fit output image size
    output_image = np.ones(shape=(box_size, box_size, 3)) * 128.0
    img_h = img.shape[0]
    img_w = img.shape[1]
    if img_h > img_w:
        scale = box_size / (img_h * 1.0)

        # Relocalize points
        points[:, 0] *= scale
        points[:, 1] *= scale

        # Resize image
        image = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
        resized_img_h, resized_img_w = image.shape[0], image.shape[1]

        offset = resized_img_w % 2

        output_image[:, int(box_size / 2 - math.floor(resized_img_w / 2)): int(
            box_size / 2 + math.floor(resized_img_w / 2) + offset), :] = image
        points[:, 0] += (box_size / 2 - math.floor(resized_img_w / 2))

    else:
        scale = box_size / (img_w * 1.0)

        # Relocalize points
        points[:, 0] *= scale
        points[:, 1] *= scale

        # Resize image
        image = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
        resized_img_h, resized_img_w = image.shape[0], image.shape[1]

        offset = resized_img_h % 2

        output_image[int(box_size / 2 - math.floor(resized_img_h / 2)): int(
            box_size / 2 + math.floor(resized_img_h / 2) + offset), :, :] = image
        points[:, 1] += (box_size / 2 - math.floor(resized_img_h / 2))

    return output_image, points


def main():
    model = cpm_model.CPM_Model(input_size=FLAGS.input_size,
                                heatmap_size=FLAGS.heatmap_size,
                                stages=FLAGS.cpm_stages,
                                joints=FLAGS.num_of_joints,
                                img_type=FLAGS.color_channel,
                                is_training=False)
    model.build_loss(FLAGS.init_lr, FLAGS.lr_decay_rate, FLAGS.lr_decay_step, optimizer='RMSProp')
    saver = tf.train.Saver()

    g = Ensemble_data_generator.ensemble_data_generator(FLAGS.train_img_dir,
                                                        None,
                                                        FLAGS.batch_size, FLAGS.input_size, True, False,
                                                        FLAGS.augmentation_config, False)

    device_count = {'GPU': 1} if FLAGS.use_gpu else {'GPU': 0}
    sess_config = tf.ConfigProto(device_count=device_count)
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True
    with tf.Session(config=sess_config) as sess:

        model_path_suffix = os.path.join(FLAGS.network_def,
                                         'input_{}_output_{}'.format(FLAGS.input_size, FLAGS.heatmap_size),
                                         'joints_{}'.format(FLAGS.num_of_joints),
                                         'stages_{}'.format(FLAGS.cpm_stages),
                                         'init_{}_rate_{}_step_{}'.format(FLAGS.init_lr, FLAGS.lr_decay_rate,
                                                                          FLAGS.lr_decay_step)
                                         )
        model_save_dir = os.path.join('models',
                                      'weights',
                                      model_path_suffix)
        print('Load model from [{}]'.format(os.path.join(model_save_dir, FLAGS.model_path)))
        if FLAGS.model_path.endswith('pkl'):
            model.load_weights_from_file(FLAGS.model_path, sess, False)
        else:
            saver.restore(sess, os.path.join(model_save_dir, FLAGS.model_path))
        print('Load model done')


        bbox_offset = 100
        for person_dir in os.listdir(FLAGS.train_img_dir):
            json_file_path = os.path.join(FLAGS.train_img_dir, person_dir, 'attr_data.json')
            hnm_json_list = [[] for _ in range(11)]

            with open(json_file_path, 'r') as f:
                json_file = json.load(f)

            loss_cnt = 0
            img_cnt = 0
            hnm_cnt = 0
            for cam_id in range(11):
                for img_id in range(len(json_file[cam_id])):
                    img_path = os.path.join(FLAGS.train_img_dir,
                                            person_dir,
                                            'undistorted_img',
                                            json_file[cam_id][img_id]['name'])
                    img = cv2.imread(img_path)

                    # Read joints
                    hand_2d_joints = np.zeros(shape=(21, 2))
                    bbox = json_file[cam_id][img_id]['bbox']
                    bbox[0] = max(bbox[0] - bbox_offset, 0)
                    bbox[1] = max(bbox[1] - bbox_offset, 0)
                    bbox[2] = min(bbox[2] + bbox_offset, img.shape[0])
                    bbox[3] = min(bbox[3] + bbox_offset, img.shape[1])
                    img = img[bbox[1]:bbox[3],
                          bbox[0]:bbox[2]]

                    for i, finger_name in enumerate(['thumb', 'index', 'middle', 'ring', 'pinky']):
                        for j, joint_name in enumerate(['tip', 'dip', 'pip', 'mcp']):
                            hand_2d_joints[i * 4 + j, :] = \
                            json_file[cam_id][img_id][finger_name][joint_name]['pose2']
                    hand_2d_joints[20, :] = json_file[cam_id][img_id]['wrist']['pose2']
                    hand_2d_joints[:, 0] -= bbox[0]
                    hand_2d_joints[:, 1] -= bbox[1]

                    # for i in range(hand_2d_joints.shape[0]):
                    #     cv2.circle(img, (int(hand_2d_joints[i][0]), int(hand_2d_joints[i][1])), 5, (0, 255, 0), -1)
                    # print(img_path)
                    img = img / 255.0 - 0.5

                    img, hand_2d_joints = scale_square_data(img, hand_2d_joints, FLAGS.input_size)
                    # for i in range(hand_2d_joints.shape[0]):
                    #     cv2.circle(img, (int(hand_2d_joints[i][0]), int(hand_2d_joints[i][1])), 5, (0, 255, 0), -1)
                    # cv2.imshow('', img)
                    # cv2.waitKey(0)

                    img = np.expand_dims(img, axis=0)
                    hand_2d_joints = np.expand_dims(hand_2d_joints, axis=0)

                    gt_heatmap_np = cpm_utils.make_heatmaps_from_joints(FLAGS.input_size,
                                                                              FLAGS.heatmap_size,
                                                                              FLAGS.joint_gaussian_variance,
                                                                              hand_2d_joints)


                    loss, = sess.run([model.total_loss], feed_dict={model.input_images: img,
                                                                    model.gt_hmap_placeholder: gt_heatmap_np})

                    # loss_cnt += loss
                    img_cnt += 1
                    # print(img_path, float(loss_cnt)/ img_cnt)

                    if loss > 150.0:
                        hnm_json_list[cam_id].append(json_file[cam_id][img_id])
                        hnm_cnt += 1
                        print('hnm cnt {} / {}'.format(hnm_cnt, img_cnt))

            with open(os.path.join(FLAGS.train_img_dir, person_dir, 'attr_data_hnm.json'), 'wb') as f:
                json.dump(hnm_json_list, f)
                print('write done with {}'.format(person_dir))


if __name__ == '__main__':
    main()
