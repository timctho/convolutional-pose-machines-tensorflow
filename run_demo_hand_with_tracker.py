import tensorflow as tf
import numpy as np
from utils import cpm_utils, tracking_module, utils
import cv2
import time
import math
import importlib
import os

from config import FLAGS

cpm_model = importlib.import_module('models.nets.' + FLAGS.network_def)

joint_detections = np.zeros(shape=(21, 2))


def main(argv):
    global joint_detections
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu_id)

    """ Initial tracker
    """
    tracker = tracking_module.SelfTracker([FLAGS.webcam_height, FLAGS.webcam_width], FLAGS.input_size)

    """ Build network graph
    """
    model = cpm_model.CPM_Model(input_size=FLAGS.input_size,
                                heatmap_size=FLAGS.heatmap_size,
                                stages=FLAGS.cpm_stages,
                                joints=FLAGS.num_of_joints,
                                img_type=FLAGS.color_channel,
                                is_training=False)
    saver = tf.train.Saver()

    """ Get output node
    """
    output_node = tf.get_default_graph().get_tensor_by_name(name=FLAGS.output_node_names)

    device_count = {'GPU': 1} if FLAGS.use_gpu else {'GPU': 0}
    sess_config = tf.ConfigProto(device_count=device_count)
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.2
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
            saver.restore(sess, 'models/weights/cpm_hand')

        # Check weights
        for variable in tf.global_variables():
            with tf.variable_scope('', reuse=True):
                var = tf.get_variable(variable.name.split(':0')[0])
                print(variable.name, np.mean(sess.run(var)))

        # Create webcam instance
        if FLAGS.DEMO_TYPE in ['MULTI', 'SINGLE', 'Joint_HM']:
            cam = cv2.VideoCapture(FLAGS.cam_id)

        # Create kalman filters
        if FLAGS.use_kalman:
            kalman_filter_array = [cv2.KalmanFilter(4, 2) for _ in range(FLAGS.num_of_joints)]
            for _, joint_kalman_filter in enumerate(kalman_filter_array):
                joint_kalman_filter.transitionMatrix = np.array(
                    [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
                    np.float32)
                joint_kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
                joint_kalman_filter.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                                               np.float32) * FLAGS.kalman_noise
        else:
            kalman_filter_array = None

        if FLAGS.DEMO_TYPE.endswith(('png', 'jpg')):
            test_img = cpm_utils.read_image(FLAGS.DEMO_TYPE, [], FLAGS.input_size, 'IMAGE')
            test_img_resize = cv2.resize(test_img, (FLAGS.input_size, FLAGS.input_size))

            test_img_input = normalize_and_centralize_img(test_img_resize)

            t1 = time.time()
            predict_heatmap, stage_heatmap_np = sess.run([model.current_heatmap,
                                                          output_node,
                                                          ],
                                                         feed_dict={model.input_images: test_img_input}
                                                         )
            print('fps: %.2f' % (1 / (time.time() - t1)))

            correct_and_draw_hand(test_img,
                                  cv2.resize(stage_heatmap_np[0], (FLAGS.input_size, FLAGS.input_size)),
                                  kalman_filter_array, tracker, tracker.input_crop_ratio, test_img)

            # Show visualized image
            # demo_img = visualize_result(test_img, stage_heatmap_np, kalman_filter_array)
            cv2.imshow('demo_img', test_img.astype(np.uint8))
            cv2.waitKey(0)

        elif FLAGS.DEMO_TYPE in ['SINGLE', 'MULTI']:
            while True:
                # # Prepare input image
                _, full_img = cam.read()

                test_img = tracker.tracking_by_joints(full_img, joint_detections=joint_detections)
                crop_full_scale = tracker.input_crop_ratio
                test_img_copy = test_img.copy()

                # White balance
                test_img_wb = utils.img_white_balance(test_img, 5)
                test_img_input = normalize_and_centralize_img(test_img_wb)

                # Inference
                t1 = time.time()
                stage_heatmap_np = sess.run([output_node],
                                            feed_dict={model.input_images: test_img_input})
                print('FPS: %.2f' % (1 / (time.time() - t1)))

                local_img = visualize_result(full_img, stage_heatmap_np, kalman_filter_array, tracker, crop_full_scale,
                                             test_img_copy)

                cv2.imshow('local_img', local_img.astype(np.uint8))
                cv2.imshow('global_img', full_img.astype(np.uint8))
                if cv2.waitKey(1) == ord('q'): break


        elif FLAGS.DEMO_TYPE == 'Joint_HM':
            while True:
                # Prepare input image
                test_img = cpm_utils.read_image([], cam, FLAGS.input_size, 'WEBCAM')
                test_img_resize = cv2.resize(test_img, (FLAGS.input_size, FLAGS.input_size))

                test_img_input = normalize_and_centralize_img(test_img_resize)

                # Inference
                t1 = time.time()
                stage_heatmap_np = sess.run([output_node],
                                            feed_dict={model.input_images: test_img_input})
                print('FPS: %.2f' % (1 / (time.time() - t1)))

                demo_stage_heatmap = stage_heatmap_np[len(stage_heatmap_np) - 1][0, :, :,
                                     0:FLAGS.num_of_joints].reshape(
                    (FLAGS.heatmap_size, FLAGS.heatmap_size, FLAGS.num_of_joints))
                demo_stage_heatmap = cv2.resize(demo_stage_heatmap, (FLAGS.input_size, FLAGS.input_size))

                vertical_imgs = []
                tmp_img = None
                joint_coord_set = np.zeros((FLAGS.num_of_joints, 2))

                for joint_num in range(FLAGS.num_of_joints):
                    # Concat until 4 img
                    if (joint_num % 4) == 0 and joint_num != 0:
                        vertical_imgs.append(tmp_img)
                        tmp_img = None

                    demo_stage_heatmap[:, :, joint_num] *= (255 / np.max(demo_stage_heatmap[:, :, joint_num]))

                    # Plot color joints
                    if np.min(demo_stage_heatmap[:, :, joint_num]) > -50:
                        joint_coord = np.unravel_index(np.argmax(demo_stage_heatmap[:, :, joint_num]),
                                                       (FLAGS.input_size, FLAGS.input_size))
                        joint_coord_set[joint_num, :] = joint_coord
                        color_code_num = (joint_num // 4)

                        if joint_num in [0, 4, 8, 12, 16]:
                            joint_color = list(
                                map(lambda x: x + 35 * (joint_num % 4), FLAGS.joint_color_code[color_code_num]))
                            cv2.circle(test_img, center=(joint_coord[1], joint_coord[0]), radius=3, color=joint_color,
                                       thickness=-1)
                        else:
                            joint_color = list(
                                map(lambda x: x + 35 * (joint_num % 4), FLAGS.joint_color_code[color_code_num]))
                            cv2.circle(test_img, center=(joint_coord[1], joint_coord[0]), radius=3, color=joint_color,
                                       thickness=-1)

                    # Put text
                    tmp = demo_stage_heatmap[:, :, joint_num].astype(np.uint8)
                    tmp = cv2.putText(tmp, 'Min:' + str(np.min(demo_stage_heatmap[:, :, joint_num])),
                                      org=(5, 20), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.3, color=150)
                    tmp = cv2.putText(tmp, 'Mean:' + str(np.mean(demo_stage_heatmap[:, :, joint_num])),
                                      org=(5, 30), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.3, color=150)
                    tmp_img = np.concatenate((tmp_img, tmp), axis=0) \
                        if tmp_img is not None else tmp

                # Plot FLAGS.limbs
                for limb_num in range(len(FLAGS.limbs)):
                    if np.min(demo_stage_heatmap[:, :, FLAGS.limbs[limb_num][0]]) > -2000 and np.min(
                            demo_stage_heatmap[:, :, FLAGS.limbs[limb_num][1]]) > -2000:
                        x1 = joint_coord_set[FLAGS.limbs[limb_num][0], 0]
                        y1 = joint_coord_set[FLAGS.limbs[limb_num][0], 1]
                        x2 = joint_coord_set[FLAGS.limbs[limb_num][1], 0]
                        y2 = joint_coord_set[FLAGS.limbs[limb_num][1], 1]
                        length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                        if length < 10000 and length > 5:
                            deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
                            polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
                                                       (int(length / 2), 3),
                                                       int(deg),
                                                       0, 360, 1)
                            color_code_num = limb_num // 4
                            limb_color = list(
                                map(lambda x: x + 35 * (limb_num % 4), FLAGS.joint_color_code[color_code_num]))

                            cv2.fillConvexPoly(test_img, polygon, color=limb_color)

                if tmp_img is not None:
                    tmp_img = np.lib.pad(tmp_img, ((0, vertical_imgs[0].shape[0] - tmp_img.shape[0]), (0, 0)),
                                         'constant', constant_values=(0, 0))
                    vertical_imgs.append(tmp_img)

                # Concat horizontally
                output_img = None
                for col in range(len(vertical_imgs)):
                    output_img = np.concatenate((output_img, vertical_imgs[col]), axis=1) if output_img is not None else \
                        vertical_imgs[col]

                output_img = output_img.astype(np.uint8)
                output_img = cv2.applyColorMap(output_img, cv2.COLORMAP_JET)
                test_img = cv2.resize(test_img, (300, 300), cv2.INTER_LANCZOS4)
                cv2.imshow('hm', output_img)
                cv2.moveWindow('hm', 2000, 200)
                cv2.imshow('rgb', test_img)
                cv2.moveWindow('rgb', 2000, 750)
                if cv2.waitKey(1) == ord('q'): break


def normalize_and_centralize_img(img):
    if FLAGS.color_channel == 'GRAY':
        img = np.dot(img[..., :3], [0.299, 0.587, 0.114]).reshape((FLAGS.input_size, FLAGS.input_size, 1))

    if FLAGS.normalize_img:
        test_img_input = img / 256.0 - 0.5
        test_img_input = np.expand_dims(test_img_input, axis=0)
    else:
        test_img_input = img - 128.0
        test_img_input = np.expand_dims(test_img_input, axis=0)
    return test_img_input


def visualize_result(test_img, stage_heatmap_np, kalman_filter_array, tracker, crop_full_scale, crop_img):
    demo_stage_heatmaps = []
    if FLAGS.DEMO_TYPE == 'MULTI':
        for stage in range(len(stage_heatmap_np)):
            demo_stage_heatmap = stage_heatmap_np[stage][0, :, :, 0:FLAGS.num_of_joints].reshape(
                (FLAGS.heatmap_size, FLAGS.heatmap_size, FLAGS.num_of_joints))
            demo_stage_heatmap = cv2.resize(demo_stage_heatmap, (FLAGS.input_size, FLAGS.input_size))
            demo_stage_heatmap = np.amax(demo_stage_heatmap, axis=2)
            demo_stage_heatmap = np.reshape(demo_stage_heatmap, (FLAGS.input_size, FLAGS.input_size, 1))
            demo_stage_heatmap = np.repeat(demo_stage_heatmap, 3, axis=2)
            demo_stage_heatmap *= 255
            demo_stage_heatmaps.append(demo_stage_heatmap)

        last_heatmap = stage_heatmap_np[len(stage_heatmap_np) - 1][0, :, :, 0:FLAGS.num_of_joints].reshape(
            (FLAGS.heatmap_size, FLAGS.heatmap_size, FLAGS.num_of_joints))
        last_heatmap = cv2.resize(last_heatmap, (FLAGS.input_size, FLAGS.input_size))
    else:
        last_heatmap = stage_heatmap_np[len(stage_heatmap_np) - 1][0, :, :, 0:FLAGS.num_of_joints].reshape(
            (FLAGS.heatmap_size, FLAGS.heatmap_size, FLAGS.num_of_joints))
        last_heatmap = cv2.resize(last_heatmap, (FLAGS.input_size, FLAGS.input_size))

    correct_and_draw_hand(test_img, last_heatmap, kalman_filter_array, tracker, crop_full_scale, crop_img)

    if FLAGS.DEMO_TYPE == 'MULTI':
        if len(demo_stage_heatmaps) > 3:
            upper_img = np.concatenate((demo_stage_heatmaps[0], demo_stage_heatmaps[1], demo_stage_heatmaps[2]), axis=1)
            lower_img = np.concatenate(
                (demo_stage_heatmaps[3], demo_stage_heatmaps[len(stage_heatmap_np) - 1], crop_img),
                axis=1)
            demo_img = np.concatenate((upper_img, lower_img), axis=0)
            return demo_img
        else:
            return np.concatenate((demo_stage_heatmaps[0], demo_stage_heatmaps[len(stage_heatmap_np) - 1], crop_img),
                                  axis=1)

    else:
        return crop_img


def correct_and_draw_hand(full_img, stage_heatmap_np, kalman_filter_array, tracker, crop_full_scale, crop_img):
    global joint_detections
    joint_coord_set = np.zeros((FLAGS.num_of_joints, 2))
    local_joint_coord_set = np.zeros((FLAGS.num_of_joints, 2))

    mean_response_val = 0.0

    # Plot joint colors
    if kalman_filter_array is not None:
        for joint_num in range(FLAGS.num_of_joints):
            tmp_heatmap = stage_heatmap_np[:, :, joint_num]
            joint_coord = np.unravel_index(np.argmax(tmp_heatmap),
                                           (FLAGS.input_size, FLAGS.input_size))
            mean_response_val += tmp_heatmap[joint_coord[0], joint_coord[1]]
            joint_coord = np.array(joint_coord).reshape((2, 1)).astype(np.float32)
            kalman_filter_array[joint_num].correct(joint_coord)
            kalman_pred = kalman_filter_array[joint_num].predict()
            correct_coord = np.array([kalman_pred[0], kalman_pred[1]]).reshape((2))
            local_joint_coord_set[joint_num, :] = correct_coord

            # Resize back
            correct_coord /= crop_full_scale

            # Substract padding border
            correct_coord[0] -= (tracker.pad_boundary[0] / crop_full_scale)
            correct_coord[1] -= (tracker.pad_boundary[2] / crop_full_scale)
            correct_coord[0] += tracker.bbox[0]
            correct_coord[1] += tracker.bbox[2]
            joint_coord_set[joint_num, :] = correct_coord

    else:
        for joint_num in range(FLAGS.num_of_joints):
            tmp_heatmap = stage_heatmap_np[:, :, joint_num]
            joint_coord = np.unravel_index(np.argmax(tmp_heatmap),
                                           (FLAGS.input_size, FLAGS.input_size))
            mean_response_val += tmp_heatmap[joint_coord[0], joint_coord[1]]
            joint_coord = np.array(joint_coord).astype(np.float32)

            local_joint_coord_set[joint_num, :] = joint_coord

            # Resize back
            joint_coord /= crop_full_scale

            # Substract padding border
            joint_coord[0] -= (tracker.pad_boundary[2] / crop_full_scale)
            joint_coord[1] -= (tracker.pad_boundary[0] / crop_full_scale)
            joint_coord[0] += tracker.bbox[0]
            joint_coord[1] += tracker.bbox[2]
            joint_coord_set[joint_num, :] = joint_coord

    draw_hand(full_img, joint_coord_set, tracker.loss_track)
    draw_hand(crop_img, local_joint_coord_set, tracker.loss_track)
    joint_detections = joint_coord_set

    if mean_response_val >= 1:
        tracker.loss_track = False
    else:
        tracker.loss_track = True

    cv2.putText(full_img, 'Response: {:<.3f}'.format(mean_response_val),
                org=(20, 20), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255, 0, 0))


def draw_hand(full_img, joint_coords, is_loss_track):
    if is_loss_track:
        joint_coords = FLAGS.default_hand

    # Plot joints
    for joint_num in range(FLAGS.num_of_joints):
        color_code_num = (joint_num // 4)
        if joint_num in [0, 4, 8, 12, 16]:
            joint_color = list(map(lambda x: x + 35 * (joint_num % 4), FLAGS.joint_color_code[color_code_num]))
            cv2.circle(full_img, center=(int(joint_coords[joint_num][1]), int(joint_coords[joint_num][0])), radius=3,
                       color=joint_color, thickness=-1)
        else:
            joint_color = list(map(lambda x: x + 35 * (joint_num % 4), FLAGS.joint_color_code[color_code_num]))
            cv2.circle(full_img, center=(int(joint_coords[joint_num][1]), int(joint_coords[joint_num][0])), radius=3,
                       color=joint_color, thickness=-1)

    # Plot limbs
    for limb_num in range(len(FLAGS.limbs)):
        x1 = int(joint_coords[int(FLAGS.limbs[limb_num][0])][0])
        y1 = int(joint_coords[int(FLAGS.limbs[limb_num][0])][1])
        x2 = int(joint_coords[int(FLAGS.limbs[limb_num][1])][0])
        y2 = int(joint_coords[int(FLAGS.limbs[limb_num][1])][1])
        length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        if length < 150 and length > 5:
            deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
            polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
                                       (int(length / 2), 3),
                                       int(deg),
                                       0, 360, 1)
            color_code_num = limb_num // 4
            limb_color = list(map(lambda x: x + 35 * (limb_num % 4), FLAGS.joint_color_code[color_code_num]))
            cv2.fillConvexPoly(full_img, polygon, color=limb_color)


if __name__ == '__main__':
    tf.app.run()
