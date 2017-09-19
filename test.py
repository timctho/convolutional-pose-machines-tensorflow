# import numpy as np
# import cv2
# import imageio
#
# # cap = cv2.VideoCapture('test_imgs/test.avi')
# #
# # while(cap.isOpened()):
# #     ret, frame = cap.read()
# #     if ret==True:
# #         # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# #         print(frame)
# #
# #         # cv2.imshow('frame',gray)
# #     else:
# #         break
# #     # if chr(cv2.waitKey(1)) == 'q':
# #     #     break
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break
# #
# # cap.release()
# # # cv2.destroyAllWindows()
#
# reader = imageio.get_reader('test_imgs/test.avi')
# for i, im in enumerate(reader):
#     print(im)

# import imageio
#
# reader = imageio.get_reader('test_imgs/single_man.mp4')
# length = reader.get_length()
# print(length)
# print(str(reader))
# for i, im in enumerate(reader):
#     # if im is None:
#     #     break
#     print('Mean of frame %i is %1.1f' % (i, im.mean()))
#     if i == length-1:
#         print(i)

# import tensorflow as tf
# import os
#
# FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_string('DEMO_TYPE',
#                            # default_value='test_imgs/roger.png',
#                            default_value='test_imgs/single_man.mp4',
#                            # default_value='SINGLE',
#                            docstring='MULTI: show multiple stage,'
#                                      'SINGLE: only last stage,'
#                                      'HM: show last stage heatmap,'
#                                      'paths to .jpg or .png image'
#                                      'paths to .avi or .flv or .mp4 video')
#
#
# writer_path = 'result' + '/' + FLAGS.DEMO_TYPE.split('/')[-1]
# print(writer_path)
# print(os.path.basename(FLAGS.DEMO_TYPE))
# writer_path = os.path.join('result', os.path.basename(FLAGS.DEMO_TYPE))
# print(writer_path)
