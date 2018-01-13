import numpy as np
import cv2


class SelfTracker(object):
    def __init__(self, img_shape, model_input_size):
        self.img_shape = img_shape
        self.loss_track = False
        self.prev_bbox = [0, 0, img_shape[0], img_shape[1]]
        self.init_center = [img_shape[0]//2, img_shape[1]//2]
        self.cur_center = [img_shape[0]//2, img_shape[1]//2]
        self._default_crop_size = 368
        self.bbox = [0, 0, 0, 0]
        self.pad_boundary = [0, 0, 0, 0]
        self.prev_crop_h = self._default_crop_size
        self.prev_crop_w = self._default_crop_size
        self.alpha = 0.2
        self.input_crop_ratio = 1.0
        self.input_size = float(model_input_size)


    def tracking_by_joints(self, full_img, joint_detections=None):
        if self.loss_track or joint_detections is None:
            cropped_img = self._crop_image(full_img, self.init_center, (self._default_crop_size, self._default_crop_size))
            self.input_crop_ratio = self.input_size / max(cropped_img.shape[0], cropped_img.shape[1])
            resize_img = self._resize_image(cropped_img, self.input_size)
            return self._pad_image(resize_img, max(resize_img.shape[0], resize_img.shape[1]))
        else:
            self.cur_center = np.mean(joint_detections, axis=0, dtype=np.int)
            crop_h = np.max(joint_detections[:, 0]) - np.min(joint_detections[:, 0])
            crop_w = np.max(joint_detections[:, 1]) - np.min(joint_detections[:, 1])
            crop_h = max(int(crop_h), 96)
            crop_w = max(int(crop_w), 96)
            crop_h *= 2.0
            crop_w *= 2.0
            self.prev_crop_h = self.alpha * crop_h + (1-self.alpha) * self.prev_crop_h
            self.prev_crop_w = self.alpha * crop_w + (1-self.alpha) * self.prev_crop_w

            cropped_img = self._crop_image(full_img, self.cur_center, (int(self.prev_crop_h), int(self.prev_crop_w)))
            self.input_crop_ratio = self.input_size / max(cropped_img.shape[0], cropped_img.shape[1])
            resize_img = self._resize_image(cropped_img, self.input_size)

            pad_size = max(resize_img.shape[0], resize_img.shape[1])
            return self._pad_image(resize_img, pad_size)


    def _resize_image(self, cropped_img, size):
        h, w, _ = cropped_img.shape
        if h > w:
            scale = size / h
            return cv2.resize(cropped_img, None, fx=scale, fy=scale)
        else:
            scale = size / w
            return cv2.resize(cropped_img, None, fx=scale, fy=scale)


    def _crop_image(self, full_img, center, size):
        h_offset = size[0] % 2
        w_offset = size[1] % 2
        self.bbox = [max(0, center[0]-size[0]//2), min(self.img_shape[0], center[0]+size[0]//2+h_offset),
                max(0, center[1]-size[1]//2), min(self.img_shape[1], center[1]+size[1]//2+w_offset)]
        return full_img[self.bbox[0]:self.bbox[1], self.bbox[2]:self.bbox[3], :]


    def _pad_image(self, img, size):
        h, w, _ = img.shape
        if size < h or size < w:
            raise ValueError('Pad size cannot smaller than original image size')

        pad_h_offset = (size - h) % 2
        pad_w_offset = (size - w) % 2
        self.pad_boundary = [(size-h)//2+pad_h_offset, (size-h)//2, (size-w)//2+pad_w_offset, (size-w)//2]
        return cv2.copyMakeBorder(img, top=self.pad_boundary[0],
                                  bottom=self.pad_boundary[1],
                                  left=self.pad_boundary[2],
                                  right=self.pad_boundary[3], borderType=cv2.BORDER_CONSTANT, value=(128, 128, 128))
