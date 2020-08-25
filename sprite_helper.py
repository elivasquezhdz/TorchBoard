#
import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import os, json, cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import sys


def get_masked_image(image):
    cfg = get_cfg()
    cfg.MODEL.DEVICE='cpu'
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    outputs = predictor(image)
    predicted_classes = outputs["instances"].pred_classes.numpy()
    if not (0 in predicted_classes):
        return False
    mask = outputs["instances"].pred_masks.numpy()[0].astype('uint8')*255

    #m3chan = cv2.merge((mask,mask,mask))
    #masked = cv2.bitwise_and(image,m4chan)

    #Transparency
    b_channel, g_channel, r_channel = cv2.split(image)
    #mask,_,_ = cv2.split(cropped_mask)
    
    #alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255 #creating a dummy alpha channel image.
    img_BGRA = cv2.merge((b_channel, g_channel, r_channel,mask))
    #cv2_imshow(img_BGRA)
    return img_BGRA

def crop_image(res):
    image_data_bw = res.max(axis=2)
    non_empty_columns = np.where(image_data_bw.max(axis=0)>0)[0]
    non_empty_rows = np.where(image_data_bw.max(axis=1)>0)[0]
    cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))
    cropped_key_points = res[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1 , :]
    return cropped_key_points

def get_key_points(image):
    cfg = get_cfg()
    cfg.MODEL.DEVICE='cpu'
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml")
    predictor = DefaultPredictor(cfg)
    outputs = predictor(image)
    kpoints = outputs['instances'].pred_keypoints[0].numpy()
    knames = ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder",\
    "right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip",\
    "right_hip","left_knee","right_knee","left_ankle","right_ankle"]
    kdict = {p:k[0:2] for k,p in zip(kpoints,knames)}
    return kdict
      
def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result
