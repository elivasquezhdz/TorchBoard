import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from sklearn.cluster import KMeans
#import matplotlib.pyplot as plt

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

def crop_image(res):
    image_data_bw = res.max(axis=2)
    non_empty_columns = np.where(image_data_bw.max(axis=0)>0)[0]
    non_empty_rows = np.where(image_data_bw.max(axis=1)>0)[0]
    cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))
    cropped_key_points = res[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1 , :]
    return cropped_key_points

def crop_coords(res):
    image_data_bw = res.max(axis=2)
    non_empty_columns = np.where(image_data_bw.max(axis=0)>0)[0]
    non_empty_rows = np.where(image_data_bw.max(axis=1)>0)[0]
    return (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))

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

def get_cropped_points(keypoints, coords_crop):
    points = keypoints
    miny, _, minx , _ =coords_crop
    for p in points:
        pointx,pointy = points[p]
        points[p] = int(pointx - minx), int(pointy - miny)
    return points 

def mark_points_edges(edges_img,points):
    edges_copy = edges_img.copy()
    for p in points:
        x,y = points[p]
        edges_copy = cv2.circle(edges_copy, (x,y), radius=3, color =(255,155,0),thickness=-1)
    return edges_copy

def get_masked_image_rgb(image):
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
    img_RGBA = cv2.merge((r_channel,b_channel, g_channel,mask))
    #cv2_imshow(img_BGRA)
    return img_RGBA

def kMeansImage(image, k):
    idx = segmentImgClrRGB(image, k)
    return colorClustering(idx, image, k)

def colorClustering(idx, img, k):
    clusterValues = []
    for _ in range(0, k):
        clusterValues.append([])
    
    for r in range(0, idx.shape[0]):
        for c in range(0, idx.shape[1]):
            clusterValues[idx[r][c]].append(img[r][c])

    imgC = np.copy(img)

    clusterAverages = []
    for i in range(0, k):
        clusterAverages.append(np.average(clusterValues[i], axis=0))
    
    for r in range(0, idx.shape[0]):
        for c in range(0, idx.shape[1]):
            imgC[r][c] = clusterAverages[idx[r][c]]
            
    return imgC
    
def segmentImgClrRGB(img, k):
    
    imgC = np.copy(img)
    
    h = img.shape[0]
    w = img.shape[1]
    
    imgC.shape = (img.shape[0] * img.shape[1], 4)
    
    #5. Run k-means on the vectorized responses X to get a vector of labels (the clusters); 
    #  
    kmeans = KMeans(n_clusters=k, random_state=0).fit(imgC).labels_
    
    #6. Reshape the label results of k-means so that it has the same size as the input image
    #   Return the label image which we call idx
    kmeans.shape = (h, w)

    return kmeans

def get_limbs(cropped_key_points,image):
  h,w,_ = image.shape

  rightarm_x1 = 0
  rightarm_y1 = cropped_key_points['right_hip'][1]
  rightarm_x2 = cropped_key_points['right_shoulder'][0]
  rightarm_y2 = cropped_key_points['right_shoulder'][1]

  leftarm_x1 = cropped_key_points['left_shoulder'][0]
  leftarm_y1 = cropped_key_points['left_shoulder'][1]
  leftarm_x2 = w
  leftarm_y2 = cropped_key_points['left_hip'][1]

  rightleg_x1 = cropped_key_points['right_hip'][0]
  rightleg_y1 =cropped_key_points['right_hip'][1]
  rightleg_x2 =  0
  rightleg_y2 = h

  leftleg_x1 =w
  leftleg_y1 = h
  leftleg_x2 =  cropped_key_points['left_hip'][0]
  leftleg_y2 = cropped_key_points['left_hip'][1]

  leftleg = image[leftleg_y2:leftleg_y1,leftleg_x2:leftleg_x1,:]
  rightleg = image[rightleg_y1:rightleg_y2,rightleg_x2:rightleg_x1,:]
  rightarm = image[rightarm_y2:rightarm_y1,rightarm_x1:rightarm_x2,:]
  leftarm = image[leftarm_y1:leftarm_y2,leftarm_x1:leftarm_x2,:]

  return {"leftleg":leftleg,"rightleg":rightleg,"rightarm":rightarm,"leftarm":leftarm}

def resize_(out_width=50,color_clusters=15,ksize=(10,10),image=None):

  # Get input size
  height, width = masked_crop.shape[:2]

  # Desired "pixelated" size
  factor = int(width/out_width)
  w, h = (out_width, int(height/factor))

  # Resize input to "pixelated" size
  temp = cv2.resize(masked_crop, (w, h), interpolation=cv2.INTER_LINEAR)
  #resized_down = cv2.resize(image, down_points, interpolation= cv2.INTER_LINEAR)
  # Initialize output image
  output = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
  blured = cv2.blur(kMeansImage(output,color_clusters),ksize)
  return blured

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result
