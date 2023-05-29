import cv2
import pandas as pd
import numpy as np
import torch
from ultralytics import YOLO

device = torch.device('cpu')

def risize_frame(frame, scale_percent):
    """Function to resize an image in a percent scale"""
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    return resized


def filter_tracks(centers, patience):
    """Function to filter track history"""
    filter_dict = {}
    for k, i in centers.items():
        d_frames = i.items()
        filter_dict[k] = dict(list(d_frames)[-patience:])

    return filter_dict


def update_tracking(centers_old, obj_center, thr_centers, lastKey, frame, frame_max):
    """Function to update track of objects"""
    is_new = 0
    lastpos = [(k, list(center.keys())[-1], list(center.values())[-1]) for k, center in centers_old.items()]
    lastpos = [(i[0], i[2]) for i in lastpos if abs(i[1] - frame) <= frame_max]
    # Calculating distance from existing centers points
    previous_pos = [(k, obj_center) for k, centers in lastpos if
                    (np.linalg.norm(np.array(centers) - np.array(obj_center)) < thr_centers)]
    # if distance less than a threshold, it will update its positions
    if previous_pos:
        id_obj = previous_pos[0][0]
        centers_old[id_obj][frame] = obj_center

    # Else a new ID will be set to the given object
    else:
        if lastKey:
            last = lastKey.split('D')[1]
            id_obj = 'ID' + str(int(last) + 1)
        else:
            id_obj = 'ID0'

        is_new = 1
        centers_old[id_obj] = {frame: obj_center}
        lastKey = list(centers_old.keys())[-1]

    return centers_old, id_obj, is_new, lastKey


def save_bounding_boxes(image, positions_frame, image_filename, output_filename):
    # Create empty lists to store the formatted bounding boxes
    bbox1_list = []
    bbox2_list = []

    # Iterate over each row in the DataFrame
    for _, row in positions_frame.iterrows():
        # Get the coordinates of the bounding box
        xmin = int(row['xmin'])
        ymin = int(row['ymin'])
        xmax = int(row['xmax'])
        ymax = int(row['ymax'])

        # Format the bounding box coordinates as strings
        bbox1_str = f"[({xmin}, {ymin})]"
        bbox2_str = f"[({xmax}, {ymax})]"

        # Append the formatted bounding box coordinates to the respective lists
        bbox1_list.append(bbox1_str)
        bbox2_list.append(bbox2_str)

    # Create a new DataFrame with the formatted bounding box coordinates
    output_df = pd.DataFrame({
        'filename': [image_filename] * len(positions_frame),
        'bbox1': bbox1_list,
        'bbox2': bbox2_list,
        'comma': ','
    })

    # Save the DataFrame to CSV
    output_df.to_csv(output_filename, index=False, quotechar='"', header=False, )


def create_yolo_boxes(image_path):
    #loading a YOLO model
    model = YOLO('yolov8x.pt')
    scale_percent = 100
    conf_level = 0.8
    class_IDS = [0]
    #geting names from classes
    dict_classes = model.model.names
    image_filename = image_path

    frame = cv2.imread(image_filename)

    #Applying resizing of read frame
    frame  = risize_frame(frame, scale_percent)
    area_roi = [np.array([ (1250, 400),(750,400),(700,800) ,(1200,800)], np.int32)]
    ROI = frame[390:800, 700:1300]

    # Getting predictions
    y_hat = model.predict(frame, conf = conf_level, classes = class_IDS, device = 0, verbose = False)

    # Getting the bounding boxes, confidence and classes of the recognize objects in the current frame.
    boxes   = y_hat[0].boxes.xyxy.cpu().numpy()
    conf    = y_hat[0].boxes.conf.cpu().numpy()
    classes = y_hat[0].boxes.cls.cpu().numpy()

    # Storing the above information in a dataframe
    positions_frame = pd.DataFrame(y_hat[0].cpu().numpy().boxes.data, columns = ['xmin', 'ymin', 'xmax', 'ymax', 'conf', 'class'])

    save_bounding_boxes(frame, positions_frame, image_filename, 'yolo_boxes.csv')



















