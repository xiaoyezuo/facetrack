import os
import sys
sys.path.insert(0, 'FaceBoxesV2')
sys.path.insert(0, '..')
import numpy as np
import importlib
from PIPNet.FaceBoxesV2.faceboxes_detector import *

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as tvtransforms
import torchvision.models as models

from PIPNet.lib.networks import *
import PIPNet.lib.data_utils
from PIPNet.lib.functions import *
from PIPNet.lib.mobilenetv3 import mobilenetv3_large

from ert import extract_first_frame

pipnet_dir = "/home/zuoxy/PIPNet"
project_dir = "/home/zuoxy/facetrack/"
# img_path = "/home/zuoxy/PIPNet/images/2.jpg"
vid_id = "004"

def pipnet_detect(project_dir, vid_id, multiple_faces = False):
    
    experiment_name = 'pip_32_16_60_r101_l2_l1_10_1_nb10'
    data_name = 'data_300W'
    config_path = 'PIPNet.experiments.{}.{}'.format(data_name, experiment_name)

    my_config = importlib.import_module(config_path, package='PIPNet')
    Config = getattr(my_config, 'Config')
    cfg = Config()
    cfg.experiment_name = experiment_name
    cfg.data_name = data_name
    input_size = cfg.input_size
    net_stride = cfg.net_stride
    num_nb = cfg.num_nb

    image, gray = extract_first_frame(project_dir, vid_id)

    # save_dir = os.path.join('./snapshots', cfg.data_name, cfg.experiment_name)
    save_dir = os.path.join(pipnet_dir, 'snapshots', cfg.data_name, cfg.experiment_name)

    meanface_indices, reverse_index1, reverse_index2, max_len = get_meanface(os.path.join(pipnet_dir,'data', cfg.data_name, 'meanface.txt'), cfg.num_nb)
    resnet101 = models.resnet101(pretrained=cfg.pretrained)
    net = Pip_resnet101(resnet101, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    # weight_file = os.path.join(save_dir, 'epoch%d.pth' % (cfg.num_epochs-1))
    weight_file = os.path.join(save_dir, 'epoch%d.pth' % (cfg.num_epochs-1))
    state_dict = torch.load(weight_file, map_location=device)
    net.load_state_dict(state_dict)

    normalize = tvtransforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    preprocess = tvtransforms.Compose([tvtransforms.Resize((cfg.input_size, cfg.input_size)), tvtransforms.ToTensor(), normalize])

    detector = FaceBoxesDetector('FaceBoxes', pipnet_dir+'/FaceBoxesV2/weights/FaceBoxesV2.pth', True, device)
    my_thresh = 0.6
    det_box_scale = 1.2

    net.eval()
    # image = cv2.imread(image_file)
    image_height, image_width, _ = image.shape
    detections, _ = detector.detect(image, my_thresh, 1)
    # print(len(detections))
    x_coordinates = []
    y_coordinates = []

    if multiple_faces:
        for i in range(len(detections)):
            det_xmin = detections[i][2]
            det_ymin = detections[i][3]
            det_width = detections[i][4]
            det_height = detections[i][5]
            det_xmax = det_xmin + det_width - 1
            det_ymax = det_ymin + det_height - 1

            det_xmin -= int(det_width * (det_box_scale-1)/2)
            # remove a part of top area for alignment, see paper for details
            det_ymin += int(det_height * (det_box_scale-1)/2)
            det_xmax += int(det_width * (det_box_scale-1)/2)
            det_ymax += int(det_height * (det_box_scale-1)/2)
            det_xmin = max(det_xmin, 0)
            det_ymin = max(det_ymin, 0)
            det_xmax = min(det_xmax, image_width-1)
            det_ymax = min(det_ymax, image_height-1)
            det_width = det_xmax - det_xmin + 1
            det_height = det_ymax - det_ymin + 1
            cv2.rectangle(image, (det_xmin, det_ymin), (det_xmax, det_ymax), (0, 0, 255), 2)
            det_crop = image[det_ymin:det_ymax, det_xmin:det_xmax, :]
            det_crop = cv2.resize(det_crop, (cfg.input_size, cfg.input_size))
            inputs = Image.fromarray(det_crop[:,:,::-1].astype('uint8'), 'RGB')
            inputs = preprocess(inputs).unsqueeze(0)
            inputs = inputs.to(device)
            lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = forward_pip(net, inputs, preprocess, input_size, net_stride, num_nb)
            lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten()
            tmp_nb_x = lms_pred_nb_x[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
            tmp_nb_y = lms_pred_nb_y[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
            tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1,1)
            tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1,1)
            lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten()
            lms_pred = lms_pred.cpu().numpy()
            lms_pred_merge = lms_pred_merge.cpu().numpy()


            for i in range(cfg.num_lms):
                x_pred = lms_pred_merge[i*2] * det_width
                y_pred = lms_pred_merge[i*2+1] * det_height
                x_coordinates.append(int(x_pred) + det_xmin)
                y_coordinates.append(int(y_pred) + det_ymin)
                cv2.circle(image, (int(x_pred)+det_xmin, int(y_pred)+det_ymin), 1, (0, 0, 255), 2)
    else:
        i = 0
        det_xmin = detections[i][2]
        det_ymin = detections[i][3]
        det_width = detections[i][4]
        det_height = detections[i][5]
        det_xmax = det_xmin + det_width - 1
        det_ymax = det_ymin + det_height - 1

        det_xmin -= int(det_width * (det_box_scale-1)/2)
        # remove a part of top area for alignment, see paper for details
        det_ymin += int(det_height * (det_box_scale-1)/2)
        det_xmax += int(det_width * (det_box_scale-1)/2)
        det_ymax += int(det_height * (det_box_scale-1)/2)
        det_xmin = max(det_xmin, 0)
        det_ymin = max(det_ymin, 0)
        det_xmax = min(det_xmax, image_width-1)
        det_ymax = min(det_ymax, image_height-1)
        det_width = det_xmax - det_xmin + 1
        det_height = det_ymax - det_ymin + 1
        cv2.rectangle(image, (det_xmin, det_ymin), (det_xmax, det_ymax), (0, 0, 255), 2)
        det_crop = image[det_ymin:det_ymax, det_xmin:det_xmax, :]
        det_crop = cv2.resize(det_crop, (cfg.input_size, cfg.input_size))
        inputs = Image.fromarray(det_crop[:,:,::-1].astype('uint8'), 'RGB')
        inputs = preprocess(inputs).unsqueeze(0)
        inputs = inputs.to(device)
        lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = forward_pip(net, inputs, preprocess, input_size, net_stride, num_nb)
        lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten()
        tmp_nb_x = lms_pred_nb_x[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
        tmp_nb_y = lms_pred_nb_y[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
        tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1,1)
        tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1,1)
        lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten()
        lms_pred = lms_pred.cpu().numpy()
        lms_pred_merge = lms_pred_merge.cpu().numpy()


        for i in range(cfg.num_lms):
            x_pred = lms_pred_merge[i*2] * det_width
            y_pred = lms_pred_merge[i*2+1] * det_height
            x_coordinates.append(int(x_pred) + det_xmin)
            y_coordinates.append(int(y_pred) + det_ymin)
            cv2.circle(image, (int(x_pred)+det_xmin, int(y_pred)+det_ymin), 1, (0, 0, 255), 2)

    cv2.imwrite(project_dir + 'assets/images/detect_pip.jpg', image)
    im_rgb = cv2.resize(image, (256, 256))

    # x_coordinates,y_coordinates = demo_image(image_file, True, net, preprocess, cfg.input_size, cfg.net_stride, cfg.num_nb, cfg.use_gpu, device)

    x_coords = np.array(x_coordinates)
    y_coords = np.array(y_coordinates)

    num_pts = len(x_coords)
    # Create a zeros array for the frame indices
    frame_indices = np.zeros((num_pts, 1), dtype=np.int32)

    # Combine frame indices, y coordinates, and x coordinates
    points = np.concatenate((frame_indices, y_coords.reshape(num_pts,1), x_coords.reshape(num_pts,1)), axis=1).astype(np.int32)
    return points

# points = pipnet_detect(project_dir, vid_id)