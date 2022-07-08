import argparse
import os, sys
import glob
from pathlib import Path
import time
import numpy as np
import torch
import threading
import math
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt 
from queue import Queue
from copy import copy
import yaml
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT)) # Add ROOT
ROOT = Path(os.path.relpath(ROOT, Path.cwd())) # Relative Path
import open3d
sys.path.insert(0, '../OusterTesting')
sys.path.insert(1, '../OpenPCDet-linux')
from models.common import DetectMultiBackend
import torch.backends.cudnn as cudnn
import utils_ouster
from tools.transmitter import Transmitter
from tools.visual_utils.open3d_live_vis import LiveVisualizer
from ouster import client
from contextlib import closing
from tools.visual_utils import open3d_vis_utils as V
#from pcdet.config import cfg, cfg_from_yaml_file
#from pcdet.datasets import DatasetTemplate
#from pcdet.models import build_network, load_data_to_gpu
from tools.xr_synth_utils import CSVRecorder,TimeLogger,filter_predictions,format_predictions,display_predictions
from tools.xr_synth_utils import create_logger,proj_and_format, proj_alt
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
class live_stream:
    """
    Class to stream data from a Sensor.
    Inheritance:
        DatasetTemplate:
            Uses batch processing and collation.
    """
    def __init__(self, classes,ip,stride=32,img_size=(1280,640), logger=None,auto=True):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        self.classes = classes
        self.ip = ip

        self.stride = stride
        self.img_size = img_size
        self.logger = logger
        self.auto = auto
        self.rect = True
        self.frame = 0
        
    def prep(self,img0):
        """started
        Prepare data from the lidar sensor.
        args:
            points: xyz/xyzr points from sensor.
        """
        #print(self.img_size)
        #print(img0.shape)
        img = self.reshape(copy(img0))
       # print(img.shape)
        if len(img.shape) == 3:
            img = img[None]
        #img = img[..., ::-1].transpose((0,3,1,2))  # BGR to RGB, BHWC to BCHW
        img = img.transpose((0,3,1,2))  # BGR to RGB, BHWC to BCHW
        
        img = np.ascontiguousarray(img)
        self.frame += 1
        return img0,img
    def reshape(self,img):
        """
        Reshape the data to be compatible with the model.
        Args:
            img:
        """
        img = cv2.resize(img,self.img_size,interpolation=cv2.INTER_LINEAR)
        return img

def initialize_network(args,device):
    device = select_device(args.device)
    model = DetectMultiBackend(args.weights, device=device, dnn=args.dnn, data=args.data, fp16=args.half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = (args.imgsz, args.imgsz) if isinstance(args.imgsz, int) else args.imgsz  # tuple
    imgsz = check_img_size(imgsz=imgsz, s=stride)
    imgsz = (imgsz[0],imgsz[0])
    model.warmup(imgsz=(1 if pt else 1, 3, *imgsz))
    live = live_stream(classes=names,ip=args.OU_ip,stride=stride,auto=args.auto)

    return model, stride, names, pt, device,live
def initialize_timer(transmitter,logger,args):
    time_logger = TimeLogger(logger,args.disp_pred)
    time_logger.create_metric("Ouster Processing")
    #time_logger.create_metric("Pre Processing")
    #time_logger.create_metric("Load GPU")
    time_logger.create_metric("Infrence")
    time_logger.create_metric("Post Processing")
    time_logger.create_metric("Format Predictions")

    if args.visualize:
        time_logger.create_metric("Visualize")
    if args.save_csv:
        time_logger.create_metric("Save CSV")
    if transmitter.started_udp:
        time_logger.create_metric("Transmit TD")
    if transmitter.started_ml:
        time_logger.create_metric("Transmit UE5")
    time_logger.create_metric("Full Pipeline")

    return time_logger
# class temp_scan:
#     h = 64
#     w = 1024
def visualize_yolo_2D_test(pred_dict,img,args,names=None,logger=None):
    detections = 0
    #print(f"Pre viz Average img: {img.mean()}")
    #heights = projec_2D_pred(pred,img[0],scan=s)
    img0 = np.ascontiguousarray(copy(img).squeeze().permute(1,2,0).cpu().numpy())
    annotator = Annotator(img0, line_width=args.line_thickness, example=str(names))
    for i,det,lbl,score in enumerate(zip(pred_dict["pred_boxes"],pred_dict["pred_lables"],pred_dict["scores"])):
        
        detections += 1
        if len(det):
            #print(img.shape[2:],img.squeeze().permute(1,2,0).shape)
            det[:,:4] = scale_coords(img.shape[2:], det[:,:4], img0.shape).round()
            # if args.disp_pred and logger is not None:
            #     for c in det[:, -1].unique():
            #         n = (det[:, -1] == c).sum()  # detections per class
            #         s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
            # logger.info(f"{names[int(c)]} detections: {s}")
            label = None if args.hide_labels else (names[lbl] if args.hide_conf else f'{names[lbl]} {score:.2f} {det[5]:.2f} {np.sqrt(det[0]**2+det[1]**2):.2f}')
            annotator.box_label(det[:4], label, color=colors(c, True))
            img0 = annotator.result()
            logger.info(f"Det: {det}")
            img0 = cv2.cvtColor(img0,cv2.COLOR_RGB2BGR)
            #print(f"Post viz Average img: {img0.mean()}")
            cv2.imshow("Predictions",img0)
            cv2.waitKey(1)
    img0 = annotator.result()
    logger.info(f"Det: {det}")
    img0 = cv2.cvtColor(img0,cv2.COLOR_RGB2BGR)
    #print(f"Post viz Average img: {img0.mean()}")
    cv2.imshow("Predictions",img0)
    cv2.waitKey(1)
def visualize_yolo_2D(pred,img,args,names=None,logger=None):
    detections = 0
    for i,det in enumerate(pred):
        
        detections += 1
        img0 = np.ascontiguousarray(copy(img).squeeze().permute(1,2,0).cpu().numpy())
        annotator = Annotator(img0, line_width=args.line_thickness, example=str(names))
        if len(det):
            #print(img.shape[2:],img.squeeze().permute(1,2,0).shape)
            det[:,:4] = scale_coords(img.shape[2:], det[:,:4], img0.shape).round()
            
            i = 0
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = None if args.hide_labels else (names[c] if args.hide_conf else f'{names[c]} {conf:.2f}')
                i += 1
                annotator.box_label(xyxy, label, color=colors(c, True))
            img0 = annotator.result()
            #logger.info(f"Det: {det}")
            img0 = cv2.cvtColor(img0,cv2.COLOR_RGB2BGR)
            #print(f"Post viz Average img: {img0.mean()}")
            cv2.imshow("Predictions",img0)
            cv2.waitKey(1)
        else:
            #print(img.shape)
            img0  = annotator.result()
            #print(img0.shape)
            img0 = cv2.cvtColor(img0,cv2.COLOR_RGB2BGR)
            cv2.imshow("Predictions",img0)
            #print(f"Post viz Average img: {img.mean()}")
            cv2.waitKey(1)
            



def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    #parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
    #                    help='specify the config for demo')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=1280, help='inference size h,w')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--max_det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--conf_thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--line_thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide_labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide_conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')    
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--auto', action='store_true', help='auto size using the model')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')

    #parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--OU_ip', type=str, default=None, help='specify the ip of the sensor')
    parser.add_argument('--name', type=str, default=None, help='specify the name of the sensor')
    parser.add_argument('--UE5_ip', type=str, default=None, help='specify the ip of the UE5 machine')
    parser.add_argument('--TD_ip', type=str, default="192.168.200.103", help='specify the ip of the TD machine')

    parser.add_argument('--udp_port', type=int, default=7502, help='specify the udp port of the sensor')
    parser.add_argument('--tcp_port', type=int, default=7503, help='specify the tcp port of the sensor')
    parser.add_argument('--TD_port', type=int, default=7002, help='specify the port of the TD machine')
    parser.add_argument('--UE5_port', type=int, default=7000, help='specify the port of the UE5 machine')
    parser.add_argument('--time', type=int, default=100
    , help='specify the time to stream data from a sensor')
    #parser.add_argument('--save_dir', type=str, default="../lidarCSV", help='specify the save directory')
    #parser.add_argument('--save_name', type=str, default="test_csv", help='specify the save name')
    if sys.version_info >= (3,9):
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--visualize', action=argparse.BooleanOptionalAction)
        parser.add_argument('--save_csv', action=argparse.BooleanOptionalAction)
        parser.add_argument('--log_time', action=argparse.BooleanOptionalAction)
        parser.add_argument('--disp_pred', action=argparse.BooleanOptionalAction) 
        

    else:
        parser.add_argument('--visualize', action='store_true')
        parser.add_argument('--no-visualize', dest='visualize', action='store_false')
        parser.add_argument('--save_csv', action='store_true')
        parser.add_argument('--no-save_csv', dest='save_csv', action='store_false')
        parser.add_argument('--log_time', action='store_true')
        parser.add_argument('--no-log_time', dest='log_time', action='store_false')
        parser.add_argument('--disp_pred', action='store_true')
        parser.add_argument('--no-disp_pred', dest='disp_pred', action='store_false')
        parser.set_defaults(visualize=True)
        parser.set_defaults(save_csv=False)
    args = parser.parse_args()
    with open(args.data,'r') as f:
        try:
            data_config = yaml.safe_load(f)
        except:
            raise ValueError(f"Invalid data config file: {args.data}")
        #cfg_from_yaml_file(args.cfg_file, cfg)

    return args,data_config#, cfg


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args,data_config = parse_config()
    range_limit = [10,10,5]
    cudnn.benchmark = True  # set True to speed up constant image size inference
    #model = DetectMultiBackend(args.weights, device=device, dnn=args.dnn, data=args.data, fp16=args.half)
    logger = create_logger()
    model, stride, names, pt, device,live = initialize_network(args,device)
    if args.OU_ip is None and args.name is None:
        raise ValueError('Please specify the ip or sensor name of the ')
    # Select classes to use, None -> all.
    # classes_to_use = [8]
    # Set up interactions
    #live = live_stream(cfg.DATA_CONFIG, cfg.CLASS_NAMES, logger=logger)
    if args.save_csv:
        recorder = CSVRecorder(args.save_name,args.save_dir, cfg.CLASS_NAMES)
    limits = {"ir":6000,"reflectivity": 255, "range":25000}
    #if range_limit is not None:
    #    cfg.DATA_CONFIG.POINT_CLOUD_RANGE = [-range_limit[0],-range_limit[1],-range_limit[2],range_limit[0],range_limit[1],range_limit[2]]
        
    # Set up network
    #model = initialize_network(cfg,args,logger,live)
    # Set up local network ports for IO
    transmitter = Transmitter(reciever_ip=args.TD_ip, reciever_port=args.TD_port, classes_to_send=[9])
    transmitter.start_transmit_udp()
    transmitter.start_transmit_ml()
    try:
        [cfg_ouster, host_ouster] = utils_ouster.sensor_config(args.name if args.name is not None else args.OU_ip,args.udp_port,args.tcp_port)
    except:
        raise ConnectionError('Could not connect to the sensor')
    log_time = False # False to let the program run for one loop to warm up :)
    if args.log_time:
        time_logger = initialize_timer(logger=logger,transmitter=transmitter,args=args)


    with closing(client.Scans.stream(host_ouster, args.udp_port,complete=False)) as stream:
        logger.info(f"Streaming lidar data to: Yolov5 using {args.weights}")
         # time 
        
        start_stream = time.monotonic()
        
        for i,scan in enumerate(stream): # Ouster scan object
            if log_time:
                time_logger.start("Ouster Processing")
            # Get lidar data
            img0 = utils_ouster.ir_ref_range(stream,scan,limits)
            pcd = utils_ouster.get_xyz(stream,scan)
            img0, img = live.prep(img0)
            if len(img0.shape) == 3:
                img0 = img0[None]
            img = torch.from_numpy(img).to(device)
            img = img.half() if model.fp16 else img.float()  # uint8 to fp16/32
            img_to_vis = copy(img)
            if log_time:
                time_logger.stop("Ouster Processing")
            
            #if range_limit is not None:
            #    xyzr = utils_ouster.trim_xyzr(xyzr,range_limit)
            #xyzr = utils_ouster.trim_data(data=xyzr,range_limit=range_limit,source=stream,scan=scan)
            #print(f"Input point cloud shape: {xyzr.shape}")
            if i%2 == 0 and log_time:
                time_logger.start("Full Pipeline")
            if i%2 == 1 and log_time and i != 1:
                time_logger.stop("Full Pipeline")
            
            

            #if log_time:
            #    time_logger.start("Load GPU")
            #load_data_to_gpu(data_dict)
            #if log_time:
            #    time_logger.stop("Load GPU")
           # print(img.shape)
            if log_time:
                time_logger.start("Infrence")
            
            pred = model(img,augment=args.augment)
            if log_time:
                time_logger.stop("Infrence")
            if log_time:
                time_logger.start("Post Processing")
            pred = non_max_suppression(pred, args.conf_thres, args.iou_thres, args.classes, args.agnostic_nms, max_det=args.max_det)
            if log_time:
                time_logger.stop("Post Processing")
            #print(pred)
            #pred_dict = proj_and_format(copy(pred),img[0],scan=scan)
            pred_dict = proj_alt(copy(pred),img[0].cpu().numpy(),xyz=pcd)
            if len(pred_dict["pred_labels"]) > 0 and args.disp_pred:
                display_predictions(pred_dict,names,logger)
            
                
            
            
            if args.save_csv: # If recording, save to csv
                if log_time:
                    time_logger.start("Save CSV")
                recorder.add_frame_file(copy(data_dict["points"][:,1:-1]).cpu().numpy(),pred_dict)
                if log_time:
                    time_logger.stop("Save CSV")
            
            if transmitter.started_ml:
                if log_time:
                    time_logger.start("Transmit UE5")
                transmitter.pcd = copy(pcd)
                transmitter.pred_dict = copy(pred_dict)
                transmitter.send_pcd()
                if log_time:
                    time_logger.stop("Transmit UE5")


            if transmitter.started_udp: # If transmitting, send to udp
                if log_time:
                    time_logger.start("Transmit TD")
                transmitter.pred_dict = copy(pred_dict)
                transmitter.send_dict()
                if log_time:
                    time_logger.stop("Transmit TD")

            
            if args.visualize:
                if log_time:
                    time_logger.start("Visualize")
                if range_limit is not None:
                    xyz = utils_ouster.trim_xyzr(utils_ouster.compress_mid_dim(pcd),range_limit)
                else:
                    xyz = utils_ouster.compress_mid_dim(pcd)
                if i == 0:
                    vis = LiveVisualizer("XR-SYNTHESIZER",
                                        class_names=names,
                                        first_cloud=xyz,
                                        classes_to_visualize=None
                                        )
                else:
                    vis.update(points=xyz, 
                            pred_boxes=pred_dict['pred_boxes'],
                            pred_labels=pred_dict['pred_labels'],
                            pred_scores=pred_dict['pred_scores'],
                            )
                visualize_yolo_2D(pred,img_to_vis,args,names=names,logger=logger)     
                if log_time:
                    time_logger.stop("Visualize")
                                #vis = V.create_live_scene(data_dict['points'][:,1:],ref_boxes=pred_dicts[0]['pred_boxes'],
                #ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'])
            # #elif args.visualize:
            #     start = time.monotonic()
            #     #V.update_live_scene(vis,pts,points=data_dict['points'][:,1:], ref_boxes=pred_dicts[0]['pred_boxes'],
            #     #    ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'],class_names=cfg.CLASS_NAMES)
            #     if log_time:
            #         time_logger.start("Visualize")
            #     vis.update(points=data_dict['points'][:,1:], 
            #                 pred_boxes=pred_dicts['pred_boxes'],
            #                 pred_labels=pred_dicts['pred_labels'],
            #                 pred_scores=pred_dicts['pred_scores'],
            #                 )
            #     if log_time:
            #         time_logger.stop("Visualize")
            if time.monotonic()-start_stream > args.time:
                stream.close()
                break
            if log_time and args.disp_pred:
                print("\n")
            #if i == 6:
            #    break 
            log_time = args.log_time
    transmitter.stop_transmit_udp()
    transmitter.stop_transmit_ml()
    if log_time:
        time_logger.visualize_results()
    logger.info("Stream Done")

"""
NuScenes uses the following labels:
    CLASS_NAMES: ['car','truck', 'construction_vehicle', 'bus', 'trailer',
              'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
    Note: 'pedestrian is predicted as index 9'
    This program uses has been tested with the Ouster OS0-64 sensor.
    Example Input:
        python3 live_predictions.py --cfg_file 'cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint.yaml' --ckpt "../checkpoints/cbgs_voxel0075_centerpoint_nds_6648.pth" --OU_ip "192.168.200.78" --TD_ip "192.168.200.103" --TD_port 7002  --time 300 --udp_port 7001 --tcp_port 7003 --name "OS0-64" --visualize

"""

if __name__ == '__main__':
    main()

