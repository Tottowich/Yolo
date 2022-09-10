import glob
import time
import math
import yaml
import torch
import os, sys
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# PYTORCH_ENABLE_MPS_FALLBACK=1
import argparse
import threading
from re import S
import numpy as np
from copy import copy
from tqdm import tqdm
from queue import Queue
from pathlib import Path
from tools.arguments import parse_config
from random_word import RandomWords
RANDOM_NAME = RandomWords()
import matplotlib.pyplot as plt 
from concurrent.futures import ThreadPoolExecutor
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT)) # Add ROOT
ROOT = Path(os.path.relpath(ROOT, Path.cwd())) # Relative Path
from contextlib import closing
import torch.backends.cudnn as cudnn
from utils.dataloaders import LoadStreams, LoadWebcam, LoadImages
from models.common import DetectMultiBackend
from tools.boliden_utils import (disp_pred, visualize_yolo_2D,create_logger, TimeLogger, RegionPredictionsTracker,
                                 create_logging_dir,scale_preds,initialize_digit_model,initialize_network,initialize_timer)

from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, time_sync
class LiveStream:
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
        """
        Prepare data from the lidar sensor.
        Args:
            img0: The image that is to be prepared.
        """
        img = self.reshape(copy(img0))
        if len(img.shape) == 3:
            img = img[None]
        img = img.transpose((0,3,1,2))  # BGR to RGB, BHWC to BCHW
        
        img = np.ascontiguousarray(img)
        self.frame += 1
        return img0,img
    def reshape(self,img):
        """
        Reshape the data to be compatible with the model.
        Args:
            img: the image to be reshaped.
        """
        img = cv2.resize(img,self.img_size)
        return img

# def initialize_network(args):
#     """
#     Initialize the network.
#     Create live streaming object to stream data from the sensor.
#     Create the detection model.
#     Args:
#         args: Arguments from the command line.
#     """
#     device = select_device(args.device)
#     half = args.half if device.type != 'cpu' else False  # half precision only supported on CUDA
#     model = DetectMultiBackend(args.weights, device=device, dnn=args.dnn, data=args.data, fp16=half)
#     model.eval()
#     stride, names, pt = model.stride, model.names, model.pt
#     imgsz = (args.imgsz, args.imgsz) if isinstance(args.imgsz, int) else args.imgsz  # tuple
#     imgsz = check_img_size(imgsz=imgsz, s=stride)
#     model.warmup(imgsz=(1 if pt else 1, 3, *imgsz))

#     # Create file to save logs to.
#     if args.save_time_log:
#         dir_path = create_logging_dir(args.name_run,ROOT / "logs",args)
#     else:
#         dir_path = None
#     # Horror video = https://www.youtube.com/watch?v=mto2mNFbrps&ab_channel=TromaMovies
#     source = args.source if args.source is not None else "0" if args.webcam else "https://www.youtube.com/watch?v=mto2mNFbrps&ab_channel=TromaMovies"
#     # Check if file
#     if os.path.isfile(source):
#         live = LoadImages(path=source,img_size=imgsz,stride=stride,auto=args.auto)
#     else:
#         live = LoadStreams(sources=source,img_size=imgsz,stride=stride,auto=args.auto)

#     pred_tracker = RegionPredictionsTracker(frames_to_track=args.frames_to_track,
#                                       img_size=imgsz,
#                                       threshold=args.tracker_thresh,
#                                       visualize=args.visualize,
#                                     )
#     log_file = None if not args.log_all else dir_path / "full_log.txt"

#     logger = create_logger(log_file=log_file)
#     if args.transmit:
#         transmitter = Transmitter(reciever_ip=args.TD_ip, reciever_port=args.TD_port)
#         transmitter.start_transmit_udp()
#         transmitter.start_transmit_ml()
#     else:
#         transmitter = None
#     if args.log_time:
#         time_logger = TimeLogger(logger,
#                                 args.disp_time,
#                                 save_log=args.save_time_log,
#                                 path=dir_path)
#         initialize_timer(time_logger,args)
#     else:
#         time_logger = None
#     return model, names, device, live, pred_tracker,transmitter, time_logger, logger
# def initialize_timer(time_logger:TimeLogger,args,transmitter:Transmitter=None):
#     """
#     Args:
#         time_logger: The logger object to log the time taken by various parts of the pipeline.
#         args: Arguments from the command line.
#         transmitter: If transmitter object is available then the time taken to transmit the data is also logged.
#     """
    
#     time_logger.create_metric("Pre Processing")
#     time_logger.create_metric("Infrence")
#     if args.cpu_post:
#         time_logger.create_metric("Reloading CPU")
#     time_logger.create_metric("Post Processing")
#     if args.visualize:
#         time_logger.create_metric("Visualize")
#     if args.save_csv:
#         time_logger.create_metric("Save CSV")
#     if transmitter is not None:
#         if transmitter.started_udp:
#             time_logger.create_metric("Transmit TD")
#         if transmitter.started_ml:
#             time_logger.create_metric("Transmit UE5")
#     if args.track:
#         time_logger.create_metric("Tracking Frames")
#     time_logger.create_metric("Internal Pipeline")
#     time_logger.create_metric("Full Pipeline")

#     return time_logger
            



# def parse_config():
#     """
#     Parse the configuration file.
#     """
#     parser = argparse.ArgumentParser(description='arg parser')
#     #parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
#     #                    help='specify the config for demo')
#     parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')

#     parser.add_argument('--source', type=str, default=None, help='model path(s)')
#     parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=1280, help='inference size h,w')
#     parser.add_argument('--data', type=str, default=ROOT / 'data/Argoverse.yaml', help='(optional) dataset.yaml path')
#     parser.add_argument('--max_det', type=int, default=1000, help='maximum detections per image')
#     parser.add_argument('--conf_thres', type=float, default=0.25, help='confidence threshold')
#     parser.add_argument('--iou_thres', type=float, default=0.45, help='NMS IoU threshold')
#     parser.add_argument('--line_thickness', default=3, type=int, help='bounding box thickness (pixels) visualizations')
#     parser.add_argument('--hide_labels', default=False, action='store_true', help='hide labels')
#     parser.add_argument('--hide_conf', default=False, action='store_true', help='hide confidences')
#     parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference instead of FP32 (default)')
#     parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
#     parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')    
#     parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
#     parser.add_argument('--auto', action='store_true', help='auto size using the model')
#     parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
#     parser.add_argument('--img_size', nargs='+', type=int, help='Size of input image')
#     parser.add_argument('--name_run', type=str, default=f"{RANDOM_NAME.get_random_word()}_{RANDOM_NAME.get_random_word()}", help='specify the name of the run to save the results')

#     # Object Tracking
#     parser.add_argument('--frames_to_track', type=int, default=6, help='Take the last n frames to track the certainty of the prediction.')
#     parser.add_argument('--tracker_thresh', type=float, default=0.2, help='Tracker threshold')
#     parser.add_argument('--class_to_track', type=int, default=1, help='Class index to track')
#     # Digit Tracking:
#     parser.add_argument('--track_digits', action='store_true', help='Track the digits')
#     parser.add_argument("--weights_digits",type=str,default="./runs/train/GrayBabyElephant/weights/best.pt",help="Path to model for digit detection")
#     parser.add_argument("--conf_digits",type=float,default=0.1,help="Confidence threshold for digit detection")
#     parser.add_argument("--iou_digits",type=float,default=0.1,help="NMS IoU threshold for digit detections")
#     parser.add_argument("--ind_thresh",type=float,default=0.5,help="Individual threshold if a score of an individual digit is below this threshold then the sequence is invalid")
#     parser.add_argument("--seq_thresh",type=float,default=0.5,help="Sequence threshold if the average score of the sequence is below this threshold then the sequence is invalid")
#     parser.add_argument("--out_thresh",type=float,default=0.5,help="Output threshold if the average score of the sequence of sequences is below this threshold then the sequence history is invalid")
#     parser.add_argument("--verbose",action="store_true",help="Whether to print information")
#     parser.add_argument('--data_digit', type=str, default=ROOT / './data/svhn.yaml', help='(optional) dataset.yaml path to digit dataset.')
#     parser.add_argument('--imgsz_digit', nargs='+', type=int, default=448, help='inference size h,w')


#     #parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

#     parser.add_argument('--name', type=str, default=None, help='specify the name of the sensor')

#     parser.add_argument('--udp_port', type=int, default=7502, help='specify the udp port of the sensor')
#     parser.add_argument('--tcp_port', type=int, default=7503, help='specify the tcp port of the sensor')
#     parser.add_argument('--TD_port', type=int, default=7002, help='specify the port of the TD machine')
#     parser.add_argument('--UE5_port', type=int, default=7000, help='specify the port of the UE5 machine')
#     parser.add_argument('--time', type=int, default=100
#     , help='specify the time to stream data from a sensor')
#     #parser.add_argument('--save_dir', type=str, default="../lidarCSV", help='specify the save directory')
#     #parser.add_argument('--save_name', type=str, default="test_csv", help='specify the save name')
#     if sys.version_info >= (3,9):
#         parser.add_argument('--augment', action='store_true', help='augmented inference')
#         parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
#         parser.add_argument('--visualize', action=argparse.BooleanOptionalAction)
#         parser.add_argument('--wait', action=argparse.BooleanOptionalAction,help="Wait for keypress after each visualizationqqqqqqqqqqqqq")
#         parser.add_argument('--prog_bar', action=argparse.BooleanOptionalAction)
#         parser.add_argument('--save_time_log', action=argparse.BooleanOptionalAction)
#         parser.add_argument('--track', action=argparse.BooleanOptionalAction)
#         parser.add_argument('--save_csv', action=argparse.BooleanOptionalAction)
#         parser.add_argument('--log_time', action=argparse.BooleanOptionalAction)
#         parser.add_argument('--disp_pred', action=argparse.BooleanOptionalAction)
#         parser.add_argument('--disp_time', action=argparse.BooleanOptionalAction)
#         parser.add_argument('--transmit', action=argparse.BooleanOptionalAction)
#         parser.add_argument('--pcd_vis', action=argparse.BooleanOptionalAction)
#         parser.add_argument('--webcam', action=argparse.BooleanOptionalAction)
#         parser.add_argument('--log_all', action=argparse.BooleanOptionalAction)
#         parser.add_argument('--cpu_post', action=argparse.BooleanOptionalAction)
#     args = parser.parse_args()
#     if isinstance(args.imgsz, list) and len(args.imgsz) == 1:
#         args.imgsz = args.imgsz[0]
#     with open(args.data,'r') as f:
#         try:
#             data_config = yaml.safe_load(f)
#         except:
#             raise ValueError(f"Invalid data config file: {args.data}")
#         #cfg_from_yaml_file(args.cfg_file, cfg)

#     return args,data_config#, cfg


@torch.no_grad() # No grad to save memory
def main():
    args, data_config = parse_config()
    init = True
    cudnn.benchmark = True  # set True to speed up constant image size inference
    model, names, device, live, pred_tracker, transmitter, time_logger, logger = initialize_network(args)
    if args.track_digits:
        dd = initialize_digit_model(args,logger=logger)
    else:
        dd = None
    log_time = False # False to let the program run for one loop to warm up :)


    logger.info(f"Infrence run stored @ ./logs/{args.name_run}")
    logger.info(f"Streaming data to: Yolov5 using {args.weights}")
    start_stream = time.monotonic()
    t1 = time.monotonic()

    if args.prog_bar:
        pbar = tqdm(total=args.time,bar_format = "{desc}: {percentage:.3f}%|{bar}|[{elapsed}<{remaining}")
    for rep in range(10):
        for i,(path, img, img0, vid_cap, s) in enumerate(live):       
            img0 = img0[0] if args.webcam else img0
            if log_time:
                time_logger.start("Internal Pipeline")
            if args.prog_bar:
                t2 = time.monotonic()
                pbar.update((t2 - t1))
                pbar.refresh()
                t1 = t2
            if init:
                if logger is not None:
                    logger.info(f"Image size: {img.shape}")
                    logger.info(f"img0 size: {img0.shape}")
                    pred_tracker.img_size = img.shape[2:]
                    pred_tracker.img0_size = img0.shape
                else:
                    print(f"Image: {img.shape}")
                    print(f"Img0: {img0.shape}")
                    pred_tracker.img_size = img.shape[2:]
                    pred_tracker.img0_size = img0.shape
            if log_time:
                time_logger.start("Pre Processing")
            img = torch.from_numpy(img).to(device)
            img = img.half() if args.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if log_time:
                time_logger.stop("Pre Processing")
            if len(img.shape) == 3:
                img = img.unsqueeze(0)
            if i%2 == 0 and log_time:
                time_logger.start("Full Pipeline")
            if i%2 == 1 and log_time and i != 1:
                time_logger.stop("Full Pipeline")
            
            if log_time:
                time_logger.start("Infrence")
            pred = model(img,augment=args.augment) # Inference + Loading to CPU for post processing
            if log_time:
                time_logger.stop("Infrence")
            if args.cpu_post and device != torch.device("cpu"):
                if log_time:
                    time_logger.start("Reloading CPU")
                pred = pred.cpu()
                if log_time:
                    time_logger.stop("Reloading CPU")
            if log_time:
                time_logger.start("Post Processing")
            # if device == torch.device("mps") or device == torch.device("cuda"): # Torchvision.ops.nms has not been implemented on MPS. REMOVE WHEN IMPLEMENTED
            #     pred = pred.cpu()
            pred = non_max_suppression(pred, args.conf_thres, args.iou_thres, agnostic=False, max_det=args.max_det)
            pred = scale_preds(preds=pred, img0=img0,img=img) # Returns list of Torch.Tensors. Each tensor is a prediction with shape: [x1,y1,x2,y2,conf,class]
            if log_time:
                time_logger.stop("Post Processing")
            if args.disp_pred:
                disp_pred(pred,names,logger)

            if args.track:
                if log_time:
                    time_logger.start("Tracking Frames")
                best_frame = pred_tracker.update(pred,img0,img)
                # print(best_frame)
                if log_time:
                    time_logger.stop("Tracking Frames")
                if args.track_digits and best_frame is not None:
                    if log_time:
                        time_logger.start("Tracking Digit")
                    sequence,valid = dd.detect(best_frame["image"])
                    if log_time:
                        time_logger.stop("Tracking Digit")
                    if valid: #and args.verbose:
                        logger.info(f"Predicted Sequence: {sequence}")
            if args.visualize:
                if log_time:
                    time_logger.start("Visualize")
                visualize_yolo_2D(pred,img0 = img0,img=img,args=args,names=names,wait=args.wait)     
                if log_time:
                    time_logger.stop("Visualize")
            if log_time and args.disp_pred:
                print("\n")
            if init:
                init = False
            if log_time:
                time_logger.stop("Internal Pipeline")
            log_time = args.log_time
            if time.monotonic()-start_stream > args.time:
                pbar.n = pbar.total
                pbar.refresh()
                break
    
    if args.transmit:
        transmitter.stop_transmit_udp()
        transmitter.stop_transmit_ml()
    if log_time:
        time_logger.summarize()
    logger.info("Stream Done")

"""
Example Input:
    py live_yolo.py --weights "yolov5n6.pt" --imgsz 1280  --iou_thres 0.2 --conf_thres 0.25 --log_time --time 2000 --no-prog_bar --save_time_log --max_det 10 --no-disp_pred --track --tracker_thresh 0.05 --frames_to_track 10 --visualize
"""

if __name__ == '__main__':
    main()
    #main_clean()

