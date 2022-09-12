# import glob
import time
# import math
# import yaml
import torch
import os, sys
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# PYTORCH_ENABLE_MPS_FALLBACK=1
import argparse
# import threading
from re import S
import numpy as np
from copy import copy
from tqdm import tqdm
from queue import Queue
from pathlib import Path
from colorama import Fore, Style
from tools.arguments import parse_config
from random_word import RandomWords
RANDOM_NAME = RandomWords()
# import matplotlib.pyplot as plt 
# from concurrent.futures import ThreadPoolExecutor
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT)) # Add ROOT
ROOT = Path(os.path.relpath(ROOT, Path.cwd())) # Relative Path
# from contextlib import closing
import torch.backends.cudnn as cudnn
# from utils.dataloaders import LoadStreams, LoadWebcam, LoadImages
# from models.common import DetectMultiBackend
from tools.boliden_utils import (disp_pred, visualize_yolo_2D,scale_preds,initialize_digit_model,initialize_network,initialize_timer, wait_for_input)

from utils.general import non_max_suppression
                        #     ,(LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                        #    increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
# from utils.torch_utils import select_device, time_sync
EPS = 1e-8

@torch.no_grad() # No grad to save memory
def main(args: argparse.Namespace=None) -> None:
    if args is None:
        args, data_config = parse_config()
    init = True
    cudnn.benchmark = True  # set True to speed up constant image size inference
    model, names, device, live, pred_tracker, transmitter, time_logger, logger = initialize_network(args)
    # Transmitter is currently unused
    if args.track_digits:
        dd = initialize_digit_model(args,logger=logger)
    else:
        dd = None
    log_time = False # False to let the program run for one loop to warm up :)


    logger.info(f"Infrence run stored @ ./logs/{args.name_run}")
    logger.info(f"Streaming data to: Yolov5 using {args.weights}")
    start_stream = time.monotonic()
    if args.prog_bar:
        if args.time > 0:
            t1 = time.monotonic() # Time to keep track of FPS count when using progress bar 
            pbar = tqdm(total=args.time,bar_format = "{desc}: {percentage:.3f}%|{bar}|[{elapsed}<{remaining}",leave=True)
        else:
            pbar = tqdm(total=len(live),bar_format = "{desc}: {percentage:.3f}%|{bar}|[{elapsed}<{remaining}",leave=True)
        
    
    # multidirs = os.listdir("./datasets/Examples/Sequences/")
    # lives = [LoadImages(path=os.path.join("./datasets/Examples/Sequences",dire), img_size=448, stride=32, auto=False) for dire in multidirs]
        # print(f"Starting new sequence: ")
    for i,(path, img, img0, vid_cap, s) in enumerate(live):
        img0 = img0[0] if args.webcam else img0
        if log_time:
            time_logger.start("Internal Pipeline")
        if args.prog_bar:
            if args.time > 0:
                t2 = time.monotonic()
                pbar.update((t2 - t1))
                pbar.bar_format = "{desc}: {percentage:.1f}%|{bar:10}|[{elapsed}<{remaining}]"
                pbar.set_description(str(f"Elapsed time: {time.strftime('%H:%M:%S', time.gmtime(t2 - start_stream))}|FPS: {(1/(t2-t1+EPS)):.3f}|"))
                pbar.refresh()
                t1 = t2
            else:
                pbar.update(1)
                pbar.refresh()
        if init:
            if args.verbose:
                logger.info(f"Image size: {img.shape}")
                logger.info(f"img0 size: {img0.shape}")
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
        pred = model(img,augment=args.augment) # Inference
        if log_time:
            time_logger.stop("Infrence")
        if device == torch.device("mps"): # Slow NMS on mps -> move to cpu when using mps (Apple Silicon)
            if log_time:
                time_logger.start("Reloading CPU")
            pred = pred.cpu()
            if log_time:
                time_logger.stop("Reloading CPU")
        if log_time:
            time_logger.start("Post Processing")

        pred = non_max_suppression(pred, args.conf_thres, args.iou_thres, agnostic=False, max_det=args.max_det)
        pred = scale_preds(preds=pred, img0=img0,img=img) # Returns list of Torch.Tensors. Each tensor is a prediction with shape: [x1,y1,x2,y2,conf,class]
        if log_time:
            time_logger.stop("Post Processing")
        if args.disp_pred:
            disp_pred(pred,names,logger)
        if args.visualize:
            if log_time:
                time_logger.start("Visualize")
            visualize_yolo_2D(pred,img0 = img0,img=img,args=args,names=names,wait=False,line_thickness=3,classes_not_to_show=[0])     
            if log_time:
                time_logger.stop("Visualize")

        if args.track:
            """
            If tracking is enabled, the tracker will be updated with the new predictions.
            """
            if log_time:
                time_logger.start("Tracking Frames")
            # Object Tracking
            best_frame = pred_tracker.update(pred,img0,img)
            if log_time:
                time_logger.stop("Tracking Frames")
            if args.track_digits and best_frame is not None or args.force_detect_digits:
                if log_time:
                    time_logger.start("Tracking Digit")
                # Digit Sequence Tracking.
                sequence,valid = dd.detect(img0=best_frame["image"]) if not args.force_detect_digits or best_frame is not None else dd.detect(img0)
                if log_time:
                    time_logger.stop("Tracking Digit")
                if valid:
                    logger.info(f"Predicted Sequence: {Fore.GREEN}{sequence}{Style.RESET_ALL}\n")
                    # TODO: Save to file or transmit to server. Maybe??
        if args.disp_pred or args.verbose:
            print("\n")
        if init:
            init = False
        if log_time:
            time_logger.stop("Internal Pipeline")
        log_time = args.log_time
        if time.monotonic()-start_stream > args.time and args.time != -1:
            if args.prog_bar:
                pbar.n = pbar.total
                pbar.refresh()
            break
        if args.wait:
            wait_for_input(live=live,args=args)


    if args.transmit:
        transmitter.stop_transmit_udp()
        transmitter.stop_transmit_ml()
    if log_time:
        time_logger.summarize()
    logger.info("Stream Done")

"""
Example Input:
    py live_yolo.py --weights "yolov5n6.pt" --imgsz 1280  --iou_thres 0.2 --conf_thres 0.25 --log_time --time 2000 --no-prog_bar --save_time_log --max_det 10 --no-disp_pred --track --tracker_thresh 0.05 --frames_to_track 10 --visualize
py live_yolo.py 
            # Main Object Detection:
            --data: str # Path to data file (.yaml). Should contain class names and number of classes. 
            --weights: str # Path to Object Detection Model.
            --imgsz: list[int,int] or int # Image size predictions are made in. Default: 448
            --iou_thres: float[0->1] # IOU Threshold for NMS related to object detection. Default: 0.4
            --conf_thres: float[0->1] # Confidence Threshold for object detection. Default: 0.5
            --max_det: int # Maximum number of detections per frame to be made. Default: 1000
            --track: store_true# Track best detection over time. Default: False
            --class_to_track: int # Class to track. Default 1
            --tracker_thresh: float[0->1] # Threshold the prediction must be above to be considered for tracking. Default: 0.5
            --object_frames: int # Number of frames to track object for, i.e. the number of frames the object must be detected for to be considered valid. Default: 3
            # Digit Detection:
            --track_digits: store_true # Whether to track digits in bounding boxes of tracked objects. Default: False
            --data_digit: str # Path to data file (.yaml). Should contain class names and number of classes.
            --weights_digits: str # Path to Digit Detection Model.
            --conf_digit 0.3 # Digit Prediction Confidence Threshold @ Predtiction stage
            --iou_digit 0.3 # Digit Prediction IoU Threshold @ Predtiction stage
            --ind_thresh: float[0->1] # Threshold for digit detection tracking. If one of the digits is bleow this threshold, the sequence is considered invalid. Default: 0.4
            --seq_thresh: float[0->1] # Threshold for digit detection tracking. If the average sequence score is below this threshold, the sequence is considered invalid. Default: 0.5
            --out_thresh: float[0->1] # Threshold for digit detection tracking. if the average score of the sequences in the history of previous sequences is below this threshold, the history is not valid else valid output. Default: 0.5
            --digit_frames: int # Number of frames to track digits for, i.e. the number of frames the predicted digit sequence must be valid and consistent. Default: 3
            
            --force_detect_digits: store_true # Force digit detection on every frame. Default: False.
            # Other:
            --source: str # Path to directory image or video file. Default: None
            --verbose: store_true # Extensive information to console. Default: False
            --log_time: store_true # Log time taken for each step of the pipeline. Default: False
            --prog_bar: store_true # Show progress bar for live stream. Default: False
            --save_time_log: store_true # Save time log to file as csv under logs folder. Default: False
            --disp_pred: store_true # Display predictions in terminal. Default: False
            --visualize: store_true # Visualize predictions
            --webcam: store_true # Use webcam as input
            --device: str # Device to run on
            --time: int # Time to run in seconds if -1 runs forever
            --half: store_true # Half precision FP16 inference ONLY ON GPU
            --line_thickness: int # Line thickness for visualization
            --hide_labels: store_true # Hide labels in visualization
            --hide_conf: store_true # Hide confidence in visualization
            --name_run: str # Name of run for logging of time

"""



# Example Input:
#
if __name__ == '__main__':
    main()
    #main_clean()

