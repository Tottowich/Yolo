import argparse
import os
import sys
import time
from pathlib import Path

import torch
from colorama import Fore, Style
from random_word import RandomWords

from tools.arguments import parse_config

RANDOM_NAME = RandomWords()
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT)) # Add ROOT
ROOT = Path(os.path.relpath(ROOT, Path.cwd())) # Relative Path
import torch.backends.cudnn as cudnn

from tools.boliden_utils import (ProgBar, disp_pred, initialize_digit_model, initialize_network, initialize_timer,
                                 scale_preds, visualize_yolo_2D, wait_for_input)
from utils.general import non_max_suppression

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
        pbar = ProgBar(live.is_live,args.time)
    for i,(path, img, img0, vid_cap, s) in enumerate(live):
        img0 = img0[0] if args.webcam else img0
        if log_time:
            time_logger.start("Internal Pipeline")
        if args.prog_bar and not init:
            if args.webcam:
                pbar.step()
            else:
                pbar.step()
        if log_time:
            time_logger.start("Pre Processing")
        img = torch.from_numpy(img).to(device)
        img = img.half() if args.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if init:
            t1 = time.monotonic()
            if args.verbose:
                logger.info(f"Image size: {img.shape}")
                logger.info(f"img0 size: {img0.shape}")
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
            visualize_yolo_2D(pred, img0=img0, img=img, args=args, names=names, line_thickness=3, classes_not_to_show=[0])     
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
                sequence, valid = dd.detect(img0=best_frame["image"]) if not args.force_detect_digits or best_frame is not None else dd.detect(img0)
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
                pbar.close()
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
--weights (type: str, default: ROOT / './TrainedModels/object/object.onnx')
Description: Path to the model weights file(s).
Example: --weights ./path/to/weights.onnx

--source (type: str, default: None)
Description: Path to the input source.
Example: --source ./path/to/input.mp4

--ip (type: str, default: None)
Description: IP address.
Example: --ip 192.168.0.1

--port (type: int, default: None)
Description: Port number.
Example: --port 8080

--imgsz or --img or --img-size (type: int or list[int], default: 448)
Description: Inference size (height and width) for the input image.
Example: --imgsz 512 or --img-size 640 480

--data (type: str, default: ROOT / "./TrainedModels/Object/data.yaml")
Description: Path to the dataset configuration file.
Example: --data ./path/to/data.yaml

--max_det (type: int, default: 1000)
Description: Maximum number of detections per image.
Example: --max_det 500

--conf_thres (type: float, default: 0.6)
Description: Confidence threshold for object detection.
Example: --conf_thres 0.5

--iou_thres (type: float, default: 0.1)
Description: Intersection over Union (IoU) threshold for non-maximum suppression (NMS).
Example: --iou_thres 0.2

--line_thickness (type: int, default: 3)
Description: Thickness of bounding box lines for visualizations (in pixels).
Example: --line_thickness 2

--hide_labels (action: store_true, default: False)
Description: Hide object labels in visualizations.
Example: --hide_labels

--hide_conf (action: store_true, default: False)
Description: Hide object confidences in visualizations.
Example: --hide_conf

--half (action: store_true)
Description: Use FP16 (half-precision) inference instead of FP32 (default).
Example: --half

--dnn (action: store_true)
Description: Use OpenCV DNN for ONNX inference.
Example: --dnn

--device (type: str, default: 'cuda:0')
Description: CUDA device to use for inference (e.g., 'cuda:0', 'cpu', 'mps').
Example: --device cuda:0

--ckpt (type: str, default: None)
Description: Path to the pretrained model checkpoint.
Example: --ckpt ./path/to/checkpoint.pth

--auto (action: store_true)
Description: Auto-size using the model.
Example: --auto

--classes (type: int or list[int])
Description: Filter detections by class index.
Example: --classes 0 or --classes 0 2 3

--img_size (type: int or list[int])
Description: Size of the input image.
Example: --img_size 800 or --img_size 1024 768

--name_run (type: str, default: randomly generated names)
Description: Name of the run to save the results.
Example: --name_run my_run

--object_frames (type: int, default: 3)
Description: Number of frames to track for object certainty.
Example: --object_frames 5

--tracker_thresh (type: float, default: 0.6)
Description: Tracker threshold for object tracking.
Example: --tracker_thresh 0.5

--class_to_track (type: int, default: 1)
Description: Class index to track.
Example: --class_to_track 2

--track_digits (action: store_true)
Description: Enable digit tracking.
Example: --track_digits

--digit_frames (type: int, default: 3)
Description: Number of frames to track for digit certainty.
Example: --digit_frames 5

--weights_digits (type: str, default: "./TrainedModels/digit/digit.onnx")
Description: Path to the model for digit detection.
Example: --weights_digits ./path/to/digit_model.onnx

--conf_digits (type: float, default: 0.3)
Description: Confidence threshold for digit detection.
Example: --conf_digits 0.5

--iou_digits (type: float, default: 0.1)
Description: IoU threshold for digit detections.
Example: --iou_digits 0.2

--ind_thresh (type: float, default: 0.1)
Description: Individual threshold for digit sequences.
Example: --ind_thresh 0.2

--seq_thresh (type: float, default: 0.2)
Description: Sequence threshold for digit sequences.
Example: --seq_thresh 0.3

--out_thresh (type: float, default: 0.35)
Description: Output threshold for sequence history.
Example: --out_thresh 0.4

--verbose (action: store_true)
Description: Print information during execution.
Example: --verbose

--data_digit (type: str, default: "./TrainedModels/digit/data.yaml")
Description: Path to the dataset configuration file for digit detection.
Example: --data_digit ./path/to/digit_data.yaml

--imgsz_digit (type: int or list[int], default: 448)
Description: Inference size (height and width) for digit detection.
Example: --imgsz_digit 512 or --imgsz_digit 640 480

--combination_file (type: str, default: "./TrainedModels/data/combinations.txt")
python live_yolo.py --iou_tresh
"""


# Example Input:
#
if __name__ == '__main__':
    main()
    #main_clean()

