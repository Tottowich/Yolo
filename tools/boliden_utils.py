from ast import For
from dataclasses import dataclass,field
import os
import re
import sys
from tabnanny import verbose
import cv2
import time
import yaml
import torch
import logging
import argparse
import torchvision
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
from copy import copy
from datetime import datetime
import matplotlib.pyplot as plt
from colorama import Fore, Style
# if __name__=="__main__":
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT)) # Add ROOT
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))
# print(f"ROOT: {ROOT}")
from tools.transmitter import Transmitter
from utils.dataloaders import LoadStreams, LoadWebcam, LoadImages
from models.common import DetectMultiBackend
from utils.general import LOGGER, scale_coords, check_img_size
from utils.augmentations import letterbox
from utils.torch_utils import select_device, time_sync
from utils.general import non_max_suppression
from utils.plots import Annotator, colors, save_one_box
from typing import Dict, Union, Tuple, Optional,Dict,TypeVar
EPS = 1e-9
OFFSET = 30 # ,30]
def xyxy2xywh(xyxy:tuple)->tuple:
    x1,y1,x2,y2 = xyxy
    return ((x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1)
def disp_pred(pred:Union[np.ndarray,list],names:list,logger:LOGGER)->None:
    """
    Display each prediction made,
    and how many of each class is predicted.
    Args:
        pred: Predictions from the model.
        names: Names of the classes.
        logger: Logger for logging.
    """
    assert logger is not None, "Logger object is not passed"
    logger.info(f"{Fore.GREEN}Predictions:{Style.RESET_ALL}\n")
    class_count = np.zeros((len(names),1), dtype=np.int)
    for i,det in enumerate(pred):
        if len(det):
            for j,(*xyxy, conf, cls) in enumerate(det):
                c = int(cls)
                class_count[c] += 1
    for i,name in enumerate(names):
        if class_count[i]>0:
            logger.info(f"{Fore.GREEN}{name}:{Style.RESET_ALL} {class_count[i]}")
    logger.info(f"{Fore.GREEN}Total:{Style.RESET_ALL} {np.sum(class_count)}")
    logger.info(f"{Fore.GREEN}Most Common:{Style.RESET_ALL} {names[np.argmax(class_count)]}")

def visualize_yolo_2D(pred:np.ndarray,img0:np.ndarray,args:argparse=None,names:list[str]=None,rescale:bool=False,img:torch.Tensor=None,wait:bool=False,line_thickness:int=1, hide_labels:bool=False, hide_conf:bool=False,image_name:str="Object Predicitions")->None:
    """
    Visualize the predictions.\n
    Args:
        pred: Predictions from the model.\n
        pred_dict: Dictionary of predictions.\n
        img0: Image collected from the camera. # Shape: (H,W,C).\n
        img: Image predictions where based upon. # Shape: (1,3,H,W).\n
        args: Arguments from the command line.\n
        names: Names of the classes.\n
    """
    class_string = None
    if rescale:
        img0 = cv2.resize(img0.copy(),(640,int(640/img0.shape[1]*img0.shape[0])))
    else:
        img0 = img0.copy()
    if args is None:
        assert line_thickness is not None, "Line thickness is not passed"
        assert hide_labels is not None, "Hide labels is not passed"
        assert hide_conf is not None, "Hide confidence is not passed"
    if rescale:
        assert img is not None, "Image is not passed"
    # img0 = img0.copy()
    for i,det in enumerate(pred):
        
        #img0 = np.ascontiguousarray(copy(img).squeeze().permute(1,2,0).cpu().numpy())
        annotator = Annotator(img0, line_width=line_thickness, example=str(names))
        if len(det):
            if rescale:
                det[:,:4] = scale_coords(img.shape[2:], det[:,:4], img0.shape).round()
            
            i = 0
            classes = []
            pos_x = []
            for j,(*xyxy, conf, cls) in enumerate(det):#reversed(det)):
                c = int(cls)  # integer class
                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                i += 1
                annotator.box_label(xyxy, label, color=colors(c, True))
                # classes.append(c)
                # pos_x.append(xyxy[0])
            img0 = annotator.result()
            class_string = ""
            while len(classes)>0:
                id = np.argmin(pos_x)
                class_string += f"{names[classes[id]]}"
                pos_x.pop(id)
                classes.pop(id)
            #img0 = cv2.cvtColor(img0,cv2.COLOR_RGB2BGR)
            cv2.putText(img0,class_string,(int(30),int(30)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv2.imshow(image_name,cv2.resize(img0,(640,int(640/img0.shape[1]*img0.shape[0]))))
            if wait:
                cv2.waitKey(0)
            if not wait:
                cv2.waitKey(1)
            else:
                q = cv2.waitKey(0)
                while q != ord("q"):
                    q = cv2.waitKey(0)
                    if q == ord("e"):
                        exit()
                cv2.destroyAllWindows()

        else:
            img0  = annotator.result()
            #img0 = cv2.cvtColor(img0,cv2.COLOR_RGB2BGR)
            cv2.imshow(image_name,img0)
            if not wait:
                cv2.waitKey(1)
            else:
                q = cv2.waitKey(0)
                while q != ord("q"):
                    q = cv2.waitKey(0)
        return class_string

class TimeLogger:
    """
    Class to log time and display with pandas dataframe and matplotlib.\n
    Args:
        logger: logger object to print the time.
        disp_log: if True, display logs.
        save_log: if True, save logs. Will save to timelogger.csv under './logs/' if no name is specified.
        name: name of the timelogger.
    """
    def __init__(self,logger=None,disp_log=False,save_log=False,path="timelogger"):
        super().__init__()
        self.time_dict = {}
        self.time_pd = None
        self.metrics_pd = None
        self.logger = logger
        self.save_log = save_log
        self.path = path
        if disp_log is not None:
            self.print_log = disp_log
        else:
            self.print_log = False

    def output_log(self,name:str):
        """
        Output the time taken at each step.
        Args:
            name: name of the step, i.e. pre_process, post_process, etc.
        """
        if self.logger is not None:
            self.logger.info(f"{name}: {self.time_dict[name]['times'][-1]:.3e} s <=> {1/(self.time_dict[name]['times'][-1]+EPS):.3e} Hz")
        else:
            print(f"{name}: {self.time_dict[name]['times'][-1]:.3e} s <=> {1/(self.time_dict[name]['times'][-1]+EPS):.3e} Hz")
    def create_metric(self, name: str):
        """
        Create a new metric beloning to the timelogger object.
        Args:
            name: name of the metric.
        """
        self.time_dict[name] = {}
        self.time_dict[name]["times"] = []
        self.time_dict[name]["start"] = 0
        self.time_dict[name]["stop"] = 0   
    def start(self, name: str):
        """

        """
        if name not in self.time_dict.keys():
            self.create_metric(name)
            if self.logger is not None:
                self.logger.info(f"{name} had not been initialized, initializing now.")
        self.time_dict[name]["start"] = time.monotonic()
    def stop(self, name: str):
        self.time_dict[name]["stop"] = time.monotonic()
        self.time_dict[name]["times"].append(self.time_dict[name]["stop"] - self.time_dict[name]["start"])
        if self.print_log:
            self.output_log(name)
    def log_time(self, name: str, _time: float):
        self.time_dict[name]["times"].append(_time)
    def maximum_time(self, name: str):
        if self.time_dict[name]["times"] is not None and len(self.time_dict[name]["times"])>0:
            return max(self.time_dict[name]["times"])
        return 0
    def minimum_time(self, name: str):
        if self.time_dict[name]["times"] is not None and len(self.time_dict[name]["times"])>0:
            return min(self.time_dict[name]["times"])
        return 0
    def average_time(self, name: str):
        if self.time_dict[name]["times"] is not None and len(self.time_dict[name]["times"])>0:
            return np.mean(self.time_dict[name]["times"])
        return 0
   
    def summarize(self):
        time_averages = {}
        time_max = {}
        time_min = {}
        self.time_pd = {}
        sum_ave = 0
        keys = len(self.time_dict)
        
        fig, axs = plt.subplots(keys,1)
        for i,key in enumerate(self.time_dict):
           
            if len(self.time_dict[key]["times"])>0:
                axs[i].plot(self.time_dict[key]["times"],label=key)
                axs[i].set_title(key)
                time_max[key] = self.maximum_time(key)
                time_min[key] = self.minimum_time(key)
                time_averages[key] = np.mean(np.delete(self.time_dict[key]["times"],np.argmax(self.time_dict[key]["times"])))
                sum_ave += time_averages[key] if key not in ["Full Pipeline","Internal Pipeline"] else 0
        plt.show()
        
        self.metrics_pd = pd.DataFrame([time_averages,time_max,time_min],index=["average","max","min"])
        if self.logger is not None:
            self.logger.info(f"Table To summarize:\n{self.metrics_pd}\nLoading time: {self.metrics_pd['Full Pipeline']['average']-sum_ave:.3e} s\nFrames per second: {1/self.metrics_pd['Full Pipeline']['average']:.3e} Hz")
        else:
            print(f"Table To summarize:\n{self.metrics_pd}")
        if self.save_log:
            self.metrics_pd.to_csv(f"{self.path}/timings.csv")
    


class RegionPredictionsTracker:
    """
    Track the 'n' previous frame's largest predictions.
    Calculates a score across the sequence of predictions made by calculating the certainty
    of the prediction and the size of the area.
    High certainty and large area = high score.
    If average score across the sequence is above a threshold, the prediction is considered a success and passed along the pipeline.\n
    Args:
        frames_to_track: number of frames to track.
        img_size: size of the image the predictions where made in.
        img0_size: size of the original image.
        threshold: threshold to determine if the prediction sequence is a success.
        visualize: if True, will display the tracked predictions.
        sequence_patience: time in seconds to allow seperated best frames may be considered part of the same sequence.

    """
    def __init__(self,
                    frames_to_track:int,
                    img_size:Union[tuple,int]=None,
                    img0_size:Union[tuple,int]=None,
                    threshold:float=0.5,
                    visualize:bool=False,
                    sequence_patience:float=5.0,
                    class_to_track:int=0,
                    verbose:bool=False,
                    logger:logging.Logger=None,
                ) -> None:
        assert frames_to_track > 0, "frames_to_track must be greater than 0."
        assert img_size is not None, "img_size must be specified."
        assert threshold <= 1, "threshold must be less or equal to 1."
        assert type(class_to_track) == int, "class_to_track must be an integer."
        self.frames_to_track = frames_to_track
        self.class_to_track = class_to_track
        self.img_size = img_size if isinstance(img_size,tuple) else (img_size,img_size) if img_size is not None else None
        self.img0_size = img0_size if isinstance(img0_size,tuple) else (img0_size,img0_size,3) if img0_size is not None else None
        self.threshold = threshold
        self.scores = []
        self.images = []
        self.best_frame = None
        self.best_score = 0
        self.previous_tracked_class = None
        self.visualize = visualize
        if verbose:
            assert logger is not None, "logger must be specified if verbose is True."
        self.verbose = verbose
        self.logger = logger
    def get_largest_area(self,pred:list[Union[np.ndarray,torch.Tensor]],img0:np.ndarray,img:torch.Tensor)->Union[Dict[str, Union[int,float,float,np.ndarray]],None]:
        """
        From the predictions made. Return the largest area of the bounding box.
        Args:
            pred: Predictions from the model.
            img0: Original image, single frame.
        Returns:
            largest_attribute:  Dictionary containing attributes of the largest bounding box.
                                Will contain:\n
                                    - 'index': index of the largest bounding box.
                                    - 'confidence': confidence of the largest bounding box.
                                    - 'largest_area': area of the largest bounding box.
                                    - 'image': cut out of the image the largest bounding box from original image.
        """
        largest_area = None
        largest_area_index = None
        confidence_largest_area = None
        largest_attributes ={
            "index":None,
            "confidence":None,
            "largest_area":0,
            "image": None,
            "class": None,
        }
        #img0 = np.ascontiguousarray(img0)
        for i,det in enumerate(pred):
            if len(det): # if there is a detection
                #det[:,:4] = scale_coords(img.shape[2:], det[:,:4], img0.shape).round()
                for j,(*xyxy, conf, cls) in enumerate(det):
                    if int(cls) == self.class_to_track:
                        area = (xyxy[2]-xyxy[0])*(xyxy[3]-xyxy[1])/(self.img0_size[0]*self.img0_size[1]) # Procentage of the image size, float
                        if area > largest_attributes["largest_area"]:
                            largest_attributes["index"] = j
                            largest_attributes["confidence"] = float(conf)
                            largest_attributes["largest_area"] = float(area)
                            largest_attributes["image"] = get_cut_out(img0,xyxy,offset=OFFSET)
                            largest_attributes["class"] = int(cls)
        return largest_attributes if largest_attributes["largest_area"] > 0 else None

    def update(self,predictions:np.ndarray,img0:np.ndarray,img:torch.Tensor)->Union[None,Dict[str,Union[np.ndarray,float]]]:
        """
        Update the predictions.
        Args:
            predictions: predictions of the current frame.
            img: image predictions where made in.
            img0: image of the current frame.

        Returns:
            None: If the sequence is not long enough or the average score is below the threshold.
            best_frame: Dictionary containing the best frame and the sequence's combined score.
        """
        largest_attributes = self.get_largest_area(predictions,img0,img)
        best_frame = {}
        if largest_attributes is not None: # If there is a prediction made
            score_best = largest_attributes["confidence"]*largest_attributes["largest_area"]
            image_selected = largest_attributes["image"]
            self.scores.append(score_best)
            self.images.append(image_selected)
            # if largest_attributes["class"]!=self.previous_tracked_class: # Reset the list if no prediction is made
            #     # print(f"Resetting the list, wrong class largest.")
            #     self.reset()
            #     self.previous_tracked_class = largest_attributes["class"]
        else:
            if self.verbose:
                self.logger.info("No prediction made.")
            self.reset()
        
        if len(self.scores)>self.frames_to_track: # Remove the first element of the list as if it was a queue.
            self.scores.pop(0) 
            self.images.pop(0) 
        if len(self.scores)==self.frames_to_track: # If the sequence is long enough evaluate 
            combined_score, best_index = self.evaluate()
            if combined_score>=self.threshold:# and combined_score>=self.best_score: # If the combined score is above the threshold and is better than the previous best score
                # print(f"Combined score: {combined_score}")
                if self.verbose:
                    self.logger.info(f"Combined object score over time {Fore.GREEN}above{Style.RESET_ALL}: {combined_score:.3f}/{self.threshold:.3f}")
                best_frame["score"] = combined_score
                img0_cut = self.images[best_index]
                best_frame["image"] = img0_cut
                # self.best_score = combined_score Previous version used the combined score as the best score
                self.best_score = self.scores[best_index]
                self.best_frame = img0_cut
                self.previous_tracked_class = largest_attributes["class"]
            else:
                if self.verbose:
                    self.logger.warning(f"Combined object score is {Fore.RED}below{Style.RESET_ALL}: {combined_score:.3f}/{self.threshold:.3f}")
                best_frame = None
        else:
            if self.verbose:
                self.logger.warning(f"Object sequence is {Fore.RED}not long enough{Style.RESET_ALL}: {len(self.scores)}/{self.frames_to_track}")
            best_frame = None

        return best_frame

    def evaluate(self)->Tuple[float,int]:
        """
        Evaluate the sequence of predictions.
        Args:
            None
        Returns:
            average: Average score of the sequence.
            best_index: Index of the best frame.
        """
        average = np.mean(self.scores)
        best_index = np.argmax(self.scores)
        return average, best_index
    def __len__(self):
        return len(self.scores)
    def reset(self):
        if self.verbose:
            self.logger.info("Resetting the lists.")
        self.scores = []
        self.images = []
        self.best_frame = None
        self.best_score = 0
        self.previous_tracked_class = None
        
        
def get_cut_out(image:np.ndarray,xyxy:tuple,offset:Union[int,list]=0)->np.ndarray:
    """
    Get a cut out of the image.
    Args:
        image: Image from which the cut out is made.
        xyxy: Bounding box coordinates.
        offset: Offset of the cut out. Either a list of (4 or 2) or an int.
    Returns:
        cut_out: Cut out of the image.
    """
    x1,y1,x2,y2 = xyxy
    if isinstance(offset,int):
        x1 = int(x1)-offset if int(x1)-offset>=0 else 0
        y1 = int(y1)-offset if int(y1)-offset>=0 else 0
        x2 = int(x2)+offset if int(x2)+offset<image.shape[1] else image.shape[1]
        y2 = int(y2)+offset if int(y2)+offset<image.shape[0] else image.shape[0]
    else:
        if len(offset)==2:
            offset.append(offset[0])
            offset.append(offset[1])
        x1 = int(x1)-offset[0] if int(x1)-offset[0]>=0 else 0
        y1 = int(y1)-offset[1] if int(y1)-offset[1]>=0 else 0
        x2 = int(x2)+offset[2] if int(x2)+offset[2]<image.shape[1] else image.shape[1]
        y2 = int(y2)+offset[3] if int(y2)+offset[3]<image.shape[0] else image.shape[0]

    cut_out = image[int(y1):int(y2),int(x1):int(x2),:]
    return cut_out

def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger

def create_logging_dir(run_name:str,log_dir:str,args)->str:
    """
    Create a directory for the logs.
    Args:
        run_name: Name of the run.
        log_dir: Directory where the logs are stored.
        args: Arguments of the run.
    Returns:
        log_dir: Directory where the logs are stored.
    """
    if run_name is None:
        run_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.join(log_dir, run_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # Write the arguments to a file
    with open(os.path.join(log_dir, 'args.yaml'), 'w') as f:
        yaml.dump(args, f)
    return log_dir
def scale_preds(preds:list[Union[torch.Tensor,np.ndarray]],img0:np.ndarray,img:torch.Tensor,filter:bool=False,classes_to_keep:list[int]=None)->np.ndarray:
    """
    Scale the predictions.
    Args:
        preds: Predictions of the current frame.
        img: Image predictions where made in.
        img0: Image of the current frame.
    Returns:
        preds: Scaled predictions.
    """
    if filter:
        assert classes_to_keep is not None, "classes_to_keep must be specified if filter is True"
    count = 0
    for i,pred in enumerate(preds):
        if len(pred): # if there is a detection
            pred[:,:4] = scale_coords(img.shape[2:], pred[:,:4], img0.shape).round()
            preds[i] = pred
            count += 1
    return preds#[:count]

class DigitPrediction:
    """
    Class to store separate predictions of the digit detector.
    Args:
        digit: Digit predicted.
        score: Score of the prediction.
        sequence_order: Order of the prediction in the sequence.
        xyxy: Bounding box coordinates.
        img_size: Size of the image.(height,width)
    """
    def __init__(self,
                 label:int,
                 sequence_order:int,
                 score:float,
                 xyxy:tuple[int,int,int,int],
                 img_size:tuple[int,int],
                 verbose:bool=False,
                 logger:logging.Logger=None,
                 ):
        self.label = label
        self.sequence_order = sequence_order
        self.score = score
        self.xyxy = xyxy
        self.img_size = img_size
        self.verbose = verbose
        self.logger = logger
        
        if verbose:
            assert logger is not None, "logger must be specified if verbose is True"
            logger.info(f"Created DigitPrediction with label: {label}, img_size: {img_size}")
        self.rel_xyxy = self.relative_coords() 
    def relative_coords(self):
        """
        Get the relative coordinates of the bounding box.
        Args:
            None
        Returns:
            relative_coords: Relative coordinates of the bounding box.
        """
        x1,y1,x2,y2 = self.xyxy
        img_h,img_w = self.img_size
        relative_coords = (float(x1)/img_w,float(y1)/img_h,float(x2)/img_w,float(y2)/img_h)
        return relative_coords
    @property
    def x1(self)->float:
        return self.rel_xyxy[0]
    @property
    def y1(self)->float:
        return self.rel_xyxy[1]
    @property
    def x2(self)->float:
        return self.rel_xyxy[2]
    @property
    def y2(self)->float:
        return self.rel_xyxy[3]
    @property
    def w(self)->float:
        return self.x2-self.x1
    @property
    def h(self)->float:
        return self.y2-self.y1
    @property
    def area(self)->float:
        return self.w*self.h
    @property
    def center(self)->tuple[float,float]:
        return ((self.x1+self.x2)/2,(self.y1+self.y2)/2)
    @property
    def ix1(self)->int:
        return int(self.xyxy[0])
    @property
    def iy1(self)->int:
        return int(self.xyxy[1])
    @property
    def ix2(self)->int:
        return int(self.xyxy[2])
    @property
    def iy2(self)->int:
        return int(self.xyxy[3])
    @property
    def ih(self)->int:
        return self.iy2-self.iy1
    @property
    def iw(self)->int:
        return self.ix2-self.ix1
    @property
    def iarea(self)->int:
        return (self.ix2-self.ix1)*(self.iy2-self.iy1)
    @property
    def icenter(self)->tuple[int,int]:
        return ((self.ix1+self.ix2)//2,(self.iy1+self.iy2)//2)

    def __repr__(self):
        return f"DigitPrediction(label={self.label},sequence_order={self.sequence_order},score={self.score},x1={self.x1},y1={self.y1},x2={self.x2},y2={self.y2},w={self.w},h={self.h},area={self.area},center={self.center})"
    def __str__(self) -> str:
        return f"Label: {self.label} @ {self.center}, Score: {self.score:.2f}"
class DigitSequence:
    """
    Class to store the prediction sequence of the digit detector.
    Args:
        predicitons: list of torch tensors or numpy arrays containing the predictions. Shape: (N,6) where N is number of predictions and 6 corresponds to (x1,y1,x2,y2,score,class).
        img_size: Size of the image the predictions were made in.
    """
    def __init__(self,predictions:list[Union[torch.Tensor,np.ndarray]],img_size:tuple[int,int]=(448,448),verbose:bool=False,logger:logging.Logger=None):
        self.predictions = predictions
        self.img_size = img_size
        self.verbose = verbose
        self.logger = logger
        self.digit_sequence:list[DigitPrediction] = self.generate_digit_sequence()
        if verbose:
            assert logger is not None, "logger must be specified if verbose is True"
            self.logger.info(f"Created DigitSequence with {len(self.digit_sequence)} digits. Image size: {self.img_size}")
        self.sort_by_x()
    @property
    def lbl_seq(self)->list[int]:
        """
        Get the sequence of the predictions.
        Args:
            None
        Returns:
            sequence: List of the predictions.
        """
        return [digit.label for digit in self.digit_sequence]
    @property
    def int_seq(self)->int:
        """
        Get the sequence of the predictions as an integer.
        Args:
            None
        Returns:
            integer: Integer representation of the sequence.
        """
        return int(self.str_seq)
    @property
    def str_seq(self)->str:
        """
        Get the sequence of the predictions as a string.
        Args:
            None
        Returns:
            sequence_str: String of the predictions.
        """
        return "".join([str(digit.label) for digit in self.digit_sequence])
    @property
    def min_score(self)->float:
        """
        Get the minimum score of the sequence.
        Args:
            None
        Returns:
            min_sequence_score: Minimum score of the sequence.
        """
        return min([digit.score for digit in self.digit_sequence])
    @property
    def score_seq(self):
        """
        Get the sequence of the scores.
        Args:
            None
        Returns:
            sequence: List of the scores.
        """
        return [digit.score for digit in self.digit_sequence]
    @property
    def ave_score(self)->float:
        """
        Get the average score of the sequence.
        Args:
            None
        Returns:
            average_sequence_score: Average score of the sequence.
        """
        return sum([digit.score for digit in self.digit_sequence])/len(self.digit_sequence)
    def generate_digit_sequence(self)->list[DigitPrediction]:
        """
        Generate the digit sequence.
        Args:
            None
        Returns:
            digit_predictions: List of DigitPrediction objects.
        """
        digit_predictions = []
        for i,pred in enumerate(self.predictions):
            if len(pred):
                if isinstance(pred,torch.Tensor):
                    pred = pred.cpu().numpy()
                for p in pred:
                    digit_predictions.append(DigitPrediction(label=int(p[5]),sequence_order=None,score=float(p[4]),xyxy=numpy_to_tuple(p[:4]),img_size=self.img_size,verbose=self.verbose,logger=self.logger))
        return digit_predictions
    def sort_by_x(self):
        """
        Sort the predictions by their x coordinate.
        Args:
            None
        Returns:
            None
        """
        self.digit_sequence.sort(key=lambda x: x.ix1)

    def __len__(self):
        return len(self.digit_sequence)
    def __repr__(self):
        s = [f"({digit.label},{digit.score:3f})" for digit in self.digit_sequence]
        return f"DigitSequence({s}->{self.str_seq})"
    # def __str__(self) -> str:
    #     return self.str_seq
class DigitPredictionTracker:
    """
    Class to track the predictions of digits.
    """
    def __init__(self,
                 frames_to_track:int=5,
                 img_size:tuple[int,int]=(448,448),
                 ind_threshold:float=0.5,
                 seq_threshold:float=0.3,
                 output_threshold:float=0.5,
                 visualize:bool=False,
                 list_of_combinations:list[DigitPrediction]=None,
                 verbose:bool=False,
                 logger:logging.Logger=None,
                 ):
        """
        Args:
            frames_to_track: Number of frames to track.
            threshold: Threshold for the combined score.
            visualize: bool, Visualize the best frame.
            list_of_combinations: List of combinations to select from.
        """
        self.verbose = verbose
        self.logger = logger
        self.img_size = img_size
        assert frames_to_track>0, "frames_to_track must be greater than 0."
        assert ind_threshold>=0, "ind_threshold must be greater than or equal to 0."
        assert seq_threshold>=0, "seq_threshold must be greater than or equal to 0."
        assert output_threshold>=0, "output_threshold must be greater than or equal to 0."
        if verbose:
            assert logger is not None, "If verbose is True, logger must be provided."
        self.frames_to_track = frames_to_track
        self.ind_threshold = ind_threshold
        self.seq_threshold = seq_threshold
        self.output_threshold = output_threshold
        self.visualize = visualize
        self.list_of_combinations = self.get_list_of_combinations(list_of_combinations)
        self.history:list[DigitSequence] = []
        self.best_sequence:DigitSequence = None
        self.best_score:float = 0
        self.best_frame:int = 0
        self.best_frame_img:np.ndarray = None
    def get_list_of_combinations(self,_list_of_combinations)->list[int]:
        """
        Get the list of combinations to select from.
        Args:
            list_of_combinations: List of combinations to select from.
        Returns:
            None
        """
        list_of_combinations = []
        if self.verbose:
            self.logger.info("Getting list of combinations...")
        for comb in _list_of_combinations:
            assert isinstance(comb,int) or isinstance(comb,str) and comb.isnumeric(),f"list_of_combinations must be a list of integers or strings of integers. Got {comb} of type {type(comb)}"
            if isinstance(comb,str):
                comb = comb.strip('-./ ')
            if comb not in list_of_combinations:
                list_of_combinations.append(str(comb))
            else:
                if self.verbose:
                    self.logger.warning(f"Combination {comb} already in list_of_combinations. Skipping...")
        list_of_combinations.sort()
        if self.verbose:
            self.logger.info(f"List of combinations: {list_of_combinations}")
        return list_of_combinations
    def reset(self):
        """
        Reset the tracker.
        Args:
            None
        Returns:
            None
        """
        if self.verbose:
            self.logger.info("Resetting tracker...")
        self.history = []
        self.best_sequence = None
        self.best_score = 0
        self.best_frame = 0
        self.best_frame_img = None
    def sort_history(self):
        """
        Sort the history by the minimum score.
        Args:
            None
        Returns:
            None
        """
        self.history.sort(key=lambda x: x.min_score)
    def validate_sequence(self,sequence:DigitSequence)->bool:
        """
        Evaluate the sequence.
        Args:
            sequence: DigitSequence object.
        Returns:
            valid: True if the sequence is valid and should therefore be added to history.
        """
        valid = True
        if not len(sequence):
            if self.verbose:
                self.logger.warning("Sequence is empty. Skipping...")
            return False
        # Check if sequence is in list of combinations:
        
        if sequence.str_seq not in self.list_of_combinations or not len(sequence):
            if self.verbose:
                self.logger.warning(f"Sequence: {sequence}, string: {sequence.str_seq} not in list of combinations.")
            valid = False# Must be valid sequence
        
        # Check if sequence is already in history must be the same sequence accross time:
        if sequence.int_seq not in [seq.int_seq for seq in self.history] and len(self.history)>0: # Check if not matching history sequence or if history is empty
            if self.verbose:
                self.logger.warning(f"Sequence: {sequence} did not match history.")
            valid = False # Must be valid sequence accross time
        
        if sequence.min_score < self.ind_threshold:
            if self.verbose:
                self.logger.warning(f"Sequence: {sequence} did not meet individual threshold.")
            valid = False # The minimum score must be above the threshold. Uncertainty in one digit is too high.
        
        if sequence.ave_score < self.seq_threshold:
            if self.verbose:
                self.logger.warning(f"Sequence: {sequence} did not meet sequence threshold.")
            valid = False # The average score must be above the threshold. Uncertainty in the sequence is too high.
        if valid and self.verbose:
            self.logger.info(f"Sequence: {sequence} is {Fore.GREEN}valid.{Style.RESET_ALL}")
        return valid

    def add_sequence(self,sequence:DigitSequence):
        """
        Add a sequence to the history.
        Args:
            sequence: DigitSequence object.
        Returns:
            None
        """
        if self.verbose:
            self.logger.info(f"Adding sequence: {sequence} to history...")
        self.history.append(sequence)
    def validate_history(self)->bool:
        """
        Validate the history.
        Args:
            None
        Returns:
            None
        """
        average_history_score = sum([seq.ave_score for seq in self.history])/len(self.history)
        if self.verbose:
            self.logger.info(f"Average history score: {average_history_score}")
        if average_history_score > self.output_threshold:
            if self.verbose:
                self.logger.info(f"Average history score: {average_history_score} is above threshold: {self.output_threshold}.-> {Fore.GREEN}Output{Style.RESET_ALL}")
            return True
        else:
            if self.verbose:
                self.logger.info(f"Average history score: {average_history_score} is below threshold: {self.output_threshold}.-> {Fore.RED}No output{Style.RESET_ALL}")
            return False
    def get_best_sequence(self)->DigitSequence:
        """
        Get the best sequence based on average score.
        Args:
            None
        Returns:
            best_sequence: DigitSequence object.
        """
        best_sequence = self.history[np.argmax([seq.ave_score for seq in self.history])]
        return best_sequence

        
        
        
    def update(self,predictions:list[Union[torch.Tensor,np.ndarray]],img:np.ndarray)->Tuple[DigitSequence,bool]:
        """
        Update the tracker.
        Args:
            predictions: List of predictions.
            img: Image the predictions were made in.
        Returns:
            None if the sequence did does not satisfy the requirements. Otherwise the sequence of numbers, i.e. the integer value of the sequence.
        """
        if self.verbose:
            self.logger.info("Updating tracker...")
        current_sequence = DigitSequence(predictions,img_size=self.img_size,verbose=self.verbose,logger=self.logger)
        if self.verbose:
            self.logger.info(f"Current sequence: {current_sequence}")
        valid = self.validate_sequence(current_sequence)
        if not valid:
            self.reset()
            return current_sequence,valid # Sequence is not valid.
        # If the average score of the history exceeds the threshold, we have an output sequence.
        # This means that the sequence must be valid over time. 
        # Criteria can be changed under validate_sequence for individual sequence added to history.
        # Criteria can be changed under validate_history for the history to be considered a valid output sequence.
        self.add_sequence(current_sequence)
        if len(self.history) > self.frames_to_track:
            self.history.pop(0)
        elif len(self.history) < self.frames_to_track:
            if self.verbose:
                self.logger.info(f"History is not long enough to be considered a valid output sequence.")
            return current_sequence, False # Not enough frames to make a prediction.
        valid = best = self.validate_history()
        if best:
            best_sequence = self.get_best_sequence()
            if self.verbose:
                self.logger.info(f"Best sequence score: {best_sequence.ave_score}")
            assert isinstance(best_sequence,DigitSequence), f"best_seq must be a DigitSequence object. Got {best_sequence} of type {type(best_sequence)}"
            self.reset()
            return best_sequence,best
        else:
            if self.verbose:
                self.logger.warning(f"Sequence {current_sequence} {Fore.RED}didn't pass{Style.RESET_ALL} -> score: {current_sequence.ave_score:.3f}/{self.output_threshold:.2f}")
            return current_sequence, valid
    def __repr__(self):
        return f"DigitPredictionTracker(\n\t\tFrames={self.frames_to_track},\n\t\tIndvidual Tresh={self.ind_threshold},\n\t\tSequence Threshold={self.seq_threshold},\n\t\tOutput Threshold={self.output_threshold},\n\t\tCombinations={self.list_of_combinations})"
class DigitDetector:
    """
    A class which digits in an image.
    Args:
        model: A model which takes an image as input and outputs a list of predictions.
        tracker: A tracker object which tracks the predictions over time.
        device: The device to run the model on.
        verbose: Whether to print information about the class.
    """
    def __init__(self,
                model:nn.Module,

                device:torch.device==torch.device("cuda:0"),
                img_size:tuple[int,int]=(448,448),
                verbose:bool=False,
                # Detection Thresholds
                iou_threshold:float=0.5,
                conf_threshold:float=0.5,
                nms_threshold:float=0.5,
                # DigitPredictionTracker Parameters
                frames_to_track:int=5,
                ind_threshold:float=0.5,
                seq_threshold:float=0.5,
                output_threshold:float=0.5,
                list_of_combinations:list[DigitPrediction]=None,
                logger:logging.Logger=None,
                visualize:bool=False,
                wait:bool=False,
                ):
        
        self.model = model
        self.img_size = img_size
        self.tracker = DigitPredictionTracker(frames_to_track=frames_to_track,
                                                img_size=img_size,
                                                ind_threshold=ind_threshold,
                                                seq_threshold=seq_threshold,
                                                output_threshold=output_threshold,
                                                list_of_combinations=list_of_combinations,
                                                visualize=visualize,
                                                verbose=verbose,
                                                logger=logger,
                                              )
        self.device = device
        self.verbose = verbose
        self.logger = logger
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.classes = [0,1,2,3,4,5,6,7,8,9]
        # self.reshaper = torchvision.transforms.Resize(img_size)
        self.visualize = visualize
        self.wait = wait
        assert self.device == self.model.device, f"Device of model and detector must be the same. Got {self.device} and {self.model.device}"
        if self.verbose:
            self.logger.info("Initializing DigitDetector...")
            self.logger.info(f"Model: {self.model.__class__.__name__}")
            self.logger.info(f"Tracker: {self.tracker}")
            self.logger.info(f"Device: {self.device}")
        
    def detect(self,img0:np.ndarray)->Union[None,DigitSequence]:
        """
        Detect digits in an image.
        Args:
            img: Image to detect digits in.
        Returns:
            None if no sequence was detected. Otherwise the sequence of numbers, i.e. the integer value of the sequence.
        """
        was_tensor = False
        if isinstance(img0,torch.Tensor):
            was_tensor = True
            device = img0.device
            img0 = img0.cpu().numpy().transpose(1,2,0) if len(img0.shape) == 3 else img0.cpu().numpy().transpose(0,2,3,1)[0]
        img0 = to_gray(img0)
        img0 = increase_contrast(img0)
        img = img0.copy()
        if self.verbose:
            self.logger.info("Detecting digits...")
        if isinstance(img,torch.Tensor):
            img = img.to(self.device) # Normalize
        else:
            img,_,_ = letterbox(img,self.img_size,auto=False)
            # Image.fromarray(img).show()
            img = torch.from_numpy(img).to(self.device)
        img = img.permute(2,0,1) # (H,W,C) -> (C,H,W)
        img = img.unsqueeze(0)/255 # (C,H,W) -> (1,C,H,W) && 255-0 -> 1.0-0.0 
        predictions = self.model(img)
        predictions = non_max_suppression(predictions,conf_thres=self.conf_threshold,iou_thres=self.iou_threshold,agnostic=True)
        predictions = [pred.detach().cpu().numpy() for pred in predictions]
        sequence, valid = self.tracker.update(predictions,img)
        if self.visualize:
            visualize_yolo_2D(img0=img0,pred=predictions,names=self.classes,rescale=True,img=img,wait=self.wait,image_name="Digit Detector")
        if self.verbose:
            self.logger.info(f"Sequence: {sequence} from {f'{Fore.GREEN}valid{Style.RESET_ALL}' if valid else f'{Fore.RED}invalid{Style.RESET_ALL}'} sequence.")
        return sequence, valid
    def __repr__(self):
        return f"DigitDetector(\n\t\tModel={self.model.__class__.__name__},\n\t\tDevice={self.device},\n\t\tImage size={self.img_size}\n\t\tTracker={self.tracker},\n\t\tVerbose={self.verbose})"
def numpy_to_tuple(arr:np.ndarray)->tuple:
    return tuple(arr.tolist())

def load_img(path:str)->np.ndarray:
    img = cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img
def to_gray(img:np.ndarray)->np.ndarray:
    """
    Make gray scale image with RGB values.
    Args:
        img: Image to make gray scale.
    Returns:
        Gray scale image.
    """
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    return img

def increase_contrast(img:np.ndarray)->np.ndarray:
    """
    Increase contrast of image.
    Args:
        img: Image to increase contrast of.
    Returns:
        Image with increased contrast.
    """
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    return img
def initialize_yolo_model(args):
    """
    Initialize YOLO model.
    Args:
        args: Arguments.
    Returns:
        YOLO model.
    """
    device = select_device(args.device)
    half = args.half if device.type != 'cpu' else False  # half precision only supported on CUDA
    model = DetectMultiBackend(args.weights, device=device, dnn=args.dnn, data=args.data, fp16=half)
    model.eval()
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = (args.imgsz, args.imgsz) if isinstance(args.imgsz, int) else args.imgsz  # tuple
    imgsz = check_img_size(imgsz=imgsz, s=stride)
    model.warmup(imgsz=(1 if pt else 1, 3, *imgsz))
    return model,imgsz,names
def initialize_network(args):
    """
    Initialize the network.
    Create live streaming object to stream data from the sensor.
    Create the detection model.
    Args:
        args: Arguments from the command line.
    """
    device = select_device(args.device)
    half = args.half if device.type != 'cpu' else False  # half precision only supported on CUDA
    model = DetectMultiBackend(args.weights, device=device, dnn=args.dnn, data=args.data, fp16=half)
    model.eval()
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = (args.imgsz, args.imgsz) if isinstance(args.imgsz, int) else args.imgsz  # tuple
    imgsz = check_img_size(imgsz=imgsz, s=stride)
    model.warmup(imgsz=(1 if pt else 1, 3, *imgsz))

    # Create file to save logs to.
    if args.save_time_log:
        dir_path = create_logging_dir(args.name_run,ROOT / "logs",args)
    else:
        dir_path = None
    # Horror video = https://www.youtube.com/watch?v=mto2mNFbrps&ab_channel=TromaMovies
    source = args.source if args.source is not None else "0" if args.webcam else "https://www.youtube.com/watch?v=mto2mNFbrps&ab_channel=TromaMovies"
    # Check if file
    if os.path.isfile(source) or os.path.isdir(source):
        live = LoadImages(path=source,img_size=imgsz,stride=stride,auto=args.auto)
    else:
        live = LoadStreams(sources=source,img_size=imgsz,stride=stride,auto=args.auto)

                                    
    log_file = None if not args.log_all else dir_path / "full_log.txt"

    logger = create_logger(log_file=log_file)
    pred_tracker = RegionPredictionsTracker(frames_to_track=args.object_frames,
                                      img_size=imgsz,
                                      threshold=args.tracker_thresh,
                                      visualize=args.visualize,
                                      class_to_track = args.class_to_track,
                                      verbose=args.verbose,
                                      logger=logger,)
    if args.transmit:
        transmitter = Transmitter(reciever_ip=args.ip, reciever_port=args.port)
        transmitter.start_transmit_udp()
        transmitter.start_transmit_ml()
    else:
        transmitter = None
    if args.log_time:
        time_logger = TimeLogger(logger,
                                args.disp_time,
                                save_log=args.save_time_log,
                                path=dir_path)
        initialize_timer(time_logger,args)
    else:
        time_logger = None
    return model, names, device, live, pred_tracker,transmitter, time_logger, logger
def initialize_timer(time_logger:TimeLogger,args,transmitter:Transmitter=None):
    """
    Args:
        time_logger: The logger object to log the time taken by various parts of the pipeline.
        args: Arguments from the command line.
        transmitter: If transmitter object is available then the time taken to transmit the data is also logged.
    """
    
    time_logger.create_metric("Pre Processing")
    time_logger.create_metric("Infrence")
    if args.cpu_post:
        time_logger.create_metric("Reloading CPU")
    time_logger.create_metric("Post Processing")
    if args.visualize:
        time_logger.create_metric("Visualize")
    if args.save_csv:
        time_logger.create_metric("Save CSV")
    if transmitter is not None:
        if transmitter.started_udp:
            time_logger.create_metric("Transmit TD")
        if transmitter.started_ml:
            time_logger.create_metric("Transmit UE5")
    if args.track:
        time_logger.create_metric("Tracking Frames")
    time_logger.create_metric("Internal Pipeline")
    time_logger.create_metric("Full Pipeline")

    return time_logger

def initialize_digit_model(args,logger=None):
    """
    Initialize the digit detection model with tracking.
    Args:
        args: Arguments from the command line.
    """
    device = select_device(args.device)
    half = args.half if device.type != 'cpu' else False  # half precision only supported on CUDA
    model = DetectMultiBackend(args.weights_digits, device=device, dnn=False, data=args.data_digit, fp16=half)
    model.eval()
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = (args.imgsz_digit, args.imgsz_digit) if isinstance(args.imgsz_digit, int) else args.imgsz_digit  # tuple
    imgsz = check_img_size(imgsz=imgsz, s=stride)
    model.warmup(imgsz=(1 if pt else 1, 3, *imgsz))
    dd = DigitDetector(model=model,
                        img_size=imgsz,
                        device=device,
                        verbose=args.verbose,
                        logger=logger,
                        iou_threshold=args.iou_digits,
                        frames_to_track=args.digit_frames,
                        conf_threshold=args.conf_digits,
                        ind_threshold=args.ind_thresh,
                        seq_threshold=args.seq_thresh,
                        output_threshold=args.out_thresh,
                        list_of_combinations=["082","095","1204","1206","1308","1404","1405","1407","1408","1501","1503","1506",
                                            "1508","1509","1510","1511","1516","1601","161","1602","162","163","164","1605","165","1606","166","1607","1608","1609","1610",
                                            "1611","1612","1613","1614","1615","1617","1619","1623","1625","1625","191","193","195","196","197","198","1910","2103","2104","2105","2106","2108"], # TODO: Add the list of combinations to be tracked as read from a file.
                        visualize=args.visualize,
                        wait=args.wait,)
    return dd


def norm_preds(pred,im0s):
    """
    Normalize predictions to image size.
    """
    for i,p in enumerate(pred):
        if p is not None:
            xyxy = p[:, :4]
            xyxy[:, 0] /= im0s.shape[1]
            xyxy[:, 1] /= im0s.shape[0]
            xyxy[:, 2] /= im0s.shape[1]
            xyxy[:, 3] /= im0s.shape[0]
            pred[i][:,:4] = xyxy
    return pred
def arguments():
    parser = argparse.ArgumentParser(description="Digit Detection")
    parser.add_argument("--weights_digits",type=str,default="./runs/train/GrayBabyElephant/weights/best.pt",help="Path to model")
    parser.add_argument("--test_img",type=str,help="Path to video")
    parser.add_argument("--output_digits",type=str,help="Path to output video")
    parser.add_argument("--iou_digits",type=float,default=0.1,help="IOU threshold")
    parser.add_argument("--conf_digits",type=float,default=0.1,help="Confidence threshold")
    parser.add_argument("--frames_to_track",type=int,default=5,help="Number of frames to track")
    parser.add_argument("--ind_thresh",type=float,default=0.5,help="Individual threshold")
    parser.add_argument("--seq_thresh",type=float,default=0.5,help="Sequence threshold")
    parser.add_argument("--out_thresh",type=float,default=0.5,help="Output threshold")
    parser.add_argument("--verbose",action="store_true",help="Whether to print information")
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference instead of FP32 (default)')
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')    
    parser.add_argument('--data', type=str, default=ROOT / './data/svhn.yaml', help='(optional) dataset.yaml path')

    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=448, help='inference size h,w')

    args = parser.parse_args()
    if isinstance(args.imgsz, list) and len(args.imgsz) == 1:
        args.imgsz = args.imgsz[0]
    return args
if __name__=="__main__":
    # Random numpy array of integers with 3 digits, i.e. 100-999.
    arr = np.random.randint(100,2000,size=(20)).tolist()
    # print(arr)
    args = arguments()
    model,device,img_size,names = initialize_digit_model(args)
    logger = create_logger()
    # dpt = DigitPredictionTracker(list_of_combinations=arr,verbose=args.verbose,logger=logger)
    dd = DigitDetector(
                        model=model,
                        img_size=img_size,
                        device=device,
                        verbose=args.verbose,
                        logger=logger,
                        iou_threshold=args.iou_digits,
                        frames_to_track=args.digit_frames,
                        conf_threshold=args.conf_digits,
                        ind_threshold=args.ind_thresh,
                        seq_threshold=args.seq_thresh,
                        output_threshold=args.out_thresh,
                        list_of_combinations=arr,
                       )
    img0 = load_img(args.test_img)
    img0 = to_gray(img0)
    img0 = increase_contrast(img0)
    t1 = time.time()
    sequence, valid = dd.detect(img0)
    t2 = time.time()
    print(f"Time taken: {t2-t1}")
    print(sequence,valid)
    t1 = time.time()
    sequence, valid = dd.detect(img0)
    t2 = time.time()
    print(f"Time taken: {t2-t1}")
    