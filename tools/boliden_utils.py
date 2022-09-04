import os
import cv2
import time
import yaml
import torch
import logging
import numpy as np
import pandas as pd
from PIL import Image
from copy import copy
from datetime import datetime
import matplotlib.pyplot as plt
from colorama import Fore, Style
from utils.general import LOGGER, scale_coords
from utils.plots import Annotator, colors, save_one_box
from typing import Dict, Union, Tuple, Optional,Dict,TypeVar
argparse = TypeVar('argparse')
EPS = 1e-9
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


# def get_largest_area(pred:Union[np.ndarray,list],image_size:Tuple[int,int]=None)->Dict[Union[int,None],Union[int,None],Union[int,float]]:
#     """
#     From the predictions made. Return the largest area of the bounding box.
#     If image size is provided, area is proportional to the image size.
#     Args:
#         pred: Predictions from the model.
#         image_size: Image size.
#     Returns:
#         largest_attribute: Dictionary containing attributes of the largest bounding box.
#     """
#     largest_attributes ={
#         "index":None,
#         "confidence":None,
#         "largest_area":0,
#     }
#     for i,det in enumerate(pred):
#         if len(det):
#             for j,(*xyxy, conf, cls) in enumerate(det):
#                 if image_size is not None:
#                     area = (xyxy[2]-xyxy[0])*(xyxy[3]-xyxy[1])/(image_size[0]*image_size[1]) # Procentage of the image size, float
#                 else:
#                     area = (xyxy[2]-xyxy[0])*(xyxy[3]-xyxy[1]) # Absolute area int
#                 if area > largest_attributes["largest_area"]:
#                     largest_attributes["index"] = j
#                     largest_attributes["confidence"] = conf
#                     largest_attributes["largest_area"] = area
#     return largest_attributes

def visualize_yolo_2D(pred:np.ndarray,img0:np.ndarray,img:torch.Tensor,args:argparse,names:list[str]=None)->None:
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
    img0 = img0.copy()
    for i,det in enumerate(pred):
        
        #img0 = np.ascontiguousarray(copy(img).squeeze().permute(1,2,0).cpu().numpy())
        annotator = Annotator(img0, line_width=args.line_thickness, example=str(names))
        if len(det):
            #det[:,:4] = scale_coords(img.shape[2:], det[:,:4], img0.shape).round()
            
            i = 0
            for j,(*xyxy, conf, cls) in enumerate(det):#reversed(det)):
                c = int(cls)  # integer class
                label = None if args.hide_labels else (names[c] if args.hide_conf else f'{names[c]} {conf:.2f}')
                i += 1
                annotator.box_label(xyxy, label, color=colors(c, True))
            img0 = annotator.result()
            #img0 = cv2.cvtColor(img0,cv2.COLOR_RGB2BGR)
            cv2.imshow("Predictions",img0)
            cv2.waitKey(1)
        else:
            img0  = annotator.result()
            #img0 = cv2.cvtColor(img0,cv2.COLOR_RGB2BGR)
            cv2.imshow("Predictions",img0)
            cv2.waitKey(1)

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
    


class PredictionsTracker:
    """
    Track the 'n' previous frame's largest predictions.
    Calculates a score across the sequence of predictions made by calculating the certainty
    of the prediction and the size of the area.
    High certainty and large area = high score.
    If average score across the sequence is above a threshold, the prediction is considered a success and passed along the pipeline.\n
    Args:
        frames_to_track: number of frames to track.
        img_size: size of the image.
        threshold: When to be satisfied with the average score of a prediction sequence.
    """
    def __init__(self,
                    frames_to_track:int,
                    img_size:Union[tuple,int]=None,
                    img0_size:Union[tuple,int]=None,
                    threshold:float=0.5,
                    visualize:bool=False,
                ) -> None:
        assert frames_to_track > 0, "frames_to_track must be greater than 0."
        assert img_size is not None, "img_size must be specified."
        assert threshold <= 1, "threshold must be less or equal to 1."
        self.frames_to_track = frames_to_track
        self.img_size = img_size if isinstance(img_size,tuple) else (img_size,img_size) if img_size is not None else None
        self.img0_size = img0_size if isinstance(img0_size,tuple) else (img0_size,img0_size,3) if img0_size is not None else None
        self.threshold = threshold
        self.scores = []
        self.images = []
        self.best_frame = None
        self.best_score = 0
        self.previous_tracked_class = None
        self.visualize = visualize
    def get_largest_area(self,pred:Union[np.ndarray,list],img0:np.ndarray,img:torch.Tensor)->Union[Dict[str, Union[int,float,float,np.ndarray]],None]:
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
                    area = (xyxy[2]-xyxy[0])*(xyxy[3]-xyxy[1])/(self.img0_size[0]*self.img0_size[1]) # Procentage of the image size, float
                    if area > largest_attributes["largest_area"]:
                        largest_attributes["index"] = j
                        largest_attributes["confidence"] = float(conf)
                        largest_attributes["largest_area"] = float(area)
                        largest_attributes["image"] = get_cut_out(img0,xyxy)
                        largest_attributes["class"] = int(cls)
            else:
                return None
        return largest_attributes

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
            if largest_attributes["class"]!=self.previous_tracked_class: # Reset the list if no prediction is made
                # print(f"Resetting the list, wrong class largest.")
                self.reset()
                self.previous_tracked_class = largest_attributes["class"]
        else:
            self.reset()
        
        if len(self.scores)>self.frames_to_track: # Remove the first element of the list
            self.scores.pop(0) 
            self.images.pop(0) 
        if len(self.scores)==self.frames_to_track: # If the sequence is long enough evaluate 
            combined_score, best_index = self.evaluate()
            if combined_score>=self.threshold and combined_score>self.best_score: # If the combined score is above the threshold and is better than the previous best score
                best_frame["score"] = combined_score
                img0_cut = self.images[best_index]
                best_frame["image"] = img0_cut
                if self.visualize:      
                    cv2.imshow("Best frame",img0_cut)
                    cv2.putText(img0_cut,f"{combined_score:.2f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                    # Add text to the image with the score top left corner
                    cv2.waitKey(1)
                # self.best_score = combined_score Previous version used the combined score as the best score
                self.best_score = self.scores[best_index]
                self.best_frame = img0_cut
                self.previous_tracked_class = largest_attributes["class"]
                # im = Image.fromarray(cv2.cvtColor(img0_cut, cv2.COLOR_BGR2RGB))
                # im.show()
            else:
                best_frame = None
        else:
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
        self.scores = []
        self.images = []
        self.best_frame = None
        self.best_score = 0
        self.previous_tracked_class = None
        
        
def get_cut_out(image:np.ndarray,xyxy:tuple)->np.ndarray:
    """
    Get a cut out of the image.
    Args:
        image: Image from which the cut out is made.
        xyxy: Bounding box coordinates.
    Returns:
        cut_out: Cut out of the image.
    """
    x1,y1,x2,y2 = xyxy
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
def scale_preds(preds:np.ndarray,img0:np.ndarray,img:torch.Tensor,filter:bool=False,classes_to_keep:list[int]=None)->np.ndarray:
    """
    Scale the predictions.
    Args:
        preds: Predictions of the current frame.
        img: Image predictions where made in.
        img0: Image of the current frame.
    Returns:
        preds: Scaled predictions.


    TODO : Add the possibility to filter the predictions.
    """
    if filter:
        assert classes_to_keep is not None, "classes_to_keep must be specified if filter is True"
    count = 0
    for i,pred in enumerate(preds):
        if len(pred): # if there is a detection
            # if filter:
            #     if pred[5] not in classes_to_keep:
            #         continue
            pred[:,:4] = scale_coords(img.shape[2:], pred[:,:4], img0.shape).round()
            preds[i] = pred
            count += 1
    return preds#[:count]