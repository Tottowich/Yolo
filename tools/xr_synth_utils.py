import re
import glob
import math
import time
import torch
import open3d
import os, sys
import logging
import argparse
import numpy as np
import utils_ouster
import pandas as pd
from copy import copy
from queue import Queue
from pathlib import Path
from telnetlib import SE
import matplotlib.pyplot as plt
from datetime import datetime as dt
pd.options.display.float_format = '{:,.4e}'.format
sys.path.insert(0, '../../OusterTesting')
SENSOR_HEIGHT = 1.0
HARD_WIDTH = 0.5
HEAD_PROPORTION = 1/7
HEAD_FACTOR = 2.0
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
def proj_alt(pred,img0,xyz,R=25,azi=90,logger=None):
    #print(f"Image Shape: {img0.shape}")
    img_height = img0.shape[1]
    img_width = img0.shape[2]
    offset = 2
    #print(f"PCD Shape: {xyz.shape}")
    scale_y = xyz.shape[0]/img_height
    scale_x = xyz.shape[1]/img_width 
    obj_dim = [] # center_x, center_y, center_z, width, height, depth, rotation_x, rotation_y, rotation_z
    heights = []
    for det in pred[0]:
        xyxy = det[:4].cpu().numpy()
        x0,y0,x1,y1 = xyxy
        x0_scaled,y0_scaled,x1_scaled,y1_scaled = x0*scale_x,y0*scale_y,x1*scale_x,y1*scale_y
        pol_c_x,c_y = (x0+x1)/2,(y0+y1)/2
        head_size = (y1-y0)*HEAD_PROPORTION*HEAD_FACTOR
        chest_offset = ((y1-y0)/2-head_size) if c_y+((y1-y0)/2-head_size) < img_height else 0
        pol_c_y = c_y-chest_offset
        ix, iy = int(pol_c_x), int(pol_c_y)
        
        # Find region of interest, i.e. the chest.
        low_y,offset_low_y = [iy-offset,iy-offset] if iy-offset>=0 else [0,iy%offset]
        high_y,offset_high_y = [iy+offset,iy+offset] if iy+offset<img_height else [img_height-1, img_height-iy-1]
        low_x,offset_low_x =  [ix-offset,ix-offset] if ix-offset>=0 else [0,iy%offset]
        high_x,offset_high_x = [ix+offset,ix+offset] if ix+offset<img_width else [img_width-1,img_width-iy-1]

        roi = img0[2,low_y:high_y,low_x:high_x]
        poi_img = np.unravel_index(np.argmin(np.abs(np.median(roi)-roi)),roi.shape)
        
        poi_scan  = [int((poi_img[0]+int(offset_low_y))*scale_y), int((poi_img[1]+int(offset_low_x))*scale_x)]
        rot = -float(poi_scan[1])/xyz.shape[1]*2*np.pi
        
        center_x, center_y, center_z = xyz[poi_scan[0],poi_scan[1],:]
        vert_dist = np.sqrt(center_x**2+center_y**2)       
        cone_height = 2*np.tan(azi/2)*vert_dist     #     /|
        height_cov = (y1-y0)/img_height             #    / |
        height = height_cov*cone_height/2           # [c]  |Cone height
                                                    #  | \ |
                                                    #  |  \|
        z_offset = height*HEAD_PROPORTION*HEAD_FACTOR

        center_z = center_z-z_offset
        circle_diam = 2*np.pi*vert_dist
        width_cov = (x1-x0)/img_width
        temp_width = width_cov*circle_diam
        [width,breadth] = [temp_width,HARD_WIDTH] if temp_width>HARD_WIDTH else [HARD_WIDTH,temp_width]
        
        #2*(center_z+SENSOR_HEIGHT)
        obj_dim.append([center_x,center_y,center_z,breadth,width,height,rot,0,0])
    pred_dict = {"pred_boxes": np.array(obj_dim), "pred_scores": np.array(pred[0].cpu())[:,4], "pred_labels": np.array(pred[0].cpu())[:,5].astype(np.uint8)}

    # if len(obj_dim) > 0:
    #     pred_dict = {"pred_boxes": np.array(obj_dim), "pred_scores": np.array(pred[0].cpu())[:,4], "pred_labels": np.array(pred[0].cpu())[:,5].astype(np.uint8)}
    # else:
    #     pred_dict = {"pred_boxes": None, "pred_scores": None, "pred_labels": None}
    return pred_dict


def proj_and_format(pred,img0,scan,R=25,azi=90,logger=None):
    img_height = img0.shape[1]
    img_width = img0.shape[2]
    
    scale_y = scan.h/img_height
    scale_x = scan.w/img_width 
    
    # print(f"Image height: {img_height}")
    # print(f"Image width: {img_width}")
    to_rad = np.pi/180
    obj_dim = [] # center_x, center_y, center_z, width, height, depth, rotation_x, rotation_y, rotation_z
    heights = []
    
    
    for det in pred[0]:
        xyxy = det[:4].cpu().numpy()
        x0,y0,x1,y1 = xyxy
        

        pol_c_x,pol_c_y = (x0+x1)/2,(y0+y1)/2
        #pol_c_x = x0
        #pol_c_y = y0
        width = x1-x0
        height = y1-y0

        ix, iy = int(pol_c_x), int(pol_c_y)

        x0_scaled,y0_scaled,x1_scaled,y1_scaled = x0*scale_x,y0*scale_y,x1*scale_x,y1*scale_y
        #print(f"x0: {x0_scaled},y0: {y0_scaled},x1: {x1_scaled},y1: {y1_scaled}")
        low_y = iy-10 if iy-10>=0 else 0
        high_y = iy+10 if iy+10<img_height else img_height
        low_x = ix-10 if ix-10>=0 else 0
        high_x = ix+10 if ix+10<img_width else img_width
        #print(f"low_y: {low_y},high_y: {high_y},low_x: {low_x},high_x: {high_x}")
        r = np.median(img0[2,low_y:high_y,low_x:high_x].cpu().numpy())
        #print(f"r: {r*R}")
        theta = (y0_scaled+y1_scaled)/2 - scan.h/2
        vert_dist = r*R*np.cos(theta*to_rad)
        #print(f"pol_c_x: {pol_c_x}")
        center_angles = 2*np.pi*(pol_c_x/img_width)
        #print(center_angles)
        lower_edge_angle = 2*np.pi*(x0/img_width)
        upper_edge_angle = 2*np.pi*(x1/img_width)
        #print("angles: ",angles)
        center_x = -np.cos(center_angles)*vert_dist
        low_x_edge = np.cos(lower_edge_angle)*vert_dist
        high_x_edge = np.cos(upper_edge_angle)*vert_dist

        center_y = np.sin(center_angles)*vert_dist
        #low_y_edge = np.sin(lower_edge_angle)*vert_dist
        #high_y_edge = np.sin(upper_edge_angle)*vert_dist
        width = high_x_edge-low_x_edge

        #print(f"theta: {theta}")
        l = 2*np.tan(azi/2*to_rad)*vert_dist
        #print(f"l: {l}")
        det_h = l*(y1_scaled-y0_scaled)/scan.h
        center_z = det_h/2-SENSOR_HEIGHT
        rot = center_angles
        obj_dim.append([center_x,center_y,center_z,0.3,0.3,det_h,0,0,rot])
        #print(f"Det_h: {det_h}")
        heights.append([det_h,r*R])
    pred_dict = {"pred_boxes": np.array(obj_dim), "pred_scores": np.array(pred[0].cpu())[:,4], "pred_labels": np.array(pred[0].cpu())[:,5].astype(np.uint8)}

    # if len(obj_dim) > 0:
    #     pred_dict = {"pred_boxes": np.array(obj_dim), "pred_scores": np.array(pred[0].cpu())[:,4], "pred_labels": np.array(pred[0].cpu())[:,5].astype(np.uint8)}
    # else:
    #     pred_dict = {"pred_boxes": None, "pred_scores": None, "pred_labels": None}
    return pred_dict
def sorted_alphanumeric(data):
    """
    Sort the given iterable in the way that humans expect.
    Args:
        data: An iterable.
    Returns: sorted version of the given iterable.
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)
def filter_predictions(pred_dict, classes_to_use):
    """
    Filter predictions to only include the classes we want to use.
    """
    if isinstance(pred_dict["pred_labels"],torch.Tensor):
        pred_dict["pred_labels"] = pred_dict["pred_labels"].cpu().numpy().astype(int)
    if isinstance(pred_dict["pred_boxes"],torch.Tensor):
        pred_dict["pred_boxes"] = pred_dict["pred_boxes"].cpu().numpy()
    if isinstance(pred_dict["pred_scores"],torch.Tensor):
        pred_dict["pred_scores"] = pred_dict["pred_scores"].cpu().numpy()
    if classes_to_use is not None and len(pred_dict["pred_labels"]) > 0:
        
        #indices = [np.nonzero(sum(pred_dict["pred_labels"]==x for x in classes_to_use))[0].tolist()][0]
        #print(np.nonzero((sum(pred_dict["pred_labels"]-1==x for x in classes_to_use))))
        indices = np.nonzero((sum(pred_dict["pred_labels"]-1==x for x in classes_to_use)))[0].tolist()
        #print(f"indices: {indices}")
        pred_dict["pred_boxes"] = pred_dict["pred_boxes"].reshape(pred_dict["pred_boxes"].shape[0],-1)[indices,:]
        pred_dict["pred_labels"] = pred_dict["pred_labels"].reshape(pred_dict["pred_labels"].shape[0],-1)[indices,:]-1
        pred_dict["pred_scores"] = pred_dict["pred_scores"].reshape(pred_dict["pred_scores"].shape[0],-1)[indices,:]
    elif len(pred_dict["pred_labels"]) > 0:
        pred_dict["pred_boxes"] = pred_dict["pred_boxes"].reshape(pred_dict["pred_boxes"].shape[0],-1)
        pred_dict["pred_labels"] = pred_dict["pred_labels"].reshape(pred_dict["pred_labels"].shape[0],-1)-1
        pred_dict["pred_scores"] = pred_dict["pred_scores"].reshape(pred_dict["pred_scores"].shape[0],-1)
    return pred_dict
    
def generate_distance_matrix(pred_dict):
    if isinstance(pred_dict["pred_boxes"],torch.Tensor):
        pred_dict["pred_boxes"] = pred_dict["pred_boxes"].cpu().numpy()
    pred_dict["distance_matrix"] = np.zeros((pred_dict["pred_boxes"].shape[0],pred_dict["pred_boxes"].shape[0]))
    for i in range(pred_dict["pred_boxes"].shape[0]):
        for j in range(pred_dict["pred_boxes"].shape[0]):
            pred_dict["distance_matrix"][i,j] = np.linalg.norm(pred_dict["pred_boxes"][i,:3]-pred_dict["pred_boxes"][j,:3])
    return pred_dict

def get_xyz_from_predictions(pred_dict):
    """
    Get xyz coordinates from predictions.
    """
    
    return pred_dict

def format_predictions(pred_dict):
    """
    Format predictions to be more readable.
    """
    if isinstance(pred_dict["pred_labels"],torch.Tensor):
        pred_dict["pred_labels"] = pred_dict["pred_labels"].cpu().numpy().astype(int)
    if isinstance(pred_dict["pred_boxes"],torch.Tensor):
        pred_dict["pred_boxes"] = pred_dict["pred_boxes"].cpu().numpy()
    if isinstance(pred_dict["pred_scores"],torch.Tensor):
        pred_dict["pred_scores"] = pred_dict["pred_scores"].cpu().numpy()
    if len(pred_dict["pred_labels"]) > 0:
        pred_dict["pred_labels"] = pred_dict["pred_labels"]-1
    return pred_dict
def display_predictions(pred_dict, class_names, logger=None):
    """
    Display predictions.
    args:
        pred_dict: prediction dictionary. "pred_boxes", "pred_labels", "pred_scores"
        class_names: list of class names
    """
    if logger is None:
        return
    logger.info(f"Model detected: {len(pred_dict['pred_labels'])} objects.")
    for box,lbls,score in zip(pred_dict['pred_boxes'],pred_dict['pred_labels'],pred_dict['pred_scores']):
        if isinstance(lbls,list):
            lbls = lbls[0]
        if isinstance(lbls,list):
            box = box[0]
        if isinstance(lbls,list):
            score = score[0]
        logger.info(f"lbls: {lbls} score: {score}")
        logger.info(f"\t Prediciton {class_names[lbls]}, id: {lbls} with confidence: {score:.3e}.")
        logger.info(f"\t Box: {box}")
class CSVRecorder():
    """
    Class to record predictions and point clouds to a CSV file.
    """
    def __init__(self, 
                 folder_name=f"csv_folder_{dt.now().strftime('%Y%m%d_%H%M%S')}",
                 main_folder="./lidarCSV",
                 class_names=None,
                 ):
        self.main_folder = main_folder
        self.folder_name = folder_name
        self.class_names = class_names
        self.path = os.path.join(self.main_folder, self.folder_name)
        
        self.labelfile = "label"
        self.cloudfile = "cloud"
        if not os.path.exists(self.main_folder):
            os.makedirs(self.main_folder)
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.frames = 0
    def process_labels(self,pred_dict):
        boxes = np.array(pred_dict["pred_boxes"][:,:9])
        labels = np.array([self.class_names[int(x)] for x in pred_dict["pred_labels"]] if len(pred_dict["pred_labels"]) > 0 else []).reshape(-1,1)
        scores = np.array(pred_dict["pred_scores"]).reshape(-1,1)
        #print(f"boxes: {boxes}")
        #print(f"labels: {labels}")
        #print(f"scores: {scores}")

        labels = np.concatenate((boxes,labels,pred_dict["pred_labels"].reshape(-1,1),scores),axis=1)
        return labels

    def add_frame_file(self, cloud,pred_dict):
        cloud_name = os.path.join(self.path, f"cloud_{self.frames}.csv")
        label_name = os.path.join(self.path, f"label_{self.frames}.csv")
        np.savetxt(cloud_name, cloud, header = "x, y, z, r",delimiter=",")
        np.savetxt(label_name, self.process_labels(pred_dict=pred_dict), header = "x, y, z, rotx, roty, roz, l, w, h, label, label_idx, score",delimiter=",",fmt="%s")
        self.frames += 1

# class OusterStreamer():
#     def __init__(self, stream):
#         self.stream = stream
#         self.started = False
#         self.q_xyzr = Queue()

#     def start_thread(self):
#         self.started = True
#         self.thread = threading.Thread(target=self.stream_loop)
#         #self.thread.daemon = True
#         self.thread.start()
#         time.sleep(0.2)
#     def stop_thread(self):
#         self.started = False
#         self.thread.join()
#     def stream_loop(self):
#         for scan in self.stream:
#             if not self.started:
#                 break
#             xyz = utils_ouster.get_xyz(self.stream,scan)
#             signal = utils_ouster.get_signal_reflection(self.stream,scan)
#             xyzr = utils_ouster.convert_to_xyzr(xyz,signal)
#             self.q_xyzr.put(utils_ouster.compress_mid_dim(xyzr))
#     def get_pcd(self):
#         try:
#             return self.q_xyzr.get(timeout=1e-6)
#         except:
#             return None
class TimeLogger:
    """
    Class to log time.
    """
    def __init__(self,logger=None,disp_pred=False):
        """
        Args:
            logger: logger object to print the time.
            disp_pred: if True, display predictions.
        """
        super().__init__()
        self.time_dict = {}
        self.time_pd = None
        self.metrics_pd = None
        self.logger = logger
        if disp_pred is not None:
            self.print_log = disp_pred
        else:
            self.print_log = False

    def output_log(self,name:str):
        """
        Output the time taken at each step.
        Args:
            name: name of the step, i.e. pre_process, post_process, etc.
        """
        if self.logger is not None:
            self.logger.info(f"{name}: {self.time_dict[name]['times'][-1]:.3e} s <=> {1/self.time_dict[name]['times'][-1]:.3e} Hz")
        else:
            print(f"{name}: {self.time_dict[name]['times'][-1]:.3e} s <=> {1/self.time_dict[name]['times'][-1]:.3e} Hz")
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
    def visualize_results(self):
        time_averages = {}
        time_max = {}
        time_min = {}
        self.time_pd = {}
        sum_ave = 0
        keys = len(self.time_dict)
        
        fig,axs = plt.subplots(keys,1)
        for i,key in enumerate(self.time_dict):
           
            if len(self.time_dict[key]["times"])>0:
                axs[i].plot(self.time_dict[key]["times"],label=key)
                axs[i].set_title(key)
                time_averages[key] = np.mean(self.time_dict[key]["times"])
                time_max[key] = self.maximum_time(key)
                time_min[key] = self.minimum_time(key)
                sum_ave += time_averages[key] if key != "Full Pipeline" else 0
            #self.time_pd[key] = self.time_dict[key]["times"]
        plt.show()
        #self.time_pd = pd.DataFrame(self.time_pd)
        
        self.metrics_pd = pd.DataFrame([time_averages,time_max,time_min],index=["average","max","min"])
        if self.logger is not None:
            self.logger.info(f"Table To summarize:\n{self.metrics_pd}\nSum of parts: {sum_ave:.3e} <=> {1/sum_ave:.3e} Hz s\nLoading time: {self.metrics_pd['Full Pipeline']['average']-sum_ave:.3e} s\nTime to spare: {1/20-sum_ave:.3e}\nFrames per second: {1/self.metrics_pd['Full Pipeline']['average']:.3e} Hz")

        else:
            print(f"Table To summarize:\n{self.metrics_pd}")
if __name__ == "__main__":
    print("Hello World")
    T = TimeLogger()
    data = {'a': np.random.rand(10), 'b': np.random.rand(10), 'c': np.random.rand(10)}
    T.create_metric('a')
    T.create_metric('b')
    T.create_metric('c')
    for i in range(3):
        T.start('a')
        time.sleep(np.random.random())
        T.stop('a')
        T.start('b')
        time.sleep(np.random.random())
        T.stop('b')
        T.start('c')
        time.sleep(np.random.random())
        T.stop('c')

    T.visualize_results()