# from ast import Load
import yaml
import os,sys
import cv2
import re
import shutil
import torch
import random
import numpy as np
from tqdm import tqdm
from datetime import datetime as dt
from tools.arguments import parse_config
from typing import Callable
from ultralytics.yolo.utils.ops import scale_boxes
from tools.StreamLoader import LoadImages
from tools.boliden_utils import get_cut_out
from tools.boliden_utils import initialize_yolo_model, scale_preds, get_cut_out, visualize_yolo_2D,xyxy2xywh,to_gray,increase_contrast,norm_preds

TIMESTAMP = dt.now().strftime("%Y%m%d_%H%M%S")
class DataExtractor:
    """
    A Class which reads images, detects a certain class, extract prediction cut out, and saves cut out images to folder
    Args:
        model: Yolo model to make prediction.
        class_id: Id from which class to extract bounding boxes from.
        input_folder: Path to floder which might contain subfolders containing images.
        output_folder: Path where output should be stored.
    """
    def __init__(self, model, class_id, input_folder, output_folder):
        self.model = model
        self.class_id = class_id
        self.input_folder = input_folder
        self.output_folder = output_folder
        if isinstance(self.input_folder,str):
            self.input_folder = [self.input_folder]
        self.image_paths = []
        self.get_image_paths()
        self.images_to_extract = len(self.image_paths)
        self.count = 0
    
    def get_image_paths(self):
        """
        Get all image paths from input folder.
        """
        print("Getting image paths from input folder...")
        for input_folder in self.input_folder:
            for root, dirs, files in os.walk(input_folder):
                # print(files)
                for file in files:
                    if file.endswith(".jpg") or file.endswith(".png"):
                        self.image_paths.append(os.path.join(root, file))
    def save_image_to_dir(self,img,auto=False):
        """
        Save image to output folder.
        """
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        cv2.imwrite(os.path.join(self.output_folder, str(self.count)+".jpg"), img)
        self.count += 1
    def load_img(self, img_path):
        """
        Load image from path.
        """
        return cv2.imread(img_path)

    def extract(self):
        """
        Extract images from input folder.
        """
        print("Extracting images...")
        for image_path in tqdm(self.image_paths):
            img = self.load_img(image_path)
            self.save_image_to_dir(img)
        print("Done extracting images!")

class VerifyPredictions:
    """
    Using a Yolo model display predictions made and select wether to save image with prediction or to only save image and annotate later.
    
    Args:
        model: Yolo Model used to make predicitons.
        data: LoadImages.
        output_folder: Path where output should be stored.
    """
    def __init__(self, model, data:LoadImages, names:list, output_folder:str, count_auto_annotated=0,count_manual_annotated=0,skipped=0):
        self.model = model
        self.data = data
        self.names = names
        self.output_folder = output_folder
        self.count_auto_annotated = count_auto_annotated
        self.count_manual_annotated = count_manual_annotated
        self.skipped = skipped
        self.start = self.count_auto_annotated + self.count_manual_annotated + self.skipped
        self.data.start = self.start
        self.auto_name = "autoSchenk"
        self.manual_name = "manualSchenk"
        self.valid_list = [None]
        self.create_output_dirs()
    def create_output_dirs(self):
        """
        Create output directories.
        """
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        if not os.path.exists(os.path.join(self.output_folder,self.auto_name)):
            os.makedirs(os.path.join(self.output_folder,self.auto_name))
        if not os.path.exists(os.path.join(self.output_folder,self.manual_name)):
            os.makedirs(os.path.join(self.output_folder,self.manual_name))

    def save_image_to_dir(self,img,lbls:list[torch.Tensor],auto=False):
        """
        Save image to output folder.
        """
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        if auto:
            cv2.imwrite(os.path.join(self.output_folder,self.auto_name, str(TIMESTAMP)+"_"+str(self.count_auto_annotated)+".jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            with open(os.path.join(self.output_folder,self.auto_name,str(TIMESTAMP)+"_"+str(self.count_auto_annotated)+".txt"),"w") as f:
                for l in lbls:
                    x1,y1,x2,y2,conf,cls = l[:6]
                    x,y,w,h = xyxy2xywh((x1,y1,x2,y2))
                    f.write(" ".join(str(i) for i in [int(cls),x,y,w,h])+"\n")
            self.count_auto_annotated += 1
        else:
            cv2.imwrite(os.path.join(self.output_folder,self.manual_name, str(TIMESTAMP)+"_"+str(self.count_manual_annotated)+".jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            self.count_manual_annotated += 1
    def get_input(self):
        """
        Get input key press from user.
        """
        while True:
            key = cv2.waitKey(0)
            if key == ord("y"):
                return True
            elif key == ord("n"):
                return False
            elif key == ord("s"):
                return "skip"
            elif key == ord("q"):
                return "quit"
            else:
                print("Invalid key press, press 'y' to save image or 'n' to not save image labels.")
    def verify(self,pre_process:Callable=None):
        """
        Verify predictions made by model.
        """

        print("Verifying predictions...")
        pbar = tqdm(total=len(self.data))
        # Set pbar to start at current count
        pbar.update(self.start)
        winname = "Whole Image"
        cv2.namedWindow(winname)        # Create a named window
        cv2.moveWindow(winname, 40, 30)
        for path, img0, img, *_ in self.data:
            if pre_process:
                path, img0, img, *_ = pre_process(path, img0, img, *_)
            pbar.update(1)
            img0_shape = img0.shape[:-1]
            img_shape = img.shape[2:]
            # print(im0s.shape)
            results = self.model.predict(img,
                                    verbose=False,
                                    nms=True,
                                    conf=0.2,
                                    iou=0.01,
                                    max_det=6,
                                    imgsz=img_shape)[0].cpu().numpy() # Inference
            pred = results.boxes.data
            pred[:,:4] = scale_boxes(img_shape, pred[:,:4], img0_shape).round()            
            cv2.imshow(winname,cv2.resize(img0,(640,img0.shape[0]*640//img0.shape[1])))
            class_string = visualize_yolo_2D(pred, img0=img0, img=img,names=self.names)
            save = self.get_input()
            if save == "quit":
                print("Stopped @ Auto Annotated: {}. Manual Annotated: {}. Skipped {}".format(self.count_auto_annotated,self.count_manual_annotated,self.skipped))
                exit()
            elif save =="skip":
                self.skipped += 1
                continue
            pred = norm_preds(pred,img0)
            self.save_image_to_dir(img0,pred,save)
        print("Done verifying predictions!")
    def skip_or_false(self,false_prob:float):
        """
        Skip or return false based on probability.
        """
        det = random.random() < false_prob
        return False if det else "skip"
import random_word
RANDOM_WORD = random_word.RandomWords()
class DataSplitter:
    """
    Split data into train, val and test set.
    When completed dataset folder should contain:
    
    1. train/ Folder containing images (jpg) and labels (txt), images and corresponding label has same name.
    2. val/ - || - Val set.
    3. test/ - || - Test set.
    4. data.yaml. Yaml file containing: names: - list name of classes, 
                                        path: - path to dataset folder.
                                        train: - relative path to train folder, 
                                        val: - relative path to train folder, 
                                        test: - relative path to test folder.
                                        nc: - number of classes.
    Args: 
        input_folder: Folder containing images and labels.
        output_folder: Folder to save train, val and test set.
        train: Percentage of data to use for training.
        val: Percentage of data to use for validation.
        test: Percentage of data to use for testing.
    """
    def __init__(self,input_folder:str, output_folder:str, train:float,val:float,test:float) -> None:
        assert train+val+test == 1.0, "Train, val and test must add up to 1.0"
        self.batch_name = str(RANDOM_WORD.get_random_word())
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.train = train
        self.val = val
        self.test = test
        self.train_folder = os.path.join(self.output_folder,"train")
        self.val_folder = os.path.join(self.output_folder,"val")
        self.test_folder = os.path.join(self.output_folder,"test")
        self.data_yaml = os.path.join(self.output_folder,"data.yaml")
        self.data_paths = []
        self.names = ["0","1","2","3","4","5","6","7","8","9"]
        self.nc = len(self.names)
        self.create_folders()
        self.get_paths()
        # self.split_data()
    def create_folders(self,):
        """
        Create train, val and test folders.
        """
        if not os.path.exists(self.output_folder):
            os.mkdir(self.output_folder)
        if not os.path.exists(self.train_folder):
            os.makedirs(self.train_folder)
        if not os.path.exists(self.val_folder):
            os.makedirs(self.val_folder)
        if not os.path.exists(self.test_folder):
            os.makedirs(self.test_folder)
    def get_paths(self):
        """
        Get paths to images and labels. Store them in tuple.
        """
        for file in os.listdir(self.input_folder):
            print(file)
            if file.endswith(".jpg") or file.endswith(".png"):
                img_path = os.path.join(self.input_folder,file)
                label_path = os.path.join(self.input_folder,file.split(".")[0]+".txt")
                self.data_paths.append((img_path,label_path))
    def move_data(self,paths,folder):
        import time
        """
        Copy data to train, val or test folder.
        """
        print("Copying data to: {}".format(folder))
        for img_path, label_path in paths:
            print(f"Moving {img_path} and {label_path}")
            dest_path = os.path.join(folder,img_path.split("/")[-1].split(".")[0])
            shutil.move(img_path, dest_path+".jpg")
            shutil.move(label_path, dest_path+".txt")
            # lbl = np.loadtxt(label_path, delimiter=" ", dtype=np.float32)
            # if len(lbl)>0:
            #     np.savetxt(label_path, lbl, fmt="%d %f %f %f %f", delimiter=" ")
                # try:
                #     np.savetxt(label_path, lbl, fmt="%d %f %f %f %f", delimiter=" ")
                # except ValueError:
                #     # Fallback behavior when ValueError occurs
                #     # Load an empty file or handle the error gracefully
                #     with open(label_path, 'w') as file:
                #         # Write an empty string to the file
                #         file.write("")
    def reformat_data(self,paths):
        """
        Labels constist of cls,x,y,w,h in txt files. 
        """
        raise NotImplementedError
    def create_yaml(self):
        """
        Create yaml file containing dataset information.
        """
        data = {"names":self.names,
                "path":self.output_folder,
                "train":os.path.join(self.output_folder,"train"),
                "val":os.path.join(self.output_folder,"val"),
                "test":os.path.join(self.output_folder,"test"),
                "nc":self.nc}
        with open(self.data_yaml, "w") as outfile:
            yaml.dump(data, outfile, default_flow_style=False)
    def shuffle_data(self):
        """
        Shuffle data.
        """
        random.shuffle(self.data_paths)
    def split_data(self):
        """
        Split data into train, val and test set.
        """
        print(f"Splitting {len(self.data_paths)} images into train, val and test set...")
        train_len = int(len(self.data_paths)*self.train)
        val_len = int(len(self.data_paths)*self.val)
        test_len = int(len(self.data_paths)*self.test)
        train_paths = self.data_paths[:train_len]
        val_paths = self.data_paths[train_len:train_len+val_len]
        test_paths = self.data_paths[train_len+val_len:]
        print("Train: {}, Val: {}, Test: {}".format(len(train_paths),len(val_paths),len(test_paths)))
        self.move_data(train_paths,self.train_folder)
        self.move_data(val_paths,self.val_folder)
        self.move_data(test_paths,self.test_folder)
        # self.create_yaml()
        print("Done splitting data!")

        



with torch.no_grad():
    if __name__=="__main__":
        pass
        # args, data = parse_config()
        # model, imgsz, names = initialize_yolo_model(args,data)
        # data = LoadImages(args.source,imgsz=imgsz)
        # verify = VerifyPredictions(model,data,names,"/Users/theodorjonsson/GithubProjects/Adopticum/BolidenYolo/Datasets/Boliden/",count_auto_annotated=0,count_manual_annotated=0) # 247 141
        # def pre_process(path:str, img0:np.ndarray, img:torch.Tensor, *_):
        #     # Read YOLO format labels and extract bbox of class 1
        #     label_path = path.replace(".jpg", ".txt").replace(".png", ".txt")
        #     labels = np.loadtxt(label_path, delimiter=" ", dtype=np.float32).reshape(-1, 6)
        #     labels = labels[labels[:, 0] == 1]
        #     # Extract largest bbox
        #     if len(labels):
        #         largest = np.argmax(labels[:, 4]*labels[:, 5])
        #         labels = labels[largest]
        #         cls,*xyxy = labels

        # verify.verify()
        # data_splitter = DataSplitter("../datasets/Examples/Sequence_verify/autoV2/",
        # "../datasets/YoloFormat/BolidenDigits/",0.9,0.05,0.05)
        # data_splitter.create_folders()
        # data_splitter.get_paths()
        # data_splitter.split_data()
        # # count = 0
        
        # for path, img, im0s, _,_ in tqdm(data):

        #     img = torch.from_numpy(img).to(model.device)
        #     # cv2.imshow("img",cv2.cvtColor(im0s,cv2.COLOR_RGB2BGR))
        #     # cv2.waitKey(0)
        #     img = img.float()/255.0
        #     if img.ndimension() == 3:
        #         img = img.unsqueeze(0)
        #     pred = model(img)
        #     pred = non_max_suppression(pred, 0.25, 0.45, classes=[1], agnostic=False)
        #     pred = scale_preds(pred, im0s, img)
        #     for i, det in enumerate(pred):
        #         if det is not None and len(det):
        #             # det[:, :4] = scale_preds(det[:, :4], im0s.shape)
        #             for *xyxy, conf, cls in reversed(det):
        #                 cut_out = get_cut_out(im0s, xyxy, offset=30)
        #                 cv2.imwrite("../datasets/Examples/Sequence_cut_outs/"+str(count)+".jpg", cv2.cvtColor(cut_out, cv2.COLOR_RGB2BGR))
        #                 count += 1
