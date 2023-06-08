import numpy as np
import torch
from utils.general import scale_coords
from utils.plots import Annotator, colors
import cv2
def visualize_yolo_2D(pred:np.ndarray,
                    img0:np.ndarray,
                    args:"argparse"=None,
                    names:list[str]=None,
                    rescale:bool=False,
                    img:torch.Tensor=None,
                    line_thickness:int=1, 
                    hide_labels:bool=False, 
                    hide_conf:bool=False,
                    classes_not_to_show:list[int]=None,
                    image_name:str="Object Predicitions")->str:
    """
    Visualize the predictions.\n
    Args:
        pred: Predictions from the model.\n
        pred_dict: Dictionary of predictions.\n
        img0: Image collected from the camera. # Shape: (H,W,C).\n
        img: Image predictions where based upon. # Shape: (1,3,H,W).\n
        args: Arguments from the command line.\n
        names: Names of the classes.\n
        rescale: Rescale the predictions.\n
        line_thickness: Thickness of the bounding box.\n
        hide_labels: Hide the labels.\n
        hide_conf: Hide the confidence.\n
        classes_not_to_show: Classes not to show.\n
        image_name: Name of the image.\n
    Returns:
        class_string: String of predicted classes sorted by min x value (left->right in image).\n

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
    for i,det in enumerate(pred):
        annotator = Annotator(img0, line_width=line_thickness, example=str(names))
        if len(det):
            if rescale:
                det[:,:4] = scale_coords(img.shape[2:], det[:,:4], img0.shape).round()
            
            classes = []
            pos_x = []
            for *xyxy, conf, cls in det:
                c = int(cls)  # integer class
                if classes_not_to_show is not None and c in classes_not_to_show:
                    continue
                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                annotator.box_label(xyxy, label, color=colors(c, True))
            img0 = annotator.result()
            class_string = ""
            while len(classes)>0:
                id = np.argmin(pos_x)
                class_string += f"{names[classes[id]]}"
                pos_x.pop(id)
                classes.pop(id)
            # cv2.imshow(image_name,cv2.resize(img0,(640,int(640/img0.shape[1]*img0.shape[0]))))
            # cv2.waitKey(1)
            img0 = cv2.resize(img0,(640,int(640/img0.shape[1]*img0.shape[0])))
        else:
            img0  = annotator.result()
            # cv2.imshow(image_name,img0)
            # cv2.waitKey(1)
        return img0, class_string
