import cv2
import json
import yaml
import os,sys
import argparse
import numpy as np
from typing import Union, Tuple
"""
    Converts SuperAnnotate JSON to YOLO format
    This includes training and validation sets, 
    as well as yaml file with classes and relative directories.
    Input directory should contain:
        - images folder
        - annotations file (json)
        - classes file (json)
        - config file (json)
    Output directory will contain:
        - train folder containing images (png) and labels (txt)
        - val folder containing images (png) and labels (txt)
        - data.yaml file with classes and relative directories
"""
def get_args():
    parser = argparse.ArgumentParser(description='Converts SuperAnnotate JSON to YOLO format')
    parser.add_argument('--input', type=str, default='input', help='input directory')
    parser.add_argument('--output', type=str, default='output', help='output folder')
    return parser.parse_args()

def get_classes(classes_file: str)->dict:
    """
        classes_file: 
            [
                {
                    "attribute_groups": list,
                    "color": hex,
                    "id": int, # Starts at 1
                    "name": str,
                    "opened": bool,
                },
            ]
    """
    classes = {}
    with open(classes_file) as f:
        data = json.load(f)
        n_classes = len(data)
        for class_ in data:
            classes[class_['id']-1] = class_['name']
    return classes, n_classes

def get_annotations(annotations_file: str)->dict:
    """
        annotations_file:
        {
            "image_name.png": {
                "instances": [
                    {
                        "type": str,
                        "classId": int,
                        "probability": int,
                        "points": {
                            "x1": float,
                            "x2": float,
                            "y1": float,
                            "y2": float
                        },
                        "groupId": int,
                        "pointLabels": {},
                        "locked": bool,
                        "visible": bool,
                        "attributes": []
                    },
                    ...
                ],
                "tags": [],
                "metadata": {
                    "version": "1.0.0",
                    "name": "image_name.png",
                    "status": "Completed"
                }
            },
        "image_name2.png": {...},
    """
    annotations = {}
    with open(annotations_file) as f:
        data = json.load(f)
        for annotation in data:
            annotations[annotation['image']] = annotation['instances']
    return annotations