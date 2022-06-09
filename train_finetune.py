import torch
import utils
import wandb
from train import train_main
#wandb.login()
weights = "" # Path to the weights to load
cfg = "" # Model yaml path
data = "" # Path to the data
hyp = "" # Hyperparameters path
epochs = 1 # Number of epochs to train
bs = 1 # Batch size
img_size = 640 # Image size
no_save = False # Don't save checkpoints only final
resume = False # Resume from most recent checkpoint
no_val = False # Disable validation, only final epoch
cache = True # Cache images in ram
device = "cuda" if torch.cuda.is_available() else "cpu" # Device
multi_scale = False # Multiple image sizes.
workers =  8 # Number of workers for dataloader
name = "exp" # Name of experiment
freeze = False # Freeze layers between layer.
save_period = 1 # Epochs between saving
