import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import cv2
import os
import pickle

## data path
DATA_DIR = '../../data'
DATA_NAME = DATA_DIR + '/10vid'
DATA_FEAT = DATA_DIR + '/10vid-feature'
VIDEO_CAPS = DATA_DIR + '/msvd_video_caps.pkl'

FEAT_DIM = 2048

def load_caps(caps_path):
    # load video id and captions
    with open(caps_path, 'rb') as f:
        #video_ids, video_caps = pickle.load(f)
        rec = pickle.load(f)
        video_ids = set(rec[:,0]) # get only video id
    return video_ids
	
def create_resnet():
    # load pretrained model
    resnet152 = models.resnet152(pretrained=True)
    # remove the last layer
    modules=list(resnet152.children())[:-1]
    resnet152=nn.Sequential(*modules)
    for p in resnet152.parameters():
        p.requires_grad = False
    return resnet152
	
def video2tensor(video_path):
    # load avi file
    cap = cv2.VideoCapture(video_path)
    ret=True
    frames=[]
    while ret:
        ret, frame = cap.read()
        if ret==False:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resize_rgb = cv2.resize(rgb_frame, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
        # normalize to [0,1]
        norm_rgb = cv2.normalize(resize_rgb, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # transform data using (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) 
        frames.append(np.array([(norm_rgb[:,:,0]-0.485)/0.229, (norm_rgb[:,:,1]-0.456)/0.224, (norm_rgb[:,:,2]-0.406)/0.225]))
    
    frames_var = Variable(torch.tensor(frames)) # assign it to a variable
    return frames_var

def extract_feature_resnet(resnet152, video_path):
    frames_var = video2tensor(video_path)
    features_var = resnet152(frames_var) # get the output from the last hidden layer of the pretrained resnet
    features = features_var.data # get the tensor out of the variable
    return features
	

## Main program

# load pre-trained model
model = create_resnet()

# load video id from caption file
video_ids = load_caps(VIDEO_CAPS)

# extract feature for each video which is a batch of frames
for video_id in video_ids:
    video_path = DATA_NAME+'/'+video_id
    if os.path.isfile(video_path):
        feature = extract_feature_resnet(model, video_path)
        # mean pooling
        feature_mean = torch.mean(feature, dim=0)
        np.save(DATA_FEAT+"/"+video_id[:-4]+'.npy', feature_mean.data.numpy().reshape(FEAT_DIM))
    