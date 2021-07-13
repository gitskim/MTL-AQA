# Author: Paritosh Parmar (https://github.com/ParitoshParmar)
# Code used in the following, also if you find it useful, please consider citing the following:
#
# @inproceedings{parmar2019and,
#   title={What and How Well You Performed? A Multitask Learning Approach to Action Quality Assessment},
#   author={Parmar, Paritosh and Tran Morris, Brendan},
#   booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
#   pages={304--313},
#   year={2019}
# }

import os
import torch
from torch.utils.data import DataLoader
from dataloaders.dataloader_C3DAVG import VideoDataset
import random
import scipy.stats as stats
import torch.optim as optim
import torch.nn as nn
from models.C3DAVG.C3D_altered import C3D_altered
from models.C3DAVG.my_fc6 import my_fc6
from models.C3DAVG.score_regressor import score_regressor
from models.C3DAVG.dive_classifier import dive_classifier
from models.C3DAVG.S2VTModel import S2VTModel
from opts import *
from utils import utils_1
import numpy as np
import streamlit as st
import subprocess
import os
import cv2 as cv
import tempfile
from torchvision import transforms

torch.manual_seed(randomseed);
torch.cuda.manual_seed_all(randomseed);
random.seed(randomseed);
np.random.seed(randomseed)
torch.backends.cudnn.deterministic = True


def save_model(model, model_name, epoch, path):
    model_path = os.path.join(path, '%s_%d.pth' % (model_name, epoch))
    torch.save(model.state_dict(), model_path)


def train_phase(train_dataloader, optimizer, criterions, epoch):
    criterion_final_score = criterions['criterion_final_score'];
    penalty_final_score = criterions['penalty_final_score']
    if with_dive_classification:
        criterion_dive_classifier = criterions['criterion_dive_classifier']
    if with_caption:
        criterion_caption = criterions['criterion_caption']

    model_CNN.train()
    model_my_fc6.train()
    model_score_regressor.train()
    if with_dive_classification:
        model_dive_classifier.train()
    if with_caption:
        model_caption.train()

    iteration = 0
    for data in train_dataloader:
        true_final_score = data['label_final_score'].unsqueeze_(1).type(torch.FloatTensor)
        if with_dive_classification:
            true_postion = data['label_position']
            true_armstand = data['label_armstand']
            true_rot_type = data['label_rot_type']
            true_ss_no = data['label_ss_no']
            true_tw_no = data['label_tw_no']
        if with_caption:
            true_captions = data['label_captions']
            true_captions_mask = data['label_captions_mask']
        video = data['video'].transpose_(1, 2)

        batch_size, C, frames, H, W = video.shape
        clip_feats = torch.Tensor([])

        for i in np.arange(0, frames - 17, 16):
            clip = video[:, :, i:i + 16, :, :]
            clip_feats_temp = model_CNN(clip)
            clip_feats_temp.unsqueeze_(0)
            clip_feats_temp.transpose_(0, 1)
            clip_feats = torch.cat((clip_feats, clip_feats_temp), 1)
        clip_feats_avg = clip_feats.mean(1)

        sample_feats_fc6 = model_my_fc6(clip_feats_avg)

        pred_final_score = model_score_regressor(sample_feats_fc6)
        if with_dive_classification:
            (pred_position, pred_armstand, pred_rot_type, pred_ss_no,
             pred_tw_no) = model_dive_classifier(sample_feats_fc6)
        if with_caption:
            seq_probs, _ = model_caption(clip_feats, true_captions, 'train')

        loss_final_score = (criterion_final_score(pred_final_score, true_final_score)
                            + penalty_final_score(pred_final_score, true_final_score))
        loss = 0
        loss += loss_final_score
        if with_dive_classification:
            loss_position = criterion_dive_classifier(pred_position, true_postion)
            loss_armstand = criterion_dive_classifier(pred_armstand, true_armstand)
            loss_rot_type = criterion_dive_classifier(pred_rot_type, true_rot_type)
            loss_ss_no = criterion_dive_classifier(pred_ss_no, true_ss_no)
            loss_tw_no = criterion_dive_classifier(pred_tw_no, true_tw_no)
            loss_cls = loss_position + loss_armstand + loss_rot_type + loss_ss_no + loss_tw_no
            loss += loss_cls
        if with_caption:
            loss_caption = criterion_caption(seq_probs, true_captions[:, 1:], true_captions_mask[:, 1:])
            loss += loss_caption * 0.01

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % 20 == 0:
            print('Epoch: ', epoch, ' Iter: ', iteration, ' Loss: ', loss, ' FS Loss: ', loss_final_score, end="")
            if with_dive_classification:
                print(' Cls Loss: ', loss_cls, end="")
            if with_caption:
                print(' Cap Loss: ', loss_caption, end="")
            print(' ')
        iteration += 1


def center_crop(img, dim):
  """Returns center cropped image

  Args:Image Scaling
  img: image to be center cropped
  dim: dimensions (width, height) to be cropped from center
  """
  width, height = img.shape[1], img.shape[0]
  #process crop width and height for max available dimension
  crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
  crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0]
  mid_x, mid_y = int(width/2), int(height/2)
  cw2, ch2 = int(crop_width/2), int(crop_height/2)
  crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
  return crop_img


if __name__ == '__main__':
    # loading the altered C3D backbone (ie C3D upto before fc-6)
    st.title("Olympics diving")
    video_file = st.file_uploader("Upload a video")

    # transforms.CenterCrop(H),

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        vf = cv.VideoCapture(tfile.name)

        # https: // discuss.streamlit.io / t / how - to - access - uploaded - video - in -streamlit - by - open - cv / 5831 / 8
        frames = None
        while vf.isOpened():
            ret, frame = vf.read()
            if not ret:
                break
            frame = cv.resize(frame, input_resize, interpolation=cv.INTER_LINEAR) #frame resized: (128, 171, 3)
            frame = center_crop(frame, (H, H))
            frame = transform(frame).unsqueeze(0)
            if frames is not None:
                frame = np.vstack((frames, frame))
                frames = frame
            else:
                frames = frame


        vf.release()
        cv.destroyAllWindows()
        rem = len(frames) % 16
        rem = 16 - rem

        if rem is not 0:
            padding = np.zeros((rem, C, H, H))
            print(padding.shape)
            frames = np.vstack((frames, padding))

        frames = np.expand_dims(frames, axis=0)
        print(f"frames shape: {frames.shape}")
        # frames shape: (137, 3, 112, 112)
        # suhyun frames : (137, 1080, 1920, 3)

        model_CNN_pretrained_dict = torch.load('c3d.pickle')
        model_CNN = C3D_altered()
        model_CNN_dict = model_CNN.state_dict()
        model_CNN_pretrained_dict = {k: v for k, v in model_CNN_pretrained_dict.items() if k in model_CNN_dict}
        model_CNN_dict.update(model_CNN_pretrained_dict)
        model_CNN.load_state_dict(model_CNN_dict)

        # loading our fc6 layer
        model_my_fc6 = my_fc6()
        model_my_fc6.load_state_dict(torch.load(m2_path, map_location={'cuda:0': 'cpu'}))

        # loading our score regressor
        model_score_regressor = score_regressor()
        model_score_regressor.load_state_dict(torch.load(m3_path, map_location={'cuda:0': 'cpu'}))
        print('Using Final Score Loss')

        frames = DataLoader(frames, batch_size=test_batch_size, shuffle=False)
        with torch.no_grad():
            pred_scores = [];
            # true_scores = []
            if with_dive_classification:
                pred_position = [];
                pred_armstand = [];
                pred_rot_type = [];
                pred_ss_no = [];
                pred_tw_no = []
                true_position = [];
                true_armstand = [];
                true_rot_type = [];
                true_ss_no = [];
                true_tw_no = []

            model_CNN.eval()
            model_my_fc6.eval()
            model_score_regressor.eval()

            # true_scores.extend(data['label_final_score'].data.numpy())

            # batch_size, C, frames, H, W = video.shape
            # frames shape: (137, 3, 112, 112)
            # frames shape: (1, 137, 3, 112, 112)
            for video in frames:
                print(f"video shape: {video.shape}")
                video = video.transpose_(1, 2)
                video = video.double()
                clip_feats = torch.Tensor([])
                for i in np.arange(0, len(video), 16):
                    print(i)
                    clip = video[i:i + 16, :, :, :]
                    print(f"clip shape: {clip.shape}")
                    print(f"clip type: {clip.type()}")
                    model_CNN = model_CNN.double()
                    clip_feats_temp = model_CNN(clip)
                    clip_feats_temp.unsqueeze_(0)
                    clip_feats_temp.transpose_(0, 1)
                    clip_feats = torch.cat((clip_feats, clip_feats_temp), 1)

                print(clip_feats)
                clip_feats_avg = clip_feats.mean(1)
                model_my_fc6 = model_my_fc6.double()
                sample_feats_fc6 = model_my_fc6(clip_feats_avg)
                model_score_regressor = model_score_regressor.double()
                temp_final_score = model_score_regressor(sample_feats_fc6)
                pred_scores.extend([element[0] for element in temp_final_score.data.cpu().numpy()])

                # rho, p = stats.spearmanr(pred_scores, true_scores)
                print('Predicted scores: ', pred_scores)
            # print('True scores: ', true_scores)
            # print('Correlation: ', rho)
