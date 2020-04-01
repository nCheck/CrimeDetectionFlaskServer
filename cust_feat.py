import os
import sys
import time
import argparse
import logging
import math
import gc
import decord
import json

import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet.gluon.data.vision import transforms
from gluoncv.data.transforms import video
from gluoncv.model_zoo import get_model
from gluoncv.data import VideoClsCustom


opt = {
    "data_dir" : '',
    "need_root" : False,
    "data_list" : "input.txt",
    "dtype" : 'float32',
    "gpu_id" : 0,
    "mode" : None ,
    "model" : 'i3d_resnet50_v1_kinetics400' ,
    "input_size" : 224 ,
    "use_pretrained" : True,
    "hashtag" : [],
    "resume_params" : None,
    "log_interval" : 10,
    "new_height" : 256,
    "new_width" : 340 ,
    "new_length" : 30 ,
    "new_step" : 1,
    "num_classes" : 400 ,
    "ten_crop" : False,
    "three_crop" : False,
    "video_loader" : True ,
    "use_decord" : True,
    "slowfast" : False ,
    "slow_temporal_stride" : 16 ,
    "fast_temporal_stride" : 2 ,
    "num_crop" : 1,
    "num_segments" : 32 ,
    "save_dir" : "./features",
    'skip_length' : 0
}


def sample_indices(num_frames):
    if num_frames > opt['skip_length'] - 1:
        tick = (num_frames - opt['skip_length'] + 1) / \
            float(opt['num_segments'])
        offsets = np.array([int(tick / 2.0 + tick * x)
                            for x in range(opt['num_segments'])])
    else:
        offsets = np.zeros((opt['num_segments'],))

    skip_offsets = np.zeros(opt['skip_length'] // opt['new_step'], dtype=int)
    return offsets + 1, skip_offsets


def video_TSN_decord_batch_loader(directory, video_reader, duration, indices, skip_offsets):
        sampled_list = []
        frame_id_list = []
        for seg_ind in indices:
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, opt['skip_length'], opt['new_step'])):
                if offset + skip_offsets[i] <= duration:
                    frame_id = offset + skip_offsets[i] - 1
                else:
                    frame_id = offset - 1
                frame_id_list.append(frame_id)
                if offset + opt['new_step'] < duration:
                    offset += opt['new_step']
        try:
            video_data = video_reader.get_batch(frame_id_list).asnumpy()
            sampled_list = [video_data[vid, :, :, :] for vid, _ in enumerate(frame_id_list)]
        except:
            raise RuntimeError('Error occured in reading frames {} from video {} of duration {}.'.format(frame_id_list, directory, duration))
        return sampled_list


def video_TSN_decord_slowfast_loader(directory, video_reader, duration, indices, skip_offsets):
    sampled_list = []
    frame_id_list = []
    for seg_ind in indices:
        fast_id_list = []
        slow_id_list = []
        offset = int(seg_ind)
        for i, _ in enumerate(range(0, opt['skip_length'], opt['new_step'])):
            if offset + skip_offsets[i] <= duration:
                frame_id = offset + skip_offsets[i] - 1
            else:
                frame_id = offset - 1

            if (i + 1) % opt.fast_temporal_stride == 0:
                fast_id_list.append(frame_id)

                if (i + 1) % opt['slow_temporal_stride'] == 0:
                    slow_id_list.append(frame_id)

            if offset + opt['new_step'] < duration:
                offset += opt['new_step']

        fast_id_list.extend(slow_id_list)
        frame_id_list.extend(fast_id_list)
    try:
        video_data = video_reader.get_batch(frame_id_list).asnumpy()
        sampled_list = [video_data[vid, :, :, :] for vid, _ in enumerate(frame_id_list)]
    except:
        raise RuntimeError('Error occured in reading frames {} from video {} of duration {}.'.format(frame_id_list, directory, duration))
    return sampled_list

def read_data(video_name, transform):

    decord_vr = decord.VideoReader(video_name, width=opt['new_width'], height=opt['new_height'])
    duration = len(decord_vr)

    opt['skip_length'] = opt['new_length'] * opt['new_step']
    segment_indices, skip_offsets = sample_indices(duration)

    if opt['video_loader']:
        if opt['slowfast']:
            clip_input = video_TSN_decord_slowfast_loader( video_name, decord_vr, duration, segment_indices, skip_offsets)
        else:
            clip_input = video_TSN_decord_batch_loader(video_name, decord_vr, duration, segment_indices, skip_offsets)

    clip_input = transform(clip_input)

    if opt['slowfast']:
        sparse_sampels = len(clip_input) // (opt['num_segments'] * opt['num_crop'])
        clip_input = np.stack(clip_input, axis=0)
        clip_input = clip_input.reshape((-1,) + (sparse_sampels, 3, opt['input_size'], opt['input_size']))
        clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))
    else:
        clip_input = np.stack(clip_input, axis=0)
        clip_input = clip_input.reshape((-1,) + (opt['new_length'], 3, opt['input_size'], opt['input_size']))
        clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))

    if opt['new_length'] == 1:
        clip_input = np.squeeze(clip_input, axis=2)    # this is for 2D input case

    return nd.array(clip_input)



def getModelForApp():

    print(opt)
    gc.set_threshold(100, 5, 5)

    if not os.path.exists(opt['save_dir']):
        os.makedirs(opt['save_dir'])

    # set env
    gpu_id = opt['gpu_id']
    context = mx.gpu(gpu_id)

    # get data preprocess
    image_norm_mean = [0.485, 0.456, 0.406]
    image_norm_std = [0.229, 0.224, 0.225]

    transform_test = video.VideoGroupValTransform(size=opt['input_size'], mean=image_norm_mean, std=image_norm_std)
    opt['num_crop'] = 1

    classes = opt['num_classes']
    model_name = opt['model']
    net = get_model(name=model_name, nclass=classes, pretrained=opt['use_pretrained'],
                    feat_ext=True, num_segments=opt['num_segments'], num_crop=opt['num_crop'])
    net.cast(opt['dtype'])
    net.collect_params().reset_ctx(context)

    print('Pre-trained model is successfully loaded from the model zoo.')
    print("Successfully built model {}".format(model_name))

    return net , transform_test , context , model_name




class MyFeatureExtract:

    def __init__(self):
        net , t_test , context , model_name = getModelForApp()
        self.net = net
        self.transform_test = t_test
        self.context = context
        self.model_name = model_name

    def predicter(self):
        # get data
        anno_file = opt['data_list']
        f = open(anno_file, 'r')
        data_list = f.readlines()
        print('Load %d video samples.' % len(data_list))

        start_time = time.time()
        for vid, vline in enumerate(data_list):
            video_path = vline.split()[0]
            video_name = video_path.split('/')[-1]
            if opt['need_root']:
                video_path = os.path.join(opt['data_dir'], video_path)
            video_data = read_data(video_path, self.transform_test)
            video_input = video_data.as_in_context(self.context)
            video_feat = self.net(video_input.astype(opt['dtype'], copy=False))

            feat_file = '%s_%s_feat.npy' % (self.model_name, video_name)
            np.save(os.path.join(opt['save_dir'], feat_file), video_feat.asnumpy())

            if vid > 0 and vid % opt['log_interval'] == 0:
                print('%04d/%04d is done' % (vid, len(data_list)))

        end_time = time.time()
        print('Total feature extraction time is %4.2f minutes' % ((end_time - start_time) / 60))