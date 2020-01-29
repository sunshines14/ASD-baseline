# Copyright 2019 UCLA Networked & Embedded Systems Laboratory (Author: Moustafa Alzantot)
#           2020 Sogang University Auditory Intelligence Laboratory (Author: Soonshin Seo) 
#
# MIT License


import os
import collections
import soundfile
import librosa
import h5py
import random
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from joblib import Parallel, delayed


# jobs
jobs = 8

# data_root 
logical_data_root = '/home/soonshin/sss/as/CORPUS/data_logical'
physical_data_root = '/home/soonshin/sss/as/CORPUS/data_physical'

# meta file
meta_file = collections.namedtuple('meta_file', ['spk_id', 'file_name', 'env_id', 'attack_id', 'key', 'path'])

# class
class Dataset(Dataset):
    # init
    def __init__(self, track=None, data=None, size=None, feature=None, tag=None):
        # 1) track
        if track == 'LA':
            track = 'LA'
            data_root = logical_data_root
        elif track == 'PA':
            track = 'PA'
            data_root = physical_data_root
            
        # 2) data
        if data == 'train':
            data = 'train'
            protocols_suffix = 'train.trn'
        elif data == 'dev':
            data = 'dev'
            protocols_suffix = 'dev.trl'
        elif data == 'eval':
            data = 'eval'
            protocols_suffix = 'eval.trl'    
            
        # +) protocols    
        protocols_prefix = 'ASVspoof2019_{}'.format(track)
        protocols_dir = os.path.join(data_root,'{}_protocols/'.format(protocols_prefix))
        self.protocols_file = os.path.join(protocols_dir, 'ASVspoof2019.{}.cm.{}.txt'.format(track, protocols_suffix))
        print('protocols file is ', self.protocols_file)
        
        # +) files
        self.files_dir = os.path.join(data_root, '{}_{}'.format(protocols_prefix, data), 'flac')
        print('files dir is ', self.files_dir)
        
        # +) attack id
        self.attack_id_dict = {
            '-': 0,    # bonafide
            'A01': 1,  # TTS neural waveform model 
            'A02': 2,  # TTS vocoder
            'A03': 3,  # TTS vocoder
            'A04': 4,  # TTS waveform concatenation
            'A05': 5,  # VC	vocoder
            'A06': 6,  # VC	spectral filtering
            'A07': 7,  # TTS vocoder+GAN
            'A08': 8,  # TTS neural waveform
            'A09': 9,  # TTS vocoder
            'A10': 10, # TTS neural waveform
            'A11': 11, # TTS griffin lim
            'A12': 12, # TTS neural waveform
            'A13': 13, # TTS_VC	waveform concatenation+waveform filtering
            'A14': 14, # TTS_VC	vocoder
            'A15': 15, # TTS_VC	neural waveform
            'A16': 16, # TTS waveform concatenation
            'A17': 17, # VC	waveform filtering
            'A18': 18, # VC	vocoder
            'A19': 19, # VC	spectral filtering
            
            # For PA:
            'AA': 20,  # 10-50 perfect
            'AB': 21,  # 10-50 high
            'AC': 22,  # 10-50 low
            'BA': 23,  # 50-100 perfect
            'BB': 24,  # 50-100 high
            'BC': 25,  # 50-100 low
            'CA': 26,  # > 100 perfect
            'CB': 27,  # > 100 high
            'CC': 28  # > 100 low
        }
        self.attack_id_dict_inv = {v:k for k,v in self.attack_id_dict.items()}
        
        # 3, 4, 5) size, feature, tag
        self.size = size
        self.feature = feature
        cache_file = 'features/cache_{}_{}_{}_{}_{}.npy'.format(track, data, size, feature, tag)
        print('cache_file is ', cache_file)
        if os.path.exists(cache_file):
            self.data_x, self.data_y, self.data_meta = torch.load(cache_file)
            #print('========== example ==========')
            #print(data_x[0], '\n', data_y[0], '\n', data_meta[0])
            #print('=============================')
            print('dataset loaded from ', cache_file)
        else:
            data_meta = self.parse_protocols_file()
            temp = list(map(self.read_file, data_meta))
            self.data_x, self.data_y, self.data_meta = map(list, zip(*temp))

            # +) transforms
            print('transforms ...')
            self.transforms = transforms.Compose([
                lambda x: self.pad(x),
                lambda x: librosa.util.normalize(x),
                lambda x: self.get_feature(x),
                lambda x: Tensor(x)])
            
            self.data_x = Parallel(n_jobs=jobs, prefer='threads')(delayed(self.transforms)(x) for x in self.data_x)
            torch.save((self.data_x, self.data_y, self.data_meta), cache_file)
            #print('========== example ==========')
            #print(data_x[0], '\n', data_y[0], '\n', data_meta[0])
            #print('=============================')
            print('dataset saved to ', cache_file)
        # +) len
        self.length = len(self.data_x)
        
    # function 1
    def __len__(self):
        return self.length
    
    # function 2
    def __getitem__(self, idx):
        # +) sample selection (pre-training)(train,eval) (fine-tuning)(eval) (cp-fine-tuning)(eval)
        x = self.data_x[idx]
        y = self.data_y[idx]
        return x, x, y, self.data_meta[idx]
    
        # +) sample selection (fine-tuning)(train) (cp-fine-tuning)(train)
        #x = self.data_x[idx]
        #y = self.data_y[idx]
        #randidx = int(random.randrange(self.length))
        #x_pair = self.data_x[randidx]
        #y_pair = self.data_y[randidx]
        #if y == y_pair:
        #    target = 1
        #else:
        #    target = 0
        #return x, x_pair, target, self.data_meta[idx]

        # +) sample selection (triplet)
        #x = self.data_x[idx]
        #y = self.data_y[idx]
        #while True:
        #    randidx = int(random.randrange(self.length))
        #    x_pos = self.data_x[randidx]
        #    y_pos = self.data_y[randidx]
        #    if y == y_pos:
        #        break
        #while True:
        #    randidx2 = int(random.randrange(self.length))
        #    x_neg = self.data_x[randidx2]
        #    y_neg = self.data_y[randidx2]
        #    if y != y_neg:
        #        break  
        #return x, x_pos, x_neg, self.data_meta[idx]
        
    
    # function 3     
    def parse_protocols_file(self):
        lines = open(self.protocols_file).readlines()
        data_meta = map(self.parse_line, lines)
        return list(data_meta)    

    # function 4
    def parse_line(self, lines):
        tokens = lines.strip().split(' ')
        return meta_file(
            spk_id = tokens[0],
            file_name = tokens[1],
            env_id = tokens[2],
            attack_id = self.attack_id_dict[tokens[3]],
            key = int(tokens[4] == 'bonafide'),
            path = os.path.join(self.files_dir, tokens[1] + '.flac'))
    
    # function 5
    def read_file(self, data_meta):
        data_x, sample_rate = soundfile.read(data_meta.path)
        data_y = data_meta.key
        return data_x, float(data_y), data_meta
    
    # function 6
    def pad(self, x):
        max_len = self.size
        x_len = x.shape[0]
        if x_len >= max_len:
            return x[:max_len]
        else:
            num_repeats = (max_len / x_len)+1
            x_repeat = np.repeat(x, num_repeats)
            padded_x = x_repeat[:max_len]
            return padded_x
    
    # function 7
    def get_feature(self, x):
        if self.feature == 'spect':
            s = librosa.core.stft(x, n_fft=2048, win_length=2048, hop_length=512)
            a = np.abs(s)**2
            #melspect = librosa.feature.melspectrogram(S=a)
            feats = librosa.power_to_db(a)
            return feats
        elif self.feature == 'mfcc':
            mfcc = librosa.feature.mfcc(x, sr=16000, n_mfcc=24)
            delta = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(delta)
            feats = np.concatenate((mfcc, delta, delta2), axis=0)
            return feats

#if __name__ == '__main__':
#    loader = Dataset(track='LA', data='dev', size=64000, feature='spect', tag=0)