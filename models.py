# Copyright 2019 UCLA Networked & Embedded Systems Laboratory (Author: Moustafa Alzantot)
#           2020 Sogang University Auditory Intelligence Laboratory (Author: Soonshin Seo) 
#
# MIT License


import math
import torch
import torch.nn.functional as F
from torch import nn 


class Residualblock(nn.Module):
    def __init__(self, in_depth, depth, first=False):
        super(Residualblock, self).__init__()
        self.first = first
        self.conv1 = nn.Conv2d(in_depth, depth, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(depth)
        self.lrelu = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(depth, depth, kernel_size=3, stride=3, padding=1)
        self.conv11 = nn.Conv2d(in_depth, depth, kernel_size=3, stride=3, padding=1)
        if not self.first :
            self.pre_bn = nn.BatchNorm2d(in_depth)

    def forward(self, x):
        # x is (B x d_in x T)
        prev = x
        prev_mp =  self.conv11(x)
        if not self.first:
            out = self.pre_bn(x)
            out = self.lrelu(out)
        else:
            out = x
        out = self.conv1(x)
        # out is (B x depth x T/2)
        out = self.bn1(out)
        out = self.lrelu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        # out is (B x depth x T/2)
        out = out + prev_mp
        return out
    
class AttentionBlock(nn.Module):
    def __init__(self):
        super(AttentionBlock, self).__init__()
    
    def forward(self, q, k, v, d_k):
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        score = F.softmax(score, dim=-1)
        score = torch.matmul(score, v)
        return score

class BaselineNetwork(nn.Module):
    def __init__(self, embedding_size):
        super(BaselineNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.block1 = Residualblock(32, 32,  True)
        self.mp = nn.MaxPool2d(3, stride=3, padding=1)
        self.block2 = Residualblock(32, 32,  False)
        self.block3 = Residualblock(32, 32,  False)
        self.block4= Residualblock(32, 32, False)
        self.block5= Residualblock(32, 32, False)
        self.block6 = Residualblock(32, 32, False)
        self.block7 = Residualblock(32, 32, False)
        self.block8 = Residualblock(32, 32, False)
        self.block9 = Residualblock(32, 32, False)
        self.lrelu = nn.LeakyReLU(0.01)
        self.bn = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.5)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc1 = nn.Linear(32, embedding_size)
        self.fc2 = nn.Linear(embedding_size, 2)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(dim=1)
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.mp(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.mp(out)
        out = self.block7(out)
        out = self.block8(out)
        out = self.block9(out)
        out = self.bn(out)
        out = self.lrelu(out)
        out = self.mp(out)
        out = out.view(batch_size, -1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.lrelu(out)
        out = self.fc2(out)
        out = self.logsoftmax(out)
        return out

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_size):
        super(SiameseNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.block1 = Residualblock(32, 32,  True)
        self.mp = nn.MaxPool2d(3, stride=3, padding=1)
        self.block2 = Residualblock(32, 32,  False)
        self.block3 = Residualblock(32, 32,  False)
        self.block4= Residualblock(32, 32, False)
        self.block5= Residualblock(32, 32, False)
        self.block6 = Residualblock(32, 32, False)
        self.block7 = Residualblock(32, 32, False)
        self.block8 = Residualblock(32, 32, False)
        self.block9 = Residualblock(32, 32, False)
        self.lrelu = nn.LeakyReLU(0.01)
        self.bn = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.5)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc1 = nn.Linear(32, embedding_size)
        self.fc2 = nn.Linear(embedding_size, 2)

    def forward_once(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(dim=1)
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.mp(out)
        out = self.block4(out) # cp-fine-tuning
        out = self.block5(out) # cp-fine-tuning
        out = self.block6(out) # cp-fine-tuning
        out = self.mp(out)
        out = self.block7(out)
        out = self.block8(out) # cp-fine-tuning
        out = self.block9(out)
        out = self.bn(out)
        out = self.lrelu(out)
        out = self.mp(out)
        out = out.view(batch_size, -1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.lrelu(out) # fine-tuning, cp-fine-tuning
        out = self.fc2(out) # fine-tuning, cp-fine-tuning
        out = self.logsoftmax(out) # fine-tuning, cp-fine-tuning
        return out
            
    def forward(self, x, x_pair):
        out = self.forward_once(x)
        out_pair = self.forward_once(x_pair)
        result = out
        #result = out - out_pair # fine-tuning, cp-fine-tuning
        return result
    
class TripletNetwork(nn.Module):
    def __init__(self, embedding_size):
        super(TripletNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.block1 = Residualblock(32, 32,  True)
        self.mp = nn.MaxPool2d(3, stride=3, padding=1)
        self.block2 = Residualblock(32, 32,  False)
        self.block3 = Residualblock(32, 32,  False)
        self.block4= Residualblock(32, 32, False)
        self.block5= Residualblock(32, 32, False)
        self.block6 = Residualblock(32, 32, False)
        self.block7 = Residualblock(32, 32, False)
        self.block8 = Residualblock(32, 32, False)
        self.block9 = Residualblock(32, 32, False)
        self.lrelu = nn.LeakyReLU(0.01)
        self.bn = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.5)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc1 = nn.Linear(32, embedding_size)
        self.fc2 = nn.Linear(embedding_size, 2)
        
    def forward_once(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(dim=1)
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.mp(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.mp(out)
        out = self.block7(out)
        out = self.block8(out)
        out = self.block9(out)
        out = self.bn(out)
        out = self.lrelu(out)
        out = self.mp(out)
        out = out.view(batch_size, -1)
        out = self.dropout(out)
        out = self.fc1(out)
        #out = self.lrelu(out)
        #out = self.fc2(out)
        #out = self.logsoftmax(out)
        return out
            
    def forward(self, x, x_pos, x_neg):
        out = self.forward_once(x)
        out_pos = self.forward_once(x_pos)
        out_neg = self.forward_once(x_neg)
        return out, out_pos, out_neg