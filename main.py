# Copyright 2019 UCLA Networked & Embedded Systems Laboratory (Author: Moustafa Alzantot)
#           2020 Sogang University Auditory Intelligence Laboratory (Author: Soonshin Seo) 
#
# MIT License


import os
import sys
import argparse
import librosa
import data_utils
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn
from tensorboardX import SummaryWriter
from models import SiameseNetwork
from tools.pytorchtools import EarlyStopping
from torchsummary import summary


def train(train_loader, model, device, criterion, optim):
    batch_idx = 0
    num_total = 0
    running_loss = 0
    running_correct = 0
    epoch_loss = 0
    epoch_acc = 0
    # +) train mode (parallel) 
    if device == 'cuda':
        model = nn.DataParallel(model).train()
    else:
        model.train()
    for batch_x, batch_x_copy, batch_y, batch_meta in train_loader:
        batch_idx += 1
        num_total += batch_x.size(0)
        # +) wrapping
        batch_x = batch_x.to(device)
        batch_x_copy = batch_x_copy.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        # +) forward 
        batch_out = model(batch_x, batch_x_copy)
        _, batch_pred = batch_out.max(dim=1)
        batch_loss = criterion(batch_out, batch_y)
        # +) accmulate loss stats
        running_loss += (batch_loss.item()*batch_x.size(0))
        # +) accumulate accuracy stats
        running_correct += (batch_pred == batch_y).sum(dim=0).item()
        # +) print
        if batch_idx % 10 == 0:
            sys.stdout.write('\r \t {:.5f} {:.5f}'.format(running_correct/num_total, running_loss/num_total))
        # +) zero gradient
        optim.zero_grad()
        # +) backward
        batch_loss.backward()
        # +) update
        optim.step()
    epoch_loss = running_loss/num_total
    epoch_acc = running_correct/num_total
    return epoch_loss, epoch_acc

def dev(dev_loader, model, device, criterion):
    num_total = 0
    running_correct = 0
    running_loss = 0
    epoch_loss = 0
    epoch_acc = 0
    # +) dev mode						 
    model.eval()
    for batch_x, batch_x_copy, batch_y, batch_meta in dev_loader:
        num_total += batch_x.size(0)
        batch_x = batch_x.to(device)
        batch_x_copy = batch_x_copy.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x, batch_x_copy)
        _, batch_pred = batch_out.max(dim=1)
        batch_loss = criterion(batch_out, batch_y)
        running_correct += (batch_pred == batch_y).sum(dim=0).item()
        running_loss += (batch_loss.item()*batch_x.size(0))
    epoch_loss = running_loss/num_total
    epoch_acc = running_correct/num_total
    return epoch_loss, epoch_acc
            
def train_fit(train_loader, model, device, criterion, optim):
    batch_idx = 0
    num_total = 0
    running_loss = 0
    epoch_loss = 0
    # +) train mode (parallel) 
    if device == 'cuda':
        model = nn.DataParallel(model).train()
    else:
        model.train()
    for batch_x, batch_x_pair, batch_y, batch_meta in train_loader:
        batch_idx += 1
        num_total += batch_x.size(0)
        # +) wrapping
        batch_x = batch_x.to(device)
        batch_x_pair = batch_x_pair.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        # +) forward 
        batch_out = model(batch_x, batch_x_pair)
        batch_loss = criterion(batch_out, batch_y)
        # +) accmulate loss stats
        running_loss += (batch_loss.item()*batch_x.size(0))
        # +) print
        if batch_idx % 10 == 0:
            sys.stdout.write('\r \t {:.5f}'.format(running_loss/num_total))
        # +) zero gradient
        optim.zero_grad()
        # +) backward
        batch_loss.backward()
        # +) update
        optim.step()
    epoch_loss = running_loss/num_total
    return epoch_loss

def dev_fit(dev_loader, model, device, criterion):
    num_total = 0
    running_loss = 0
    epoch_loss = 0
    # +) dev mode						 
    model.eval()
    for batch_x, batch_x_pair, batch_y, batch_meta in dev_loader:
        num_total += batch_x.size(0)
        batch_x = batch_x.to(device)
        batch_x_pair = batch_x_pair.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x, batch_x_pair)
        batch_loss = criterion(batch_out, batch_y)
        running_loss += (batch_loss.item()*batch_x.size(0))
    epoch_loss = running_loss/num_total
    return epoch_loss

def evaluate(eval_dataset, eval_loader, model, device, eval_output_path):
    num_total = 0
    file_name_list = []
    key_list = []
    attack_id_list = []
    score_list = []
    # +) eval mode
    model.eval()
    for batch_x, batch_x_pair, batch_y, batch_meta in eval_loader:
        num_total += batch_x.size(0)
        batch_x = batch_x.to(device)
        batch_x_pair = batch_x_pair.to(device)
        batch_out = model(batch_x, batch_x_pair)
        # +) compute score
        batch_score = (batch_out[:,1] - batch_out[:,0]).data.cpu().numpy().ravel()
        # +) add outputs
        file_name_list.extend(list(batch_meta[1]))
        key_list.extend(['bonafide' if key == 1 else 'spoof' for key in list(batch_meta[4])])
        attack_id_list.extend([eval_dataset.attack_id_dict_inv[attack_id.item()] for attack_id in list(batch_meta[3])])
        score_list.extend(batch_score.tolist())
    # +) save result
    with open(eval_output_path, 'w') as f:
        for file_name, attack_id, key, score in zip(file_name_list, attack_id_list, key_list, score_list):
            f.write('{} {} {} {}\n'.format(file_name, attack_id, key, score))

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, result, target, size_average=True):
        distances = (-result).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()
    
class TripletLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

if __name__ == '__main__':
    # 1) parser
    parser = argparse.ArgumentParser()
    # +) For data preparation
    parser.add_argument('--track', type=str, default='LA') # LA, PA
    parser.add_argument('--input_size', type=int, default=64000) # input size (ex. 64000)
    parser.add_argument('--feature', type=str, default='mfcc') # spect, mfcc
    parser.add_argument('--data_tag', type=str, default=0) # feature tag (ex. 0)
    # +) For training
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--dev_batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--es_patience', type=int, default=7)
    parser.add_argument('--embedding_size', type=int, default=None)
    parser.add_argument('--model_comment', type=str, default=None)
    # +) For optimizer
    parser.add_argument('--loss', type=str, default='nll')
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--sched_factor', type=float, default=0.1)
    parser.add_argument('--sched_patience', type=int, default=10)
    parser.add_argument('--sched_min_lr', type=float, default=0)
    # +) For evaluation
    parser.add_argument('--eval_mode', action='store_true', default=False) 
    parser.add_argument('--eval_batch_size', type=int, default=None)
    parser.add_argument('--eval_num_checkpoint', type=int, default=None)
    # +) For Resume
    parser.add_argument('--resume_mode', action='store_true', default=False)
    parser.add_argument('--resume_num_checkpoint', type=int, default=None) 
    parser.add_argument('--fit_mode', action='store_true', default=False)
    parser.add_argument('--fit_num_checkpoint', type=int, default=None) 
    parser.add_argument('--cp_fit_mode', action='store_true', default=False)
    parser.add_argument('--cp_fit_num_checkpoint', type=int, default=None) 
    args = parser.parse_args()
	
    # 2) model tag
    model_tag = 'model_{}_{}_{}_{}_{}_{}_{}'.format(
            args.track, args.input_size, args.feature, args.data_tag, 
            args.train_batch_size, args.num_epochs, args.embedding_size)
    if args.model_comment:
        model_tag = model_tag + '_{}'.format(args.model_comment)
    print('model tag is ', model_tag)	
    
    # 3) model save path
    if args.fit_mode:  
        if not os.path.exists('models/tune'):
            os.mkdir('models/tune')
        model_save_path = os.path.join('models/tune', model_tag)
        if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)
        print('model save path is ', model_save_path)
    elif args.cp_fit_mode:  
        if not os.path.exists('models/cp-tune'):
            os.mkdir('models/cp-tune')
        model_save_path = os.path.join('models/cp-tune', model_tag)
        if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)
        print('model save path is ', model_save_path)
    else:
        if not os.path.exists('models/pre'):
            os.mkdir('models/pre')
        model_save_path = os.path.join('models/pre', model_tag)
        if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)
        print('model save path is ', model_save_path)
						
    # 4) use cuda
    if torch.cuda.is_available():
        device = 'cuda'
        print('device is ', device)
    else:
        device = 'cpu'
        print('device is ', device)
    
    # 5) eval
    if args.eval_mode:
        # +) eval dataset
        print('========== eval dataset ==========')
        eval_dataset = data_utils.Dataset(
                track=args.track, data='eval', size=args.input_size, feature=args.feature, tag=args.data_tag)
        eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=8)
        # +) load model
        print('========== eval process ==========')
        model = SiameseNetwork(args.embedding_size).to(device)
        eval_checkpoint_path = '{}/epoch_{}.pth'.format(model_save_path, str(args.eval_num_checkpoint))
        model.load_state_dict(torch.load(eval_checkpoint_path))
        print('model loaded from ', eval_checkpoint_path)
        # +) eval 
        eval_output_path = '{}/{}.result'.format(model_save_path, str(args.eval_num_checkpoint))
        evaluate(eval_dataset, eval_loader, model, device, eval_output_path)
        print('eval output saved to ', eval_output_path)
    
    # 6) train & dev
    else:
        # +) dev dataset
        print('========== dev dataset ==========')
        dev_dataset = data_utils.Dataset(
                track=args.track, data='dev', size=args.input_size, feature=args.feature, tag=args.data_tag)
        dev_loader = DataLoader(dev_dataset, batch_size=args.dev_batch_size, shuffle=True, num_workers=8)
        # +) train dataset
        print('========== train dataset ==========')
        train_dataset = data_utils.Dataset(
                track=args.track, data='train', size=args.input_size, feature=args.feature, tag=args.data_tag)
        train_loader = DataLoader(train_dataset, batch_size=args.dev_batch_size, shuffle=True, num_workers=8)
        print('========== train process ==========')
        # +) model init (check resume mode)
        if args.resume_mode:
            model = SiameseNetwork(args.embedding_size).to(device)
            resume_checkpoint_path = '{}/epoch_{}.pth'.format(model_save_path, str(args.resume_num_checkpoint))
            model.load_state_dict(torch.load(resume_checkpoint_path))
            print('model for resume loaded from ', resume_checkpoint_path)
            summary(model, input_size=[(1025,126), (1025,126)])
            start = args.resume_num_checkpoint+1
        # +) model init (check fine-tuning mode)    
        elif args.fit_mode:
            model = SiameseNetwork(args.embedding_size).to(device)
            fit_checkpoint_path = 'models/pre/{}/epoch_{}.pth'.format(model_tag, str(args.fit_num_checkpoint))
            model.load_state_dict(torch.load(fit_checkpoint_path))
            print('model for fit loaded from ', fit_checkpoint_path)
            summary(model, input_size=[(1025,126), (1025,126)])
            start = 1
        # +) model init (check cp-fine-tuning mode)     
        elif args.cp_fit_mode:
            model = SiameseNetwork(args.embedding_size).to(device)
            fit_checkpoint_path = 'models/tune/{}/epoch_{}.pth'.format(model_tag, str(args.cp_fit_num_checkpoint))
            model.load_state_dict(torch.load(fit_checkpoint_path))
            print('model for fit loaded from ', fit_checkpoint_path)
            summary(model, input_size=[(1025,126), (1025,126)])
            start = 1    
        # +) model init
        else:
            model = SiameseNetwork(args.embedding_size).to(device)
            summary(model, input_size=[(1025,126), (1025,126)])
            start = 1  
        # +) loss
        if args.loss == 'nll':
            weight = torch.FloatTensor([1.0, 9.0]).to(device) # weight for loss (spoof:1, genuine:9)
            criterion = nn.NLLLoss(weight=weight)
        elif args.loss == 'cs':
            criterion = ContrastiveLoss()
        elif args.loss == 'tri':
            criterion = TripletLoss()
        # +) optimizer
        if args.optim == 'adam':
            optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        # +) scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim, 'min', factor=args.sched_factor, patience=args.sched_patience, min_lr=args.sched_min_lr, verbose=False)
        # +) early stopping
        #early_stopping = EarlyStopping(patience=args.es_patience, verbose=False) 
        # +) fine-tuning mode
        if args.fit_mode:
            # +) tensorboardX, log
            if not os.path.exists('logs/tune'):
                os.mkdir('logs/tune')
            writer = SummaryWriter('logs/tune/{}'.format(model_tag))
            dev_losses = []
            for epoch in range(start, args.num_epochs+1):
                train_loss = train_fit(train_loader, model, device, criterion, optim)
                dev_loss = dev_fit(dev_loader, model, device, criterion)
                writer.add_scalar('train_loss', train_loss, epoch)
                writer.add_scalar('dev_loss', dev_loss, epoch)
                print('\n{} - train loss: {:.5f} - dev loss: {:.5f}'.format(epoch, train_loss, dev_loss))
                torch.save(model.state_dict(), os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch)))
                dev_losses.append(dev_loss)
                #early_stopping(dev_loss, model)
                #if early_stopping.early_stop:
                #    print('early stopping !')
                #    break
                scheduler.step(dev_loss)
            minposs = dev_losses.index(min(dev_losses))+1
            print('lowest dev loss at epoch is {}'.format(minposs))
        # +) cp-fine-tuning mode
        elif args.cp_fit_mode:
            # +) tensorboardX, log
            if not os.path.exists('logs/cp-tune'):
                os.mkdir('logs/cp-tune')
            writer = SummaryWriter('logs/cp-tune/{}'.format(model_tag))
            dev_losses = []
            for epoch in range(start, args.num_epochs+1):
                train_loss = train_fit(train_loader, model, device, criterion, optim)
                dev_loss = dev_fit(dev_loader, model, device, criterion)
                writer.add_scalar('train_loss', train_loss, epoch)
                writer.add_scalar('dev_loss', dev_loss, epoch)
                print('\n{} - train loss: {:.5f} - dev loss: {:.5f}'.format(epoch, train_loss, dev_loss))
                torch.save(model.state_dict(), os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch)))
                dev_losses.append(dev_loss)
                #early_stopping(dev_loss, model)
                #if early_stopping.early_stop:
                #    print('early stopping !')
                #    break
                scheduler.step(dev_loss)
            minposs = dev_losses.index(min(dev_losses))+1
            print('lowest dev loss at epoch is {}'.format(minposs))
        # +) pre-training mode
        else:
            # +) tensorboardX, log
            if not os.path.exists('logs/pre'):
                os.mkdir('logs/pre')
            writer = SummaryWriter('logs/pre/{}'.format(model_tag))
            dev_losses = []
            for epoch in range(start, args.num_epochs+1):
                train_loss, train_acc = train(train_loader, model, device, criterion, optim)
                dev_loss, dev_acc = dev(dev_loader, model, device, criterion)
                writer.add_scalar('train_acc', train_acc, epoch)
                writer.add_scalar('dev_acc', dev_acc, epoch)
                writer.add_scalar('train_loss', train_loss, epoch)
                writer.add_scalar('dev_loss', dev_loss, epoch)
                print('\n{} - train acc: {:.5f} - dev acc: {:.5f} - train loss: {:.5f} - dev loss: {:.5f}'.format(
                    epoch, train_acc, dev_acc, train_loss, dev_loss))
                torch.save(model.state_dict(), os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch)))
                dev_losses.append(dev_loss)
                #early_stopping(dev_loss, model)
                #if early_stopping.early_stop:
                #    print('early stopping !')
                #    break
                scheduler.step(dev_loss)
            minposs = dev_losses.index(min(dev_losses))+1
            print('lowest dev loss at epoch is {}'.format(minposs))