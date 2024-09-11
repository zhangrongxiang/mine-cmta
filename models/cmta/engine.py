import os
import numpy as np
from tqdm import tqdm

from sksurv.metrics import concordance_index_censored
import torch.nn as nn
import torch.optim
import torch.nn.parallel
# from early_stopping import EarlyStopping

from hypll.optim import RiemannianAdam

class Engine(object):
    def __init__(self, args, results_dir, fold):
        self.args = args
        self.results_dir = results_dir
        self.fold = fold
        # tensorboard
        if args.log_data:
            from tensorboardX import SummaryWriter
            writer_dir = os.path.join(results_dir, 'fold_' + str(fold))
            if not os.path.isdir(writer_dir):
                os.mkdir(writer_dir)
            self.writer = SummaryWriter(writer_dir, flush_secs=15)
        self.best_score = 0
        self.best_epoch = 0
        self.filename_best = None

    def learning(self, model, train_loader, val_loader, criterion, optimizer, scheduler):
        if torch.cuda.is_available():
            model = model.cuda()

        if self.args.resume is not None:
            if os.path.isfile(self.args.resume):
                print("=> loading checkpoint '{}'".format(self.args.resume))
                checkpoint = torch.load(self.args.resume)
                self.best_score = checkpoint['best_score']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint (score: {})".format(checkpoint['best_score']))
            else:
                print("=> no checkpoint found at '{}'".format(self.args.resume))

        if self.args.evaluate:
            self.validate(val_loader, model, criterion)
            return

        for epoch in range(self.args.num_epoch):
            self.epoch = epoch
            # train for one epoch
            self.train(train_loader, model, criterion, optimizer)
            # evaluate on validation set
            c_index = self.validate(val_loader, model, criterion, self.args.modality)
            # remember best c-index and save checkpoint
            is_best = c_index > self.best_score
            if is_best:
                self.best_score = c_index
                self.best_epoch = self.epoch
                self.save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_score': self.best_score})
            print(' *** best c-index={:.4f} at epoch {}'.format(self.best_score, self.best_epoch))
            if scheduler is not None:
                scheduler.step()
            print('>')
        return self.best_score, self.best_epoch




    def train(self, data_loader, model, criterion, optimizer):
        model.train()
        train_loss = 0.0
        all_risk_scores = np.zeros((len(data_loader)))
        all_censorships = np.zeros((len(data_loader)))
        all_event_times = np.zeros((len(data_loader)))
        dataloader = tqdm(data_loader, desc='Train Epoch: {}'.format(self.epoch))
        for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time,
                        c) in enumerate(dataloader):

            if torch.cuda.is_available():
                data_WSI = data_WSI.cuda()
                data_omic1 = data_omic1.type(torch.FloatTensor).cuda()
                data_omic2 = data_omic2.type(torch.FloatTensor).cuda()
                data_omic3 = data_omic3.type(torch.FloatTensor).cuda()
                data_omic4 = data_omic4.type(torch.FloatTensor).cuda()
                data_omic5 = data_omic5.type(torch.FloatTensor).cuda()
                data_omic6 = data_omic6.type(torch.FloatTensor).cuda()
                label = label.type(torch.LongTensor).cuda()
                c = c.type(torch.FloatTensor).cuda()

            hazards, S, P, P_hat, G, G_hat,MLoss,fusion = model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2,
                                                   x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5,
                                                   x_omic6=data_omic6)

            # survival loss + sim loss + sim loss
            criterion_re=nn.MSELoss()
            # loss_re=criterion_re(fusion,(G + G_hat) / 2)
            sur_loss = criterion[0](hazards=hazards, S=S, Y=label, c=c)
            # sim_loss_P = criterion[1](P.detach(), P_hat)
            # sim_loss_G = criterion[1](G.detach(), G_hat)
            sim_loss= criterion[1](P, G)
            loss = sur_loss + self.args.alpha * sim_loss

            # if self.args.MoELoss:
            #     loss+=self.args.LossRate*MLoss
            # if self.args.ReLoss:
            #     loss+=self.args.LossRate*loss_re

            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.item()
            all_event_times[batch_idx] = event_time
            train_loss += loss.item()

            # # =======================================
            # euclidean_params = [p for name, p in model.named_parameters() if 'hyperbolic' not in name]
            # hyperbolic_params = [p for name, p in model.named_parameters() if 'hyperbolic' in name]
            # #
            # # # 定义优化器
            # optimizer_euclidean = torch.optim.SGD(filter(lambda p: p.requires_grad, euclidean_params), lr=self.args.lr, momentum=0.9, weight_decay=self.args.weight_decay)
            # optimizer_hyperbolic = RiemannianAdam(hyperbolic_params, lr=0.001)
            # # =======================================
            # loss.backward()
            #
            # optimizer_euclidean.step()
            # optimizer_euclidean.zero_grad()
            #
            # optimizer_hyperbolic.step()
            # optimizer_hyperbolic.zero_grad()

            euclidean_params = [p for name, p in model.named_parameters() if 'hyperbolic' not in name]
            hyperbolic_params = [p for name, p in model.named_parameters() if 'hyperbolic' in name]
            #
            # # 定义优化器
            optimizer_euclidean = torch.optim.SGD(filter(lambda p: p.requires_grad, euclidean_params), lr=self.args.lr,
                                                  momentum=0.9, weight_decay=self.args.weight_decay)
            optimizer_hyperbolic = RiemannianAdam(hyperbolic_params, lr=0.001)
            # =======================================
            loss.backward()
            optimizer_euclidean.step()
            optimizer_euclidean.zero_grad()

            optimizer_hyperbolic.step()
            optimizer_hyperbolic.zero_grad()
            # loss.backward()
            # optimizer.step()
            # optimizer.zero_grad()
        # calculate loss and error for epoch
        train_loss /= len(dataloader)
        c_index = concordance_index_censored((1 - all_censorships).astype(bool),
                                             all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        print('loss: {:.4f}, c_index: {:.4f}'.format(train_loss, c_index))

        if self.writer:
            self.writer.add_scalar('train/loss', train_loss, self.epoch)
            self.writer.add_scalar('train/c_index', c_index, self.epoch)

    def validate(self, data_loader, model, criterion,modality):
        model.eval()
        val_loss = 0.0
        all_risk_scores = np.zeros((len(data_loader)))
        all_censorships = np.zeros((len(data_loader)))
        all_event_times = np.zeros((len(data_loader)))
        dataloader = tqdm(data_loader, desc='Test Epoch: {}'.format(self.epoch))
        for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time,
                        c) in enumerate(dataloader):
            if torch.cuda.is_available():
                data_WSI = data_WSI.cuda()
                data_omic1 = data_omic1.type(torch.FloatTensor).cuda()
                data_omic2 = data_omic2.type(torch.FloatTensor).cuda()
                data_omic3 = data_omic3.type(torch.FloatTensor).cuda()
                data_omic4 = data_omic4.type(torch.FloatTensor).cuda()
                data_omic5 = data_omic5.type(torch.FloatTensor).cuda()
                data_omic6 = data_omic6.type(torch.FloatTensor).cuda()
                label = label.type(torch.LongTensor).cuda()
                c = c.type(torch.FloatTensor).cuda()
                if modality == 'Both':
                    pass
                if modality == 'G':
                    data_omic1 = torch.zeros_like(data_omic1).cuda()
                    data_omic2 = torch.zeros_like(data_omic2).cuda()
                    data_omic3 = torch.zeros_like(data_omic3).cuda()
                    data_omic4 = torch.zeros_like(data_omic4).cuda()
                    data_omic5 = torch.zeros_like(data_omic5).cuda()
                    data_omic6 = torch.zeros_like(data_omic6).cuda()
                if modality == 'P':
                    data_WSI = torch.zeros_like(data_WSI).cuda()



            with torch.no_grad():
                hazards, S, P, P_hat, G, G_hat ,MLoss,fusion= model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2,
                                                       x_omic3=data_omic3,
                                                       x_omic4=data_omic4, x_omic5=data_omic5,
                                                       x_omic6=data_omic6)  # return hazards, S, Y_hat, A_raw, results_dict

            # survival loss + sim loss + sim loss
            criterion_re=nn.MSELoss()
            # loss_re=criterion_re(fusion,(G + G_hat) / 2)
            sur_loss = criterion[0](hazards=hazards, S=S, Y=label, c=c)
            # sim_loss_P = criterion[1](P.detach(), P_hat)
            # sim_loss_G = criterion[1](G.detach(), G_hat)
            sim_loss= criterion[1](P, G)
            loss = sur_loss + self.args.alpha * sim_loss

            # if self.args.MoELoss:
            #     loss+=self.args.LossRate*MLoss
            # if self.args.ReLoss:
            #     loss+=self.args.LossRate*loss_re

            risk = -torch.sum(S, dim=1).cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.cpu().numpy()
            all_event_times[batch_idx] = event_time
            val_loss += loss.item()

        val_loss /= len(dataloader)
        c_index = concordance_index_censored((1 - all_censorships).astype(bool),
                                             all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        print('loss: {:.4f}, c_index: {:.4f}'.format(val_loss, c_index))
        if self.writer:
            self.writer.add_scalar('val/loss', val_loss, self.epoch)
            self.writer.add_scalar('val/c-index', c_index, self.epoch)
        return c_index


    def save_checkpoint(self, state):
        if self.filename_best is not None:
            os.remove(self.filename_best)
        self.filename_best = os.path.join(self.results_dir,
                                          'fold_' + str(self.fold),
                                          'model_best_{score:.4f}_{epoch}.pth.tar'.format(score=state['best_score'],
                                                                                          epoch=state['epoch']))
        print('save best model {filename}'.format(filename=self.filename_best))
        torch.save(state, self.filename_best)
