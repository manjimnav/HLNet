import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
from dataset import TimeDataset, UCRDataset
from utils import metric_regr, metric_cls, EarlyStopping
from models import HLNet, BaselineLSTM, BaselineFC, DACNet, HLInformer
from layers import GroupLayer
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from functools import partial
import os
import pickle


class Experimenter:
    def __init__(self, args):
        super(Experimenter, self)
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:0')
            print('Use GPU: cuda:0')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self):

        print(self.args.model)
        if self.args.model == 'HLNet':
            model = HLNet(
                emb_in=self.args.feature_len,
                seq_len=self.args.seq_len,
                pred_len=self.args.pred_len,
                emb_len=self.args.emb_len,
                d_model=self.args.d_model,
                dropout=self.args.dropout,
                group_factors=self.args.group_factors,
                group_operator=self.args.group_operator,
                group_step=self.args.group_step,
                has_minute=self.args.has_minute,
                has_hour=self.args.has_hour
            )
        elif self.args.model == 'DACNet':
            model = DACNet(
                emb_in=self.args.feature_len,
                pred_len=self.args.pred_len,
                emb_len=self.args.emb_len,
                d_model=self.args.d_model,
                dropout=self.args.dropout,
                has_minute=self.args.has_minute,
                has_hour=self.args.has_hour,
                k = self.args.k
            )
        elif self.args.model == 'LSTM':
            model = BaselineLSTM(
                emb_in=self.args.feature_len,
                seq_len=self.args.seq_len,
                pred_len=self.args.pred_len,
                emb_len=self.args.emb_len,
                d_model=self.args.d_model,
                dropout=self.args.dropout,
                n_layers=self.args.n_layers,
                has_minute=self.args.has_minute,
                has_hour=self.args.has_hour
            )
        elif self.args.model == 'FC':
            model = BaselineFC(
                emb_in=self.args.feature_len,
                pred_len=self.args.pred_len,
                seq_len = self.args.seq_len,
                emb_len=self.args.emb_len,
                d_model=self.args.d_model,
                dropout=self.args.dropout,
                n_layers=self.args.n_layers,
                has_minute=self.args.has_minute,
                has_hour=self.args.has_hour
            )
        elif self.args.model == 'HLInformer':
            model = HLInformer(
                enc_in=self.args.feature_len,
                dec_in=self.args.feature_len,
                c_out = 1,
                out_len=self.args.pred_len,
                d_model=self.args.d_model,
                dropout=self.args.dropout,
                e_layers=self.args.n_layers,
                d_layers=self.args.n_layers,
                has_minute=self.args.has_minute,
                has_hour=self.args.has_hour,
                group_factors=self.args.group_factors,
                group_operator=self.args.group_operator,
                group_step=self.args.group_step,
                attn = 'full'
            )
        else:
            raise Exception('Model nor found')
        
        print('Model params: '+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
        return model.double()

    def _get_data(self, split, scaler=None):
        args = self.args

        if split == 'test':
            shuffle_flag = False
        else:
            shuffle_flag = True

        if 'UCR' in args.data_path:
            data_set = UCRDataset(
            root_path=args.root_path,
            data_path=args.data_path,
            split=split,
            target=args.target,
            scaler=scaler,
            use_rolling=args.use_rolling,
            rolling=args.rolling
        )
        else:
             data_set = TimeDataset(
            root_path=args.root_path,
            data_path=args.data_path,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            split=split,
            step=args.step,
            has_minute=args.has_minute,
            has_hour=args.has_hour,
            target=args.target,
            scaler=scaler,
            use_rolling=args.use_rolling,
            rolling=args.rolling
        )
        print(split, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=self.args.batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=True)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def hierarchical_loss(self, y_pred, y_true, base_criterion=nn.MSELoss()):
        criterion = base_criterion
        total_epochs = self.args.train_epochs
        
        adaptative_epochs = int(0.25*total_epochs)
        
        head_step = (0.5-0.1)/adaptative_epochs
        
        head_beta = min(0.1 + head_step * self.current_epoch, 0.5)
        if len(self.args.group_factors) == 0:
            head_beta = 1
            
        y_pred, y_hierarchical = y_pred

        if 'UCR' not in self.args.data_path:
            true = y_true[:,-self.args.pred_len:]
        else:
            true = torch.tensor(y_true, dtype=torch.long)

        loss = criterion(y_pred.squeeze(), true)#*head_beta
        for i, (y_h_pred, group_factor) in enumerate(zip(y_hierarchical, self.args.group_factors)):
            gl = GroupLayer(group_factor, self.args.group_operator, self.args.group_step)
            if 'UCR' not in self.args.data_path:
                y_h_true = gl(y_true[:,(-self.args.pred_len-group_factor+1):].squeeze().transpose(0, 1)).transpose(0, 1).squeeze()
            else:
                y_h_true = true
            initial_betal = max(0.5, 0.99*(0.9**i))
            hierarchical_step = (initial_betal-0.5)/adaptative_epochs
            hierarchical_beta = max(initial_betal - hierarchical_step * self.current_epoch, 0.5)

            loss += criterion(y_h_pred.squeeze(), y_h_true.squeeze())#*hierarchical_beta
        loss = loss

        return loss

    def _select_criterion(self):
        if self.args.classification:
            base_criterion = nn.CrossEntropyLoss()
        else:
            base_criterion = nn.MSELoss()
            
        if self.args.model[:2] == 'HL':
            criterion = partial(self.hierarchical_loss, base_criterion=base_criterion)
        else:
            criterion = base_criterion
            
        return criterion

    def validate(self, vali_loader, criterion, train_data):
        self.model.eval()
        total_loss = []
        if 'UCR' in self.args.data_path:
            mean, std = 0, 1
        else:
            mean, std = train_data.target_scale_factors()

        for i, batch in enumerate(vali_loader):
            if 'UCR' in self.args.data_path:
                batch_x, batch_y = batch
                batch_x_mark, batch_y_mark = None,None
            else:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                batch_x_mark = batch_x_mark.double().to(self.device)
                batch_y_mark = batch_y_mark.double().to(self.device)
                
            batch_x = batch_x.double().to(self.device)
            batch_y = batch_y.double().to(self.device)


            if self.args.model == 'HLInformer':
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).double()

                dec_inp_mark = batch_y_mark[:,:,1:] 

                outputs = self.model(batch_x, batch_x_mark, dec_inp, dec_inp_mark)
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
            else:
                outputs = self.model(batch_x, batch_x_mark)
                
            if self.args.model[:2] == 'HL' and len(self.args.group_factors)>0 and 'UCR' not in self.args.data_path:
                batch_y = torch.cat((batch_x[:, (-max(self.args.group_factors)+1):, 0:1], batch_y), dim=1)
                
            if self.args.model[:2] != 'HL':
                pred = outputs.detach().cpu()*std + mean
            else:
                pred = (outputs[0].detach().cpu()*std + mean, [out.detach().cpu()*std + mean for out in outputs[1]])
            
            if self.args.classification:
                batch_y = torch.tensor(batch_y, dtype=torch.long)
                
            true = batch_y.detach().cpu().squeeze()
            
            true = true*std + mean

            loss = criterion(pred, true)

            total_loss.append(loss)

        total_loss = np.average(total_loss)

        self.model.train()
        return total_loss

    
    def plot_grad_flow(self):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.
        
        Usage: Plug this function in Trainer class after loss.backwards() as 
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        named_parameters = self.model.named_parameters()
        ave_grads = []
        max_grads= []
        layers = []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

    def train(self, setting):
        train_data, train_loader = self._get_data(split='train')
        vali_data, vali_loader = self._get_data(split='valid', scaler=train_data.scaler)
        test_data, test_loader = self._get_data(split='test', scaler=train_data.scaler)

        path = './checkpoints/' + setting
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        train_loss_epochs = []
        valid_loss_epochs = []
        test_loss_epochs = []

        for epoch in range(self.args.train_epochs):
            self.current_epoch = epoch
            iter_count = 0
            train_loss = []

            self.model.train()
            for i, batch in enumerate(train_loader):
                iter_count += 1
                
                if 'UCR' in self.args.data_path:
                    batch_x, batch_y = batch
                    batch_x_mark, batch_y_mark = None,None
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                    batch_x_mark = batch_x_mark.double().to(self.device)
                    batch_y_mark = batch_y_mark.double().to(self.device)

                model_optim.zero_grad()

                batch_x = batch_x.double().to(self.device)
                batch_y = batch_y.double().to(self.device)


                if self.args.model == 'HLInformer':
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).double()
                    #dec_inp = torch.cat([batch_x[:, -self.args.label_len:, :], dec_inp], dim=1).double().to(self.device)
                    dec_inp_mark = batch_y_mark[:,:,1:] #torch.cat([batch_x_mark[:, -self.args.label_len:, :], batch_y_mark[:,:,1:]], dim=1).double().to(self.device)
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, dec_inp_mark)
                    batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                else:
                    outputs = self.model(batch_x, batch_x_mark)
                    
                if self.args.model[:2] == 'HL' and len(self.args.group_factors)>0 and 'UCR' not in self.args.data_path:
                    batch_y = torch.cat((batch_x[:, (-max(self.args.group_factors)+1):, 0:1], batch_y), dim=1)

                if self.args.classification:
                    batch_y = torch.tensor(batch_y, dtype=torch.long)
                loss = criterion(outputs, batch_y.squeeze())
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()
                if self.args.plot_gradients:
                    self.plot_grad_flow()

            train_loss = np.average(train_loss)
            vali_loss = self.validate(vali_loader, criterion, train_data)
            test_loss = self.validate(test_loader, criterion, train_data)

            train_loss_epochs.append(train_loss)
            valid_loss_epochs.append(vali_loss)
            test_loss_epochs.append(test_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                train_loss_epochs = np.array(train_loss_epochs)
                valid_loss_epochs = np.array(valid_loss_epochs)
                test_loss_epochs = np.array(test_loss_epochs)

                np.save(path + '/train_history.npy', train_loss_epochs)
                np.save(path + '/valid_history.npy', valid_loss_epochs)
                np.save(path + '/test_history.npy', test_loss_epochs)

                break

        best_model_path = path + '/' + 'checkpoint.pth'.format(epoch)
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting):
        train_data, train_loader = self._get_data(split='train')
        test_data, test_loader = self._get_data(split='test', scaler=train_data.scaler)

        self.model.eval()

        preds = []
        trues = []
        times = []
        hierarchical_preds = []
        
        if 'UCR' in self.args.data_path:
            mean, std = 0, 1
        else:
            mean, std = train_data.target_scale_factors()
        
        for i, batch in enumerate(test_loader):
            if 'UCR' in self.args.data_path:
                batch_x, batch_y = batch
                batch_x_mark, batch_y_mark = None,None
            else:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                batch_x_mark = batch_x_mark.double().to(self.device)
                batch_y_mark = batch_y_mark.double().to(self.device)
            
            batch_x = batch_x.double().to(self.device)
            batch_y = batch_y.double().to(self.device)
            
            if self.args.model == 'HLInformer':
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).double()

                dec_inp_mark = batch_y_mark[:,:,1:] 
                
                outputs = self.model(batch_x, batch_x_mark, dec_inp, dec_inp_mark)
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
            else:
                outputs = self.model(batch_x, batch_x_mark)

            if self.args.model[:2] != 'HL':
                pred = outputs.detach().cpu()
                pred = pred*std + mean
                preds.append(pred.numpy())
            else:
                pred = (outputs[0].detach().cpu()*std + mean, [out.detach().cpu().numpy().squeeze()*std + mean for out in outputs[1]])
                preds.append(pred[0].numpy().squeeze())
                hierarchical_preds.append(pred[1])
            true = batch_y.detach().cpu().numpy().squeeze()
            
            true = true*std + mean
            trues.append(true)
            if 'UCR' not in self.args.data_path:
                times.append(batch_y_mark.detach().cpu().numpy().squeeze())
        
        preds = np.array(preds)
        trues = np.array(trues)

            
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape((-1, preds.shape[-2], preds.shape[-1]))
        trues = trues.reshape((-1, trues.shape[-2], trues.shape[-1]))
       
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        if 'UCR' not in self.args.data_path:
            times = np.array(times)
            times = times.reshape((-1, times.shape[-2], times.shape[-1]))
            np.save(folder_path + 'time.npy', times)
            
        if self.args.classification:
            preds = np.argmax(preds, axis=2)

        if self.args.classification:
            metrics = metric_cls(preds.squeeze(), trues.squeeze())
            print('prec:{}, rec:{}'.format(metrics[0], metrics[1]))
        else:
            metrics = metric_regr(preds.squeeze(), trues.squeeze())
            
            print('mae:{}, mse:{}'.format(metrics[0], metrics[1]))

        np.save(folder_path + 'metrics.npy', np.array(metrics))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        
        pickle.dump(hierarchical_preds, open(folder_path + 'hierarchical_pred.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        
        if self.args.plot_result:
            plt.figure(figsize=(20, 20))
            plt.plot(preds.reshape(-1, 8))
            plt.plot(trues.reshape(-1, 8), color='r')
