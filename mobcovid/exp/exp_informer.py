from data.data_loader import Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack

from utils.tools import EarlyStopping, adjust_learning_rate, StandardScaler
from utils.metrics import metric

import numpy as np
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import pickle
import os
import time
import warnings
warnings.filterwarnings('ignore')



class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'informer':Informer,
            'informerstack':InformerStack,
        }
        if self.args.model=='informer' or self.args.model=='informerstack':
            e_layers = self.args.e_layers if self.args.model=='informer' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in,
                self.args.emergency_in,
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'LPD':Dataset_Custom,
            'ECL':Dataset_Custom,
            'Solar':Dataset_Custom,
            'custom':Dataset_Custom,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        size = [args.seq_len1, args.seq_len2, args.label_len, args.pred_len]
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=size,
            # features=args.features,
            scale=True,
            target=self.args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        # print(flag, self.args.target[0], data_set.len_1, self.args.target[1], data_set.len_2)
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, eps=1e-3, amsgrad=True)
        # model_optim = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=0.9)

        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x_1, batch_x_2, batch_y_1, batch_y_2, batch_x_mark_1, batch_x_mark_2, batch_y_mark, batch_x_emergency_mark_1, batch_x_emergency_mark_2, batch_y_emergency_mark) in enumerate(vali_loader):
            pred_1, pred_2, true_1, true_2 = self._process_one_batch(vali_data, batch_x_1, batch_x_2, batch_y_1, batch_y_2, batch_x_mark_1, batch_x_mark_2, batch_y_mark, batch_x_emergency_mark_1, batch_x_emergency_mark_2, batch_y_emergency_mark)
            pred = torch.cat((pred_1, pred_2), axis=2)
            true = torch.cat((true_1, true_2), axis=2)
            loss = criterion(pred, true)
            # loss_1 = criterion(pred_1.detach().cpu(), true_1.detach().cpu())
            # loss_2 = criterion(pred_2.detach().cpu(), true_2.detach().cpu())
            # loss = loss_1 + loss_2
            # loss = criterion(pred, true)
            total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x_1, batch_x_2, batch_y_1, batch_y_2, batch_x_mark_1, batch_x_mark_2, batch_y_mark, batch_x_emergency_mark_1, batch_x_emergency_mark_2, batch_y_emergency_mark) in enumerate(train_loader):
                iter_count += 1

                model_optim.zero_grad()
                pred_1, pred_2, true_1, true_2 = self._process_one_batch(train_data, batch_x_1, batch_x_2, batch_y_1, batch_y_2, batch_x_mark_1, batch_x_mark_2, batch_y_mark, batch_x_emergency_mark_1, batch_x_emergency_mark_2, batch_y_emergency_mark)
                # pred = torch.cat((pred_1, pred_2), axis=2)
                # true = torch.cat((true_1, true_2), axis=2)
                # loss = criterion(pred, true)
                loss_1 = criterion(pred_1, true_1)
                loss_2 = criterion(pred_2, true_2)

                loss = loss_1 * self.args.loss_target_1_weight + loss_2 * self.args.loss_target_2_weight
                # loss = torch.max(torch.abs(pred - true)) + criterion(pred, true)

                train_loss.append(loss.item())

                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Train Loss1: {3:.7f} Train Loss2: {4:.7f} Vali Loss: {5:.7f} Test Loss: {6:.7f}".format(
            #     epoch + 1, train_steps, train_loss, train_loss1, train_loss2, vali_loss, test_loss))


            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
        # train_loss_1 = np.array(train_loss_1)
        # train_loss_2 = np.array(train_loss_2)
        if not os.path.exists('./results/' + setting +'/'):
            os.makedirs('./results/' + setting +'/')
        # np.save('./results/' + setting +'/' + 'train_loss_1.npy', train_loss_1)
        # np.save('./results/' + setting + '/' + 'train_loss_2.npy', train_loss_2)
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def test(self, setting):
        self.scaler = StandardScaler()
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds_1 = []
        preds_2 = []
        trues_1 = []
        trues_2 = []

        for i, (batch_x_1, batch_x_2, batch_y_1, batch_y_2, batch_x_mark_1, batch_x_mark_2, batch_y_mark, batch_x_emergency_mark_1, batch_x_emergency_mark_2, batch_y_emergency_mark) in enumerate(test_loader):
            pred_1, pred_2, true_1, true_2 = self._process_one_batch(test_data, batch_x_1, batch_x_2, batch_y_1, batch_y_2, batch_x_mark_1, batch_x_mark_2, batch_y_mark, batch_x_emergency_mark_1, batch_x_emergency_mark_2, batch_y_emergency_mark)
            preds_1.append(pred_1.detach().cpu().numpy())
            preds_2.append(pred_2.detach().cpu().numpy())
            trues_1.append(true_1.detach().cpu().numpy())
            trues_2.append(true_2.detach().cpu().numpy())

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        preds_1 = np.array(preds_1)
        preds_2 = np.array(preds_2)
        trues_1 = np.array(trues_1)
        trues_2 = np.array(trues_2)

        print('test shape:', preds_1.shape, trues_1.shape)
        preds_1 = preds_1.reshape(-1, preds_1.shape[-2], preds_1.shape[-1]).reshape(-1, 1)
        trues_1 = trues_1.reshape(-1, trues_1.shape[-2], trues_1.shape[-1]).reshape(-1, 1)
        preds_2 = preds_2.reshape(-1, preds_2.shape[-2], preds_2.shape[-1]).reshape(-1, 1)
        trues_2 = trues_2.reshape(-1, trues_2.shape[-2], trues_2.shape[-1]).reshape(-1, 1)

        preds = np.concatenate((preds_1, preds_2), axis=1)
        trues = np.concatenate((trues_1, trues_2), axis=1)
        preds = test_data.inverse_transform1(preds)
        trues = test_data.inverse_transform1(trues)


        preds_1 = preds[:, 0]
        trues_1 = trues[:, 0]
        preds_2 = preds[:, 1]
        trues_2 = trues[:, 1]


        # preds_1 = preds_1.reshape(preds_1.shape[0], -1)
        # trues_1 = trues_1.reshape(trues_1.shape[0], -1)
        # preds_2 = preds_2.reshape(preds_2.shape[0], -1)
        # trues_2 = trues_2.reshape(trues_2.shape[0], -1)




        mae_1, mse_1, rmse_1, mape_1, mspe_1 = metric(preds_1, trues_1)
        mae_2, mse_2, rmse_2, mape_2, mspe_2 = metric(preds_2, trues_2)

        #print('rescaled mse:{}, mae:{}, mape:{}'.format(mse, mae, mape))

        print('rescaled latetime population mse1:{}, mae1:{}, mape1:{}'.format(mse_1, mae_1, mape_1))
        print('rescaled confirmed cases mse2:{}, mae2:{}, mape2:{}'.format(mse_2, mae_2, mape_2))
        # print("mse1:{}, mae1:{}".format(mean_squared_error(preds_1, trues_1), mean_absolute_error(preds_1, trues_1)))
        # print("mse2:{}, mae2:{}".format(mean_squared_error(preds_2, trues_2), mean_absolute_error(preds_2, trues_2)))
        np.save(folder_path+'pred_latetime population_rescale.npy', preds_1)
        np.save(folder_path+'true_latetime population_rescale.npy', trues_1)
        np.save(folder_path+'pred_confirmed_each_day_rescale.npy', preds_2)
        np.save(folder_path+'true_confirmed_each_day_rescale.npy', trues_2)

        np.save(folder_path + 'metrics1.npy', np.array([mae_1, mse_1, rmse_1, mape_1, mspe_1]))
        np.save(folder_path + 'metrics2.npy', np.array([mae_2, mse_2, rmse_2, mape_2, mspe_2]))
        # np.save(folder_path+'pred.npy', preds)
        # np.save(folder_path+'true.npy', trues)

        return mse_1, mae_1, mape_1, mse_2, mae_2, mape_2

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        
        preds = []
        time = []
        trues = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark, batch_x_emergency_mark, batch_y_emergency_mark, time_range) in enumerate(pred_loader):
            pred, true, time_stamp = self._process_one_batch_validate(pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_emergency_mark, batch_y_emergency_mark, time_range)
            time_stamp = int(time_stamp[0].detach().cpu().numpy()[0] / 1000000000)
            trues.append(true.detach().cpu().numpy())
            dt_object = datetime.fromtimestamp(time_stamp) - timedelta(hours=9)
            print(dt_object)
            preds.append(pred.detach().cpu().numpy())
            time.append(dt_object.date())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        preds = self.inverse_transform(preds)

        trues = np.array(trues)
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        trues = self.inverse_transform(trues)

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path+'real_prediction.npy', preds)
        np.save(folder_path+"real_number.npy", trues)

        # time save
        with open(folder_path + "time range.pkl", "wb") as file:
            pickle.dump(time, file)
        return

    def _process_one_batch(self, dataset_object, batch_x_1, batch_x_2, batch_y_1, batch_y_2, batch_x_mark_1, batch_x_mark_2, batch_y_mark, batch_x_emergency_mark_1, batch_x_emergency_mark_2, batch_y_emergency_mark):
        batch_x_1 = batch_x_1.float().to(self.device)
        batch_x_2 = batch_x_2.float().to(self.device)

        batch_y_1 = batch_y_1.float()
        batch_y_2 = batch_y_2.float()

        batch_x_mark_1 = batch_x_mark_1.float().to(self.device)
        batch_x_mark_2 = batch_x_mark_2.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        batch_x_emergency_mark_1 = batch_x_emergency_mark_1.float().to(self.device)
        batch_x_emergency_mark_2 = batch_x_emergency_mark_2.float().to(self.device)
        batch_y_emergency_mark = batch_y_emergency_mark.float().to(self.device)

        # decoder input
        if self.args.padding==0:
            dec_inp_1 = torch.zeros([batch_y_1.shape[0], self.args.pred_len, batch_y_1.shape[-1]]).float()
        elif self.args.padding==1:
            dec_inp_1 = torch.ones([batch_y_1.shape[0], self.args.pred_len, batch_y_1.shape[-1]]).float()
        dec_inp_1 = torch.cat([batch_y_1[:,:self.args.label_len,:], dec_inp_1], dim=1).float().to(self.device)

        if self.args.padding==0:
            dec_inp_2 = torch.zeros([batch_y_2.shape[0], self.args.pred_len, batch_y_2.shape[-1]]).float()
        elif self.args.padding==1:
            dec_inp_2 = torch.ones([batch_y_2.shape[0], self.args.pred_len, batch_y_2.shape[-1]]).float()
        dec_inp_2 = torch.cat([batch_y_2[:, :self.args.label_len, :], dec_inp_2], dim=1).float().to(self.device)

        # dec_inp = batch_x[:, -self.args.pred_len:, :]
        # dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)

        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs_1, outputs_2 = self.model(batch_x_1, batch_x_2, batch_x_mark_1, batch_x_mark_2, dec_inp_1, dec_inp_2, batch_y_mark, batch_x_emergency_mark_1, batch_x_emergency_mark_2, batch_y_emergency_mark)[0]
                else:
                    outputs_1, outputs_2 = self.model(batch_x_1, batch_x_2, batch_x_mark_1, batch_x_mark_2, dec_inp_1, dec_inp_2, batch_y_mark, batch_x_emergency_mark_1, batch_x_emergency_mark_2, batch_y_emergency_mark)
        else:
            if self.args.output_attention:
                outputs_1, outputs_2 = self.model(batch_x_1, batch_x_2, batch_x_mark_1, batch_x_mark_2, dec_inp_1, dec_inp_2, batch_y_mark, batch_x_emergency_mark_1, batch_x_emergency_mark_2, batch_y_emergency_mark)[0]
            else:
                outputs_1, outputs_2 = self.model(batch_x_1, batch_x_2, batch_x_mark_1, batch_x_mark_2, dec_inp_1, dec_inp_2, batch_y_mark, batch_x_emergency_mark_1, batch_x_emergency_mark_2, batch_y_emergency_mark)
        if self.args.inverse:
            outputs_1 = dataset_object.inverse_transform(outputs_1)
            outputs_2 = dataset_object.inverse_transform(outputs_2)
        f_dim = 0
        batch_y_1 = batch_y_1[:,-self.args.pred_len:,f_dim:].to(self.device)
        batch_y_2 = batch_y_2[:, -self.args.pred_len:, f_dim:].to(self.device)

        return outputs_1, outputs_2, batch_y_1, batch_y_2

    def _process_one_batch_validate(self, dataset_object, batch_x_1, batch_x_2, batch_y_1, batch_y_2, batch_x_mark_1, batch_x_mark_2, batch_y_mark_1, batch_y_mark_2, batch_x_emergency_mark_1, batch_x_emergency_mark_2, batch_y_emergency_mark_1, batch_y_emergency_mark_2, time_range):
        batch_x_1 = batch_x_1.float().to(self.device)
        batch_x_2 = batch_x_2.float().to(self.device)

        batch_y_1 = batch_y_1.float()
        batch_y_2 = batch_y_2.float()

        batch_x_mark_1 = batch_x_mark_1.float().to(self.device)
        batch_x_mark_2 = batch_x_mark_2.float().to(self.device)
        batch_y_mark_1 = batch_y_mark_1.float().to(self.device)
        batch_y_mark_2 = batch_y_mark_2.float().to(self.device)

        batch_x_emergency_mark_1 = batch_x_emergency_mark_1.float().to(self.device)
        batch_x_emergency_mark_2 = batch_x_emergency_mark_2.float().to(self.device)
        batch_y_emergency_mark_1 = batch_y_emergency_mark_1.float().to(self.device)
        batch_y_emergency_mark_2 = batch_y_emergency_mark_2.float().to(self.device)

        # decoder input
        if self.args.padding == 0:
            dec_inp_2 = torch.zeros([batch_y_2.shape[0], self.args.pred_len, batch_y_2.shape[-1]]).float()
        elif self.args.padding == 1:
            dec_inp_2 = torch.ones([batch_y_2.shape[0], self.args.pred_len, batch_y_2.shape[-1]]).float()
        dec_inp_2 = torch.cat([batch_y_2[:, :self.args.label_len_2, :], dec_inp_2], dim=1).float().to(self.device)


        # dec_inp = batch_x[:, -self.args.pred_len:, :]
        # dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)

        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs_1, outputs_2 = self.model(batch_x_1, batch_x_2, batch_x_mark_1, batch_x_mark_2, dec_inp_1, dec_inp_2, batch_y_mark_1, batch_y_mark_2, batch_x_emergency_mark_1, batch_x_emergency_mark_2, batch_y_emergency_mark_1, batch_y_emergency_mark_2)[0]
                else:
                    outputs_1, outputs_2 = self.model(batch_x_1, batch_x_2, batch_x_mark_1, batch_x_mark_2, dec_inp_1, dec_inp_2, batch_y_mark_1, batch_y_mark_2, batch_x_emergency_mark_1, batch_x_emergency_mark_2, batch_y_emergency_mark_1, batch_y_emergency_mark_2)
        else:
            if self.args.output_attention:
                outputs_1, outputs_2 = \
                self.model(batch_x_1, batch_x_2, batch_x_mark_1, batch_x_mark_2, dec_inp_1, dec_inp_2, batch_y_mark_1,
                           batch_y_mark_2, batch_x_emergency_mark_1, batch_x_emergency_mark_2, batch_y_emergency_mark_1,
                           batch_y_emergency_mark_2)[0]
            else:
                outputs_1, outputs_2 = self.model(batch_x_1, batch_x_2, batch_x_mark_1, batch_x_mark_2, dec_inp_1, dec_inp_2, batch_y_mark_1, batch_y_mark_2, batch_x_emergency_mark_1, batch_x_emergency_mark_2, batch_y_emergency_mark_1, batch_y_emergency_mark_2)
        if self.args.inverse:
            outputs_1 = dataset_object.inverse_transform(outputs_1)
            outputs_2 = dataset_object.inverse_transform(outputs_2)
        f_dim = 0
        batch_y_1 = batch_y_1[:, -self.args.pred_len:, f_dim:].to(self.device)
        batch_y_2 = batch_y_2[:, -self.args.pred_len:, f_dim:].to(self.device)
        time_range_1 = time_range[f_dim:]
        return outputs_1, outputs_2, batch_y_1, batch_y_2, time_range_1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


