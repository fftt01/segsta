import os
import sys
import traceback

import numpy as np
import psutil
import torch
from torch import optim
from torch.utils.data import DataLoader
import time
from torch.cuda.amp import autocast, GradScaler

from segsta import segsta
from tool import *
import json
from torch import optim
from einops import rearrange
from collections import OrderedDict
from dataloader_kth_seg import load_data
import torch.nn as nn


class params():
    def __init__(self, dict):
        self.img_size = dict['img_size']
        self.patch_size = dict['patch_size']
        self.N_in_channels = dict['in_chans']
        self.N_out_channels = dict['out_chans']
        self.num_blocks = 8


# models = [unetback]
class convlstm_with_nc_vhm0:
    def __init__(self, batch_size=48, en_channels=8, de_channels=1, image_size=256, workpath="./",
                 device=0, name="?", epochs=6, train=True, branch=1, var_num=7, var_loss_weight=None,
                 patiences=10, jsonpath=None, period=20, lr=0.001, selector=None, criterion=None,
                 dropout=0.1, model=None, itrlossprint=True, iterprintnum=10, embedtime=False,
                 memory=None, halftrain=True, percept_path="./percept.pth", accumulative_step=1, interval=60, train_cur=1, alpha = 1):
        self.alpha = alpha
        self.train_cur = train_cur
        self.select_range = de_channels//train_cur
        if device < 0:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device('cuda', device)
        if model is not None:
            self.model = model(params=params({
                "img_size": (image_size, image_size), "patch_size": 8, "in_chans": en_channels * var_num,
                "out_chans": de_channels
            }), img_size=(image_size, image_size))
            self.model.to(self.device)
        self.discriminator = None
        self.perceptual_loss = None
        self.accumulative_step = accumulative_step
        self.image_size = image_size
        self.interval = interval
        # self.discriminator = discri.to(self.device)
        self.en_channels = en_channels
        self.de_channels = de_channels
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.iterprintnum = iterprintnum
        self.itrlossprint = itrlossprint
        self.var_num = var_num
        self.selector = selector
        self.train_std = None
        self.train_mean = None
        self.period = period
        self.criterion = criterion
        self.patiences = patiences
        self.embedtime = embedtime
        self.halftrain = halftrain
        self.disc_start = None
        self.dataset = None
        self.data = None
        self.setting = name + "_in{}_out{}_b{}_ep{}_v{}_lr{}".format(en_channels, de_channels, batch_size,
                                                                     epochs, var_num, lr)
        self.name = name
        self.old_bs = batch_size
        self.checkpath = None
        self.init_loss = None
        self.jsonpath = None
        self.outputpath = None
        self.modelpath = None
        self.paraldata = None
        self.inparal = False
        self.buildPath(workpath)
        self.epoch_i = 0
        self.vise_model = None



    def modelgrouptrain(self, models, device_ids=None, main_device=0, inparal=True, itrlossprint=False, train=True):
        if device_ids is None:
            device_ids = [0]
        ls = len(models)
        ls_lr = len(lrs)
        ls_bss = len(bss)
        ls_eps = len(eps)
        i = 0
        self.device = torch.device('cuda', main_device)
        devices = []
        for device_id in device_ids:
            if device_id >= 0:
                devices.append(torch.device('cuda', device_id))
            else:
                devices.append(torch.device('cpu'))
        while i < ls:
            self.init_loss = torch.inf
            if i < ls_lr:
                self.lr = lrs[i]
            if i < ls_bss:
                self.batch_size = bss[i]
            if i < ls_eps:
                self.epochs = eps[i]
            self.setting = (f"{models[i].__name__}{i}_{self.name}_in{self.en_channels}_out{self.de_channels}_"
                            f"b{self.batch_size}_ep{self.epochs}_v{self.var_num}_lr{self.lr}_i{self.interval}")
            self.jsonpath = f"{self.checkpath}{self.setting}_record.json"
            self.modelpath = f"{self.checkpath}{self.setting}.pth"
            self.itrlossprint = itrlossprint
            print(f"transfer to new model setting: {self.setting}")
            self.model = models[i](in_channels=self.en_channels * self.var_num,
                                       out_channels=self.select_range, image_size=self.image_size,
                                       var_num=self.var_num, selfdevice=self.device, train_cur=self.train_cur)
            # self.perceptual_loss = self.get_percept()
            # self.discriminator = discriminator()
            if inparal:
                self.paralleltrain(device_ids=device_ids, main_device=main_device, train=train)
            elif train is True:
                self.model.to(self.device)
                self.train()
            else:
                return
            i = i + 1

    def lazyload(self):
        if self.data is None or self.old_bs != self.batch_size:
            dataloader_train, _, dataloader_test = \
                load_data(batch_size=self.batch_size,
                          val_batch_size=self.batch_size,
                          data_root='/export/home/jsj_2817/fangteng/dataset',
                          num_workers=4,
                          pre_seq_length=10, aft_seq_length=20, train_cur=self.train_cur)
            self.data = {
                'train': dataloader_train,
                'valid': dataloader_test,
                'test': dataloader_test
            }
            self.old_bs = self.batch_size

    def buildPath(self, workpath):
        if workpath[-1] != '/':
            workpath += '/'
        self.checkpath = mkdir(f"{workpath}checkpoints/")
        self.jsonpath = f"{self.checkpath}{self.setting}_record.json"
        self.outputpath = mkdir(f"{workpath}hiddenFeature/")
        self.modelpath = f"{self.checkpath}{self.setting}.pth"
        self.drawpath = f"{self.checkpath}{self.setting}_loss.png"
        self.memorypath = f"{self.checkpath}{self.setting}_memory.npy"

    def paralleltrain(self, device_ids=None, main_device=0, train=True):
        if device_ids is None:
            device_ids = [0, 1]
        device_ids.remove(main_device)
        device_ids.insert(0, main_device)
        print(f"main device: {self.device} | devices list: {device_ids}")
        self.model = torch.nn.DataParallel(self.model.to(self.device), device_ids=device_ids)
        # self.discriminator = torch.nn.DataParallel(self.discriminator.to(self.device), device_ids=device_ids)
        # self.perceptual_loss = torch.nn.DataParallel(self.perceptual_loss.to(self.device), device_ids=device_ids)
        self.inparal = True
        self.train()


    def train(self):
        self.lazyload()
        countParas(self.model, "generator")
        # countParas(self.discriminator, "discriminator")
        time_train_start = time.time()
        print(f"in parallel is {self.inparal}")
        print('>>>>>>>start training : {} | time : {}>>>>>>>>>>>'.format(self.setting,
                                                                         time.strftime('%Y-%m-%d %H:%M:%S',
                                                                                       time.localtime(
                                                                                           time_train_start))))
        train_steps = len(self.data['train'])
        criterion = self.criterion
        train_epochs = self.epochs
        scaler = GradScaler()
        best_dict = {}
        best_vali = torch.inf
        generate_optim = torch.optim.AdamW(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=0)
        dataloader = self.data['train']
        steps_per_epoch = len(dataloader)
        num_steps = steps_per_epoch * self.period
        gan_lr = 0.0001
        # discriminate_optim = torch.optim.AdamW(self.discriminator.parameters(), lr=gan_lr, betas=(0.9, 0.999),
        #                                        weight_decay=0)
        # self.disc_start = 1000
        # self.model.load_state_dict(torch.load("/root/autodl-tmp/code2/lowsky/checkpoints/muti0_selfvise_first_in5_out5_b16_ep80_v54_lr0.0005_i1.pth"))

        print(f"num_steps: {num_steps} | warmup iterations: {int(num_steps * 0.02)}")
        # test_loss = self.vali(self.data["test"], criterion, 0)
        # return
        early_stopping = EarlyStopping(verbose=True, patience=self.patiences, best_score=self.init_loss)
        for epochi in range(train_epochs):
            iter_count = 0
            mse_loss_bin = []
            final_loss_bin = []
            gan_loss_bin = []
            self.model.train()
            epoch_time = time.time()
            time_now = time.time()
            for i, (rand_i,x, y) in enumerate(dataloader):
                """
                batchsize, var, time, width, height
                """
                iter_count += 1
                adjust_lr_cos_warm(generate_optim, epochi * len(dataloader) + i + 1, num_steps, warmup=True,
                                   lr_max=self.lr)
                x = x.to(self.device).float()
                x = rearrange(x, "b t v w h -> b (t v) w h")
                rand_i = rand_i.to(self.device).long()
                pred = self.model(x, rand_i)
                y = y.to(self.device).float()
                y = rearrange(y, "b t v w h -> b (t v) w h")
                mse = (y - pred) ** 2
                mse = mse.mean(dim=(1, 2, 3))
                loss = (((self.train_cur - rand_i)**self.alpha) * mse / ((self.train_cur - rand_i)**self.alpha).sum()).sum()
                allloss = loss
                allloss = allloss / self.accumulative_step
                allloss.backward()
                if (i + 1) % self.accumulative_step == 0 or (i + 1) == len(dataloader):
                    generate_optim.step()
                    generate_optim.zero_grad()  # 清空梯度
                final_loss_bin.append(allloss.item() * self.accumulative_step)
                mse_loss_bin.append(loss.item())
                if i % (len(dataloader) // self.iterprintnum) == 0 and self.itrlossprint:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f} all_loss: {3:.7f}".format(i + 1, epochi + 1,
                                                                                              loss.item(),
                                                                                              allloss.item() * self.accumulative_step))
                    # if disc_factor != 0:
                    #     print("\t\tgan | real_loss: {0:.7f} fake_loss: {1:.7f}".format(d_loss_real.item(), d_loss_fake.item()))
                    # print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epochi + 1,
                    #                                                         loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((train_epochs - epochi) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    print('\tlearning rate: {:.3e}'.format(generate_optim.state_dict()['param_groups'][0]['lr']))
                    time_now = time.time()
                    iter_count = 0
            print("Epoch: {} cost time: {}".format(epochi + 1, time.time() - epoch_time))
            # adjust_lr_exp(optimizer, epochi, self.lr)
            mse_loss_bin = np.average(mse_loss_bin)
            final_loss_bin = np.average(final_loss_bin)
            # gan_loss_bin = np.average(gan_loss_bin, axis=0)
            test_loss = self.vali(self.data["test"], criterion, epochi)
            vali_loss = test_loss
            if vali_loss < best_vali:
                best_vali = vali_loss
                best_dict.update({
                    'train': "{:.5f}".format(mse_loss_bin),
                    'vali': "{:.5f}".format(vali_loss),
                    'test': "{:.5f}".format(test_loss),
                    # 'allloss': "{:.5f}".format(all_loss),
                    'epoch': epochi + 1
                })
                self.record_json({
                    'best': best_dict,
                })
            else:
                print(f"best loss: {best_dict}")
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f} all loss: {5:.7f}".format(
                    epochi + 1, train_steps, mse_loss_bin, vali_loss, test_loss, final_loss_bin))
            early_stopping(vali_loss, self.model, self.modelpath)
            self.record_json({
                'epoch' + str(epochi + 1) + "_mseloss": {
                    'train': "{:.5f}".format(mse_loss_bin),
                    'vali': "{:.5f}".format(vali_loss),
                    'test': "{:.5f}".format(test_loss),
                    # 'allloss': "{:.5f}".format(all_loss),
                }
            })
            if early_stopping.early_stop:
                print("Early stopping")
                break
        print('>>>>>>>end training : {} | time : {}>>>>>>>>>>>\n\n\n'.format(self.setting,
                                                                             time.strftime('%Y-%m-%d %H:%M:%S',
                                                                                           time.localtime(
                                                                                               time.time()))))


    def vali(self, vali_loader, criterion, epochi=-1):
        self.lazyload()
        self.model.eval()
        total_loss = []
        all_loss = []
        mse_loss = []
        loss_model = 0.0
        di_loss = []
        time_arr = []
        cur_num = self.de_channels // self.en_channels
        rest = self.de_channels - cur_num * self.en_channels
        train_steps = len(vali_loader)
        with torch.no_grad():
            for i, (_, x,y) in enumerate(vali_loader):
                if self.halftrain:
                    x = x.to(self.device)
                    with autocast():
                        pred = self.model(x)
                    loss = criterion(pred, y[:, 0].to(self.device))
                else:
                    x = x.to(self.device).float()
                    x = rearrange(x, "b t v w h -> b (t v) w h")
                    preds = []
                    for time_step in range(self.train_cur):
                        t = torch.full((x.shape[0],), time_step, dtype=torch.int).to(self.device)
                        pred = self.model(x,t)
                        preds.append(pred)
                    preds = torch.cat(preds, dim=1)
                    y = y.to(self.device).float()
                    y = rearrange(y, "b t v w h -> b (t v) w h")
                    mse = (y - preds) ** 2
                    loss = mse.mean()
                total_loss.append(loss.item())
                mse_loss.append(torch.mean(mse, dim=(0, 1)).sum().item())
        total_loss = np.average(total_loss)
        mse_loss = np.average(mse_loss)
        print(f"MSE: {total_loss:.5f}")
        self.model.train()
        return mse_loss

    def record_json(self, dict):
        writeJson(self.jsonpath, dict)

    def read_one_json(self, key: str):
        with open(self.jsonpath, "a+", encoding='utf-8') as f:
            f.seek(0, os.SEEK_SET)
            try:
                data = json.load(f)
                return data[key]
            except:
                return None


    def load_paral(self, path):
        state_dict = torch.load(path, "cpu")
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():  # k为module.xxx.weight, v为权重
            name = k[7:]  # 截取`module.`后面的xxx.weight
            new_state_dict[name] = v
        # load params
        self.model.load_state_dict(new_state_dict)


def torchMSE(pred, true):
    a = pred.float() - true.float()
    return torch.mean(a ** 2)

epoch = 100
selector = []
models = []
lrs = [0.00015]
bss = []
eps = []
selectedVar = ["VHM0", "VHM0_SW1", "VHM0_WW", "VTM01_WW", "VHM0_SW2", "VTM02", "VSDY"]
conv = convlstm_with_nc_vhm0(batch_size=16, epochs=epoch, name="kth", en_channels=10, de_channels=20,
                             device=0, period=epoch, lr=0.0008, criterion=torchMSE, patiences=epoch, var_num=1,
                             selector=None, embedtime=True, itrlossprint=True, model=None, alpha = 1,
                             halftrain=False, image_size=128, iterprintnum=2, accumulative_step=1, interval=1, train_cur=10)
devices = [1]
process = psutil.Process()
print(f"Started success, pid of process: {process.pid}, devices: {devices}")
conv.modelgrouptrain([segsta,segsta,segsta], inparal=False, device_ids=devices, itrlossprint=True, main_device=devices[0],
                     train=True)

