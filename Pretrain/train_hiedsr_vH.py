import os
import time
import numpy as np
from tqdm import tqdm
import subprocess

import torch
import torch.nn as nn
import torch.optim as optim
from ProcessData.PrepareData_tensorH import GSE130711Module
from torch.utils.tensorboard import SummaryWriter

from Models.Hiedsr_gan import Generator, Discriminator  # note the import should be carefull
from Utils.loss.Hiedsr_loss import GeneratorLoss as G1_loss
from Utils.loss.Hiedsrgan_loss import GeneratorLoss as G2_loss

from Utils.loss.SSIM import ssim
from math import log10


class hiedsr(object):
    def __init__(self, Gan = True, epoch = 250, batch_s = 1, cellN = 1, celline = 'Human', percentage = 0.75):

        self.epochs = epoch
        self.Gan = Gan

        # whether using GPU for training
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device

        # load the network for different models
        self.netG = Generator().to(device)
        if Gan:
            self.netD = Discriminator().to(device)

        root_dir = '../pretrained'
        if not os.path.exists(root_dir):
            subprocess.run("mkdir -p " + root_dir, shell = True)

        # out_dir: directory storing checkpoint files and parameters for saving to the our_dir

        self.cell_line = celline  #note: test here should pay attention
        self.cell_no = cellN
        ratio = "Downsample_"+str(percentage)
        ratiot = ratio + '_' + self.cell_line + str(self.cell_no)
        self.out_dir = os.path.join(root_dir, ratiot)  # for weights and bias
        self.out_dirM = os.path.join(root_dir, ratiot, "metrics")  # for metrics
        os.makedirs(self.out_dir, exist_ok = True)  # makedirs will make all the directories on the path if not exist.
        os.makedirs(self.out_dirM, exist_ok = True)

        self.datestr = time.strftime('%m_%d_%Y')
        self.ratio = ratio
        self.res = '40kb'
        self.chunk = 40
        self.stride = 40
        #self.cell_line = 'Dros_cell'
        #self.cell_no = 1
        self.runs = self.cell_line+str(self.cell_no)+'_'+str(percentage)

        # prepare training and valid dataset
        DataModule = GSE130711Module(batch_size = batch_s, cell_No = cellN, percent = percentage)
        DataModule.prepare_data()
        DataModule.setup(stage = 'fit')

        self.train_loader = DataModule.train_dataloader()
        self.valid_loader = DataModule.val_dataloader()

    def fit_model(self):
        if self.Gan:
            # initionlization
            self.netG.init_params()
            self.netD.init_params()

            # optimizer
            optimizerG = optim.Adam(self.netG.parameters(), lr = 0.0001)
            optimizerD = optim.Adam(self.netD.parameters(), lr = 0.0001)

            #loss function
            criterionG = G2_loss().to(self.device)
            criterionD = torch.nn.BCELoss().to(self.device)

            name = 'hiedsrgan_' +self.cell_line + str(self.cell_no) + "_" + self.ratio
            tb = SummaryWriter(self.runs+'/'+name)

            ssim_scores = []
            psnr_scores = []
            mse_scores = []
            mae_scores = []

            best_ssim = 0
            for epoch in range(1, self.epochs + 1):
                run_result = {'nsamples': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

                self.netG.train()
                self.netD.train()
                train_bar = tqdm(self.train_loader)
                for data, target, _ in train_bar:
                    batch_size = data.size(0)
                    run_result['nsamples'] += batch_size
                    ############################
                    # (1) Update D network: maximize D(x)-1-D(G(z))
                    ###########################
                    real_img = target.to(self.device)  # deal with target
                    z = data.to(self.device)
                    fake_img = self.netG(z)

                    ######### Train discriminator #########
                    self.netD.zero_grad()
                    real_out = self.netD(real_img)
                    labels_real = torch.ones_like(real_out).to(self.device)
                    fake_out = self.netD(fake_img)
                    labels_fake = torch.zeros_like(fake_out).to(self.device)

                    d_loss_real = criterionD(real_out, labels_real)
                    d_loss_fake = criterionD(fake_out, labels_fake)
                    d_loss = d_loss_real + d_loss_fake
                    d_loss.backward(retain_graph=True)
                    optimizerD.step()

                    ######### Train generator #########
                    self.netG.zero_grad()
                    fake_out = self.netD(fake_img)
                    _, _, _, g_loss = criterionG(fake_out.mean(),fake_img, real_img)
                    g_loss.backward()
                    optimizerG.step()

                    run_result['g_loss'] += g_loss.item() * batch_size
                    run_result['d_loss'] += d_loss.item() * batch_size
                    run_result['d_score'] += real_out.mean().item() * batch_size
                    run_result['g_score'] += fake_out.mean().item() * batch_size

                    train_bar.set_description(
                        desc = f"[{epoch}/{self.epochs}] Loss_D: {run_result['d_loss'] / run_result['nsamples']:.6f} Loss_G: {run_result['g_loss'] / run_result['nsamples']:.6f} D(x): {run_result['d_score'] / run_result['nsamples']:.6f} D(G(z)): {run_result['g_score'] / run_result['nsamples']:.6f}")

                train_gloss = run_result['g_loss'] / run_result['nsamples']
                train_dloss = run_result['d_loss'] / run_result['nsamples']
                train_dscore = run_result['d_score'] / run_result['nsamples']
                train_gscore = run_result['g_score'] / run_result['nsamples']
                tb.add_scalar("train_gloss", train_gloss, epoch)
                tb.add_scalar("train_dloss", train_dloss, epoch)

                valid_result = {'g_loss': 0, 'd_loss': 0, 'g_score': 0, 'd_score': 0,
                         'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'nsamples': 0}

                batch_ssims = []
                batch_mses = []
                batch_psnrs = []
                batch_maes = []

                self.netG.eval()
                self.netD.eval()
                valid_bar = tqdm(self.valid_loader)
                with torch.no_grad():
                    for val_lr, val_hr, inds in valid_bar:
                        batch_size = val_lr.size(0)
                        valid_result['nsamples'] += batch_size
                        lr = val_lr.to(self.device)
                        hr = val_hr.to(self.device)
                        sr = self.netG(lr)

                        sr_out = self.netD(sr)
                        hr_out = self.netD(hr)
                        _, _, _, g_loss = criterionG(sr_out.mean(), sr, hr)

                        d_loss_real = criterionD(hr_out, torch.ones_like(hr_out))
                        d_loss_fake = criterionD(sr_out, torch.zeros_like(sr_out))
                        d_loss = d_loss_real + d_loss_fake

                        valid_result['g_loss'] += g_loss.item() * batch_size
                        valid_result['d_loss'] += d_loss.item() * batch_size

                        batch_mse = ((sr - hr) ** 2).mean()
                        batch_mae = (abs(sr - hr)).mean()
                        valid_result['mse'] += batch_mse * batch_size
                        batch_ssim = ssim(sr, hr)
                        valid_result['ssims'] += batch_ssim * batch_size
                        valid_result['psnr'] = 10 * log10(1 / (valid_result['mse'] / valid_result['nsamples']))
                        valid_result['ssim'] = valid_result['ssims'] / valid_result['nsamples']
                        valid_bar.set_description(
                                desc = f"[Predicting in Test set] PSNR: {valid_result['psnr']:.6f} dB SSIM: {valid_result['ssim']:.6f}")

                        batch_ssims.append(valid_result['ssim'])
                        batch_psnrs.append(valid_result['psnr'])
                        batch_mses.append(batch_mse)
                        batch_maes.append(batch_mae)

                ssim_scores.append((sum(batch_ssims) / len(batch_ssims)))
                psnr_scores.append((sum(batch_psnrs) / len(batch_psnrs)))
                mse_scores.append((sum(batch_mses) / len(batch_mses)))
                mae_scores.append((sum(batch_maes) / len(batch_maes)))

                valid_gloss = valid_result['g_loss'] / valid_result['nsamples']
                valid_dloss = valid_result['d_loss'] / valid_result['nsamples']
                valid_gscore = valid_result['g_score'] / valid_result['nsamples']
                valid_dscore = valid_result['d_score'] / valid_result['nsamples']

                Nssim = sum(batch_ssims) / len(batch_ssims)
                Npsnr = sum(batch_psnrs) / len(batch_psnrs)
                Nmse  = sum(batch_mses) / len(batch_mses)
                Nmae  = sum(batch_maes) / len(batch_maes)

                tb.add_scalar("valid_gloss", valid_gloss, epoch)
                tb.add_scalar("valid_dloss", valid_dloss, epoch)
                tb.add_scalar("ssim", Nssim, epoch)
                tb.add_scalar("psnr", Npsnr, epoch)

                now_ssim = valid_result['ssim'].item()

                if now_ssim > best_ssim:
                    best_ssim = now_ssim
                    print(f'Now, Best ssim is {best_ssim:.6f}')
                    best_ckpt_file = f'bestg_{self.res}_c{self.chunk}_s{self.stride}_{self.cell_line}{self.cell_no}_hiedsrgan.pytorch'
                    torch.save(self.netG.state_dict(), os.path.join(self.out_dir, best_ckpt_file))
            final_ckpt_g = f'finalg_{self.res}_c{self.chunk}_s{self.stride}_{self.cell_line}{self.cell_no}_hiedsrgan.pytorch'
            torch.save(self.netG.state_dict(), os.path.join(self.out_dir, final_ckpt_g))

            ######### Uncomment to track scores across epochs #########
            ssim_scores = torch.tensor(ssim_scores)
            psnr_scores = torch.tensor(psnr_scores)
            mse_scores = torch.tensor(mse_scores)
            mae_scores = torch.tensor(mae_scores)

            ssim_scores = ssim_scores.cpu().detach().numpy()
            psnr_scores = psnr_scores.cpu().detach().numpy()
            mse_scores = mse_scores.cpu().detach().numpy()
            mae_scores = mae_scores.cpu().detach().numpy()

            # name is defined before summerwrite:
            np.savetxt(self.out_dirM+"/"+f'valid_ssim_scores_{name}'+'.txt', X=ssim_scores, delimiter=',')
            np.savetxt(self.out_dirM+"/"+f'valid_psnr_scores_{name}'+'.txt', X=psnr_scores, delimiter=',')
            np.savetxt(self.out_dirM+"/"+f'valid_mse_scores_{name}'+'.txt', X=mse_scores, delimiter=',')
            np.savetxt(self.out_dirM+"/"+f'valid_mae_scores_{name}'+'.txt', X=mae_scores, delimiter=',')

        else:
            self.netG.init_params()

            # optimizer
            optimizer = optim.Adam(self.netG.parameters(), lr = 0.0001)

            # loss function
            criterion = G1_loss().to(self.device)

            name = 'hiedsr_' + self.cell_line + str(self.cell_no) + "_" + self.ratio
            tb = SummaryWriter(self.runs+'/' + name)

            ssim_scores = []
            psnr_scores = []
            mse_scores = []
            mae_scores = []

            best_ssim = 0
            for epoch in range(1, self.epochs + 1):
                run_result = {'nsamples': 0, 'g_loss': 0, 'g_score': 0}

                for p in self.netG.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()

                self.netG.train()
                train_bar = tqdm(self.train_loader)
                for data, target, _ in train_bar:
                    batch_size = data.size(0)
                    run_result['nsamples'] += batch_size

                    real_img = target.to(self.device)
                    z = data.to(self.device)
                    fake_img = self.netG(z)

                    self.netG.zero_grad()
                    _, _, _, loss = criterion(fake_img, real_img)
                    loss.backward()
                    optimizer.step()
                    run_result['g_loss'] += loss.item() * batch_size
                    train_bar.set_description(
                        desc = f"[{epoch}/{self.epochs}]  Train Loss: {run_result['g_loss'] / run_result['nsamples']:.6f}")

                train_gloss = run_result['g_loss'] / run_result['nsamples']
                # train_gscore = run_result['g_score'] / run_result['nsamples']

                valid_result = {'g_loss': 0, 'g_score': 0, 'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'nsamples': 0}
                self.netG.eval()

                batch_ssims = []
                batch_mses = []
                batch_psnrs = []
                batch_maes = []

                valid_bar = tqdm(self.valid_loader)
                with torch.no_grad():
                    for val_lr, val_hr, _ in valid_bar:
                        batch_size = val_lr.size(0)
                        valid_result['nsamples'] += batch_size
                        lr = val_lr.to(self.device)
                        hr = val_hr.to(self.device)
                        sr = self.netG(lr)

                        _, _, _, g_loss = criterion(sr, hr)
                        valid_result['g_loss'] += g_loss.item() * batch_size

                        batch_mse = ((sr - hr) ** 2).mean()
                        batch_mae = (abs(sr - hr)).mean()
                        valid_result['mse'] += batch_mse * batch_size
                        batch_ssim = ssim(sr, hr)
                        valid_result['ssims'] += batch_ssim * batch_size
                        valid_result['psnr'] = 10 * log10(1 / (valid_result['mse'] / valid_result['nsamples']))
                        valid_result['ssim'] = valid_result['ssims'] / valid_result['nsamples']
                        valid_bar.set_description(
                            desc = f"[Predicting in Test set] PSNR: {valid_result['psnr']:.6f} dB SSIM: {valid_result['ssim']:.6f}")

                        batch_ssims.append(valid_result['ssim'])
                        batch_psnrs.append(valid_result['psnr'])
                        batch_mses.append(batch_mse)
                        batch_maes.append(batch_mae)

                ssim_scores.append((sum(batch_ssims) / len(batch_ssims)))
                psnr_scores.append((sum(batch_psnrs) / len(batch_psnrs)))
                mse_scores.append((sum(batch_mses) / len(batch_mses)))
                mae_scores.append((sum(batch_maes) / len(batch_maes)))

                valid_gloss = valid_result['g_loss'] / valid_result['nsamples']
                #valid_gscore = valid_result['g_score'] / valid_result['nsamples']

                Nssim = sum(batch_ssims) / len(batch_ssims)
                Npsnr = sum(batch_psnrs) / len(batch_psnrs)
                Nmse = sum(batch_mses) / len(batch_mses)
                Nmae = sum(batch_maes) / len(batch_maes)

                tb.add_scalar("train_gloss", train_gloss, epoch)
                tb.add_scalar("valid_gloss", valid_gloss, epoch)
                tb.add_scalar("ssim", Nssim, epoch)
                tb.add_scalar("psnr", Npsnr, epoch)

                now_ssim = valid_result['ssim'].item()

                if now_ssim > best_ssim:
                    best_ssim = now_ssim
                    print(f'Now, Best ssim is {best_ssim:.6f}')
                    best_ckpt_file = f'bestg_{self.res}_c{self.chunk}_s{self.stride}_{self.cell_line}{self.cell_no}_hiedsr.pytorch'
                    torch.save(self.netG.state_dict( ), os.path.join(self.out_dir, best_ckpt_file))

            final_ckpt_g = f'finalg_{self.res}_c{self.chunk}_s{self.stride}_{self.cell_line}{self.cell_no}_hiedsr.pytorch'
            torch.save(self.netG.state_dict(), os.path.join(self.out_dir, final_ckpt_g))

            ######### Uncomment to track scores across epochs #########
            ssim_scores = torch.tensor(ssim_scores)
            psnr_scores = torch.tensor(psnr_scores)
            mse_scores = torch.tensor(mse_scores)
            mae_scores = torch.tensor(mae_scores)

            ssim_scores = ssim_scores.cpu().detach().numpy()
            psnr_scores = psnr_scores.cpu().detach().numpy()
            mse_scores = mse_scores.cpu().detach().numpy()
            mae_scores = mae_scores.cpu().detach().numpy()

            # name is defined before summerwrite:
            np.savetxt(self.out_dirM + "/" + f'valid_ssim_scores_{name}' + '.txt', X = ssim_scores, delimiter = ',')
            np.savetxt(self.out_dirM + "/" + f'valid_psnr_scores_{name}' + '.txt', X = psnr_scores, delimiter = ',')
            np.savetxt(self.out_dirM + "/" + f'valid_mse_scores_{name}' + '.txt', X = mse_scores, delimiter = ',')
            np.savetxt(self.out_dirM + "/" + f'valid_mae_scores_{name}' + '.txt', X = mae_scores, delimiter = ',')


if __name__ == "__main__":
    train_model = hiedsr(Gan = False, epoch = 400, batch_s = 1, cellN = 1, percentage = 0.02)
    train_model.fit_model()
    print("\n\nTraining hiedsr is done!!!\n")

    train_model = hiedsr(Gan = True, epoch = 400, batch_s = 1, cellN = 1, percentage = 0.02)
    train_model.fit_model()
    print("\n\nTraining hiedsrgan is done!!!\n")
