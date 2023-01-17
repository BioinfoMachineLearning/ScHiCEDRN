#the implementation of SSIM in this file is pulled from DeepHiC https://github.com/omegahh/DeepHiC
import torch.nn as nn
import torch.nn.functional as F
from math import log10
from Models.schicedrn_gan import Generator  # here should be modified
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import sys
sys.path.append(".")
sys.path.append("../")
import numpy as np
from tqdm import tqdm
import torch
from ProcessData.PrepareData_tensor import GSE131811Module
from ProcessData.PrepareData_tensorH import GSE130711Module
from Utils.loss.SSIM import ssim
from Utils.GenomeDISCO import compute_reproducibility

class VisionMetrics:
    def __init__(self):
        self.ssim         = ssim
        self.metric_logs = {
            #"pre_pcc":[],
            "pas_pcc":[],
            #"pre_spc":[],
            "pas_spc":[],
            #"pre_psnr":[],
            "pas_psnr":[],
            #"pre_ssim":[],
            "pas_ssim":[],
            #"pre_mse":[],
            "pas_mse":[],
            #"pre_snr":[],
            "pas_snr":[],
            "pas_gds":[]
            }

    def log_means(self, name):
        return (name, np.mean(self.metric_logs[name]))


    def setDataset(self, chro = "test", percent = 0.75, cellN = 21,  cell_line="Dros_cell"):
        if cell_line == "Dros_cell":
            dm_test      = GSE131811Module(batch_size=1,  percent = percent, cell_No = cellN)
        if cell_line == "Human":
            dm_test = GSE130711Module(batch_size=1, percent=percent, cell_No=cellN)
            '''if cell_line == "K562":
            self.dm_test      = K562Module(batch_size=1, res=res, piece_size=piece_size)'''
        dm_test.prepare_data()
        dm_test.setup(stage=chro)
        self.test_loader = dm_test.test_dataloader()

    def getMetrics(self, model, spliter):
        self.metric_logs = {
            #"pre_pcc":[],
            "pas_pcc":[],
            #"pre_spc":[],
            "pas_spc":[],
            #"pre_psnr":[],
            "pas_psnr":[],
            #"pre_ssim":[],
            "pas_ssim":[],
            #"pre_mse":[],
            "pas_mse":[],
            #"pre_snr":[],
            "pas_snr":[],
            "pas_gds":[]
            }

        #for e, epoch in enumerate(self.test_loader):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        result_data = []
        result_inds = []

        batch_ssims = []
        batch_mses = []
        batch_psnrs = []
        batch_snrs = []
        batch_spcs = []
        batch_pccs = []
        
        batch_gds = []


        test_result = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'nsamples': 0, 'pccs':0, 'pcc':0, 'spcs':0, 'spc':0, 'snrs':0, 'snr':0}

        test_bar = tqdm(self.test_loader)
        with torch.no_grad():
            for lr, hr, inds in test_bar:
                batch_size = lr.size(0)
                test_result['nsamples'] += batch_size

                lr = lr.to(device)
                hr = hr.to(device)
                # data, full_target, info = epoch

                if spliter == "hiedsr" or spliter == "hiedsrgan":  #no need padding the input data
                    out = model(lr)

                if spliter == "hicplus" or spliter == "hicsr":  #need padding the input data
                    temp = F.pad(lr, (6, 6, 6, 6), mode='constant')
                    out = model(temp)

                if spliter == "deephic" or spliter=='unet' or spliter=='hicarn': # no need padding the data
                    out = model(lr)
                #print(out)
                #input("press enter to contnue....")
                batch_mse = ((out - hr) ** 2).mean()
                test_result['mse'] += batch_mse * batch_size
                batch_ssim = self.ssim(out, hr)
                test_result['ssims'] += batch_ssim * batch_size
                test_result['psnr'] = 10 * log10(1 / (test_result['mse'] / test_result['nsamples']))
                test_result['ssim'] = test_result['ssims'] / test_result['nsamples']

                batch_snr = (hr.sum() / ((hr - out) ** 2).sum().sqrt())
                if ((hr - out) ** 2).sum().sqrt() == 0 and hr.sum() == 0:
                    batch_snr = torch.tensor(0.0)
                test_result['snrs'] += batch_snr * batch_size
                test_result['snr'] = test_result['snrs']
                batch_pcc = pearsonr(out.cpu().flatten(), hr.cpu().flatten())[0]
                batch_spc = spearmanr(out.cpu().flatten(), hr.cpu().flatten())[0]
                test_result['pccs'] += batch_pcc * batch_size
                test_result['spcs'] += batch_spc * batch_size
                test_result['pcc'] = test_result['pccs']/test_result['nsamples']
                test_result['spc'] = test_result['spcs']/test_result['nsamples']

                batch_ssims.append(test_result['ssim'])
                batch_psnrs.append(test_result['psnr'])
                batch_mses.append(batch_mse)
                batch_snrs.append(test_result['snr'])
                batch_pccs.append(test_result['pcc'])
                batch_spcs.append(test_result['spc'])
                
                for i, j in zip(hr, out):
                    if hr.sum() == 0:
                        continue
                    out1 = torch.squeeze(j, dim = 0)
                    hr1 = torch.squeeze(i, dim = 0)
                    out2 = out1.cpu().detach().numpy()
                    hr2 = hr1.cpu().detach().numpy()
                    genomeDISCO = compute_reproducibility(out2, hr2, transition = True)
                    batch_gds.append(genomeDISCO)
                

        Nssim = sum(batch_ssims) / len(batch_ssims)
        Npsnr = sum(batch_psnrs) / len(batch_psnrs)
        Nmse = sum(batch_mses) / len(batch_mses)
        Nsnr = sum(batch_snrs) / len(batch_snrs)
        Npcc = sum(batch_pccs) / len(batch_pccs)
        Nspc = sum(batch_spcs) / len(batch_spcs)
        Ngds = sum(batch_gds) / len(batch_gds)

        self.metric_logs['pas_ssim'].append(Nssim.cpu())
        self.metric_logs['pas_psnr'].append(Npsnr)
        self.metric_logs['pas_mse'].append(Nmse.cpu())
        self.metric_logs['pas_snr'].append(Nsnr.cpu())
        self.metric_logs['pas_pcc'].append(Npcc)
        self.metric_logs['pas_spc'].append(Nspc)
        self.metric_logs['pas_gds'].append(Ngds)

            # self._logPCC(data=data, target=full_target, output=output)
            # self._logSPC(data=data, target=full_target, output=output)
            # self._logMSE(data=data, target=full_target, output=output)
            # self._logPSNR(data=data, target=full_target, output=output)
            # self._logSNR(data=data, target=full_target, output=output)
            # self._logSSIM(data=data, target=full_target, output=output)
        print(list(map(self.log_means, self.metric_logs.keys())))
        return self.metric_logs

if __name__=='__main__':
    print("\nTest for the stardrd metrics\n")
    '''device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    visionMetrics = VisionMetrics()
    visionMetrics.setDataset(20, cell_line="GSE131811")
    WEIGHT_PATH   = "deepchromap_weights.ckpt"
    model         =     Generator().to(device)
    pretrained_model = model.load_from_checkpoint(WEIGHT_PATH)
    pretrained_model.freeze()
    visionMetrics.getMetrics(model=pretrained_model, spliter=False)'''
