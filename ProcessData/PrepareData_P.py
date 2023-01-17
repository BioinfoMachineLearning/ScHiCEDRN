import matplotlib.pyplot as plt
import os
import sys
sys.path.append(".")
sys.path.append("../")
from Utils import utils as ut
import pdb
import subprocess
import glob
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import random_split, DataLoader, Dataset
import pandas as pd
from scipy.stats import pearsonr
import torch
import gc
from scipy.stats import pearsonr

def noisy(noise_typ,image):
    if noise_typ == "gauss":
        row,col= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        # the second method to add poisson noise
        noisy = image + np.random.poisson(image)
        return noisy
    elif noise_typ =="speckle":
        row,col = image.shape
        gauss = np.random.randn(row,col)
        gauss = gauss.reshape(row,col)
        noisy = image + image * gauss
        return noisy

def splitPieces(fn, piece_size, step):
    data   = np.load(fn)
    pieces = []
    bound  = data.shape[0]
    for i in range(0, bound-piece_size+1, step):
        pieces.append(data[i:i+piece_size, i:i+piece_size])
    pieces = np.asarray(pieces)
    pieces = np.expand_dims(pieces,1)
    return pieces

def loadBothConstraints(stria, strib, res):
    contact_mapa  = np.loadtxt(stria)
    contact_mapb  = np.loadtxt(strib)
    rowsa         = (contact_mapa[:,0]/res).astype(int)
    colsa         = (contact_mapa[:,1]/res).astype(int)
    valsa         = contact_mapa[:,2]
    rowsb         = (contact_mapb[:,0]/res).astype(int)
    colsb         = (contact_mapb[:,1]/res).astype(int)
    valsb         = contact_mapb[:,2]
    bigbin        = np.max((np.max((rowsa, colsa)), np.max((rowsb, colsb))))
    smallbin      = np.min((np.min((rowsa, colsa)), np.min((rowsb, colsb))))
    mata          = np.zeros((bigbin-smallbin+1, bigbin-smallbin+1), dtype='float32')
    matb          = np.zeros((bigbin-smallbin+1, bigbin-smallbin+1), dtype='float32')
    coordinates   = list(range(smallbin, bigbin))
    i=0
    for ra,ca,ia in zip(rowsa, colsa, valsa):
        i = i+1
        #print(str(i)+"/"+str(len(valsa)+len(valsb)))
        mata[ra-smallbin, ca-smallbin] = ia
        mata[ca-smallbin, ra-smallbin] = ia
    for rb,cb,ib in zip(rowsb, colsb, valsb):
        i = i+1
        #print(str(i)+"/"+str(len(valsa)+len(valsb)))
        matb[rb-smallbin, cb-smallbin] = ib
        matb[cb-smallbin, rb-smallbin] = ib
    diaga         = np.diag(mata)
    diagb         = np.diag(matb)
    removeidx     = np.unique(np.concatenate((np.argwhere(diaga==0)[:,0], np.argwhere(diagb==0)[:,0], np.argwhere(np.isnan(diagb))[:,0])))

    # print(removeidx)
    print("Shape : ", mata.shape, removeidx.shape)
    mata = np.delete(mata, removeidx, axis=0)

    gc.collect()
    # input("Press Enter to continue...")
    mata = np.delete(mata, removeidx, axis=1)
    print("after deletion the mata shape: ", mata.shape)

    per_a       = np.percentile(mata, 99.9)
    mata        = np.clip(mata, 0, per_a)
    mata        = mata/per_a
    matb = np.delete(matb, removeidx, axis=0)
    matb = np.delete(matb, removeidx, axis=1)
    per_b       = np.percentile(matb, 99.9)
    matb        = np.clip(matb, 0, per_b)
    matb        = matb/per_b

    return mata, matb


class GSE100569Module(pl.LightningDataModule):
    def __init__(self,
                 batch_size = 64,
                 res = 40000,
                 juicer_tool = "other_tools/juicer_tools_1.22.01.jar",
                 piece_size = 65):
        super( ).__init__( )
        self.batch_size = batch_size
        self.res = res
        self.step = 12
        self.piece_size = piece_size

    def extract_create_numpy(self):
        if not os.path.exists("Data/Full_Mats"):
            subprocess.run("mkdir Data/Full_Mats", shell = True)

        globs = glob.glob("Data/Constraints/chrom_1_" + str(self.res) + ".txt")
        if len(globs) == 0:
            print("wait.. first we need to extract mats and double check the mats")
            input("Press Enter to continue...")
        for i in range(1, 21):
            target, data = loadBothConstraints(
                "Data/Constraints/chrom_" + str(i) + "_" + str(self.res) + ".txt",
                "Data/Constraints/chrom_" + str(i) + "_" + str(self.res) + ".txt",
                self.res)
            target = target.astype(float)
            data = noisy("gauss", data)
            # print(target.dtype)
            # print(data.dtype)
            target = np.float32(target)
            data = np.float32(data)
            '''print("the second time to convert float64 to float32")
            print(target.dtype)
            print(data.dtype)
            input("Press Enter to continue...")'''
            np.save("Data/Full_Mats/GSE100569_mat_clear_chr_" + str(i) + "_" + str(self.res), target)
            np.save("Data/Full_Mats/GSE100569_mat_noise_chr_" + str(i) + "_" + str(self.res), data)

    def split_numpy(self):
        if not os.path.exists("Data/Splits"):
            subprocess.run("mkdir Data/Splits", shell = True)

        globs = glob.glob("Data/Full_Mats/GSE100569_mat_clear_chr_1_" + str(self.res) + ".npy")
        if len(globs) == 0:
            self.extract_create_numpy( )

        for i in range(1, 21):
            target = splitPieces("Data/Full_Mats/GSE100569_mat_clear_chr_" + str(i) + "_" + str(self.res) + ".npy",
                                    self.piece_size, self.step)
            data = splitPieces("Data/Full_Mats/GSE100569_mat_noise_chr_" + str(i) + "_" + str(self.res) + ".npy",
                                  self.piece_size, self.step)
            '''
            print(target.dtype, target.shape)
            print(data.dtype, data.shape)
            input("Press Enter to continue...")
            '''
            np.save(
                "Data/Splits/GSE100569_clear_chr_" + str(i) + "_" + str(self.res) + "_piece_" + str(self.piece_size),
                target)
            np.save(
                "Data/Splits/GSE100569_noise_chr_" + str(i) + "_" + str(self.res) + "_piece_" + str(self.piece_size),
                data)

    def prepare_data(self):
        print("Preparing the Preparations ...")
        globs = glob.glob(
            "Data/Splits/GSE100569_clear_chr_*_" + str(self.res) + "_piece_" + str(self.piece_size) + str(".npy"))
        if len(globs) > 18:
            print("Ready to go")
        else:
            print(".. wait, first we need to split the mats")
            self.split_numpy( )

    class gse100569Dataset(Dataset):
        def __init__(self, full, tvt, res, piece_size):
            self.piece_size = piece_size
            self.tvt = tvt
            self.res = res
            self.full = full
            if full == True:
                if tvt in list(range(1, 21)):
                    self.chros = [tvt]
                if tvt == "train":
                    self.chros = [1, 3, 5, 6, 7, 8, 9, 10, 12, 13, 15, 17, 19, 20]
                elif tvt == "val":
                    self.chros = [2, 11, 18]
                elif tvt == "test":
                    self.chros = [4, 14, 16]

                self.target = np.load(
                    "Data/Splits/GSE100569_clear_chr_" + str(self.chros[0]) + "_" + str(self.res) + "_piece_" + str(
                        self.piece_size) + ".npy")
                self.data = np.load(
                    "Data/Splits/GSE100569_noise_chr_" + str(self.chros[0]) + "_" + str(self.res) + "_piece_" + str(
                        self.piece_size) + ".npy")
                self.info = np.repeat(self.chros[0], self.data.shape[0])
                for c, chro in enumerate(self.chros[1:]):
                    temp = np.load(
                        "Data/Splits/GSE100569_clear_chr_" + str(chro) + "_" + str(self.res) + "_piece_" + str(
                            self.piece_size) + ".npy")
                    self.target = np.concatenate((self.target, temp))
                    temp = np.load(
                        "Data/Splits/GSE100569_noise_chr_" + str(chro) + "_" + str(self.res) + "_piece_" + str(
                            self.piece_size) + ".npy")
                    self.data = np.concatenate((self.data, temp))
                    self.info = np.concatenate((self.info, np.repeat(chro, temp.shape[0])))
            else:
                if tvt == "train":
                    self.chros = [15]
                elif tvt == "val":
                    self.chros = [16]
                elif tvt == "test":
                    self.chros = [17]
                self.target = np.load(
                    "Data/Splits/GSE100569_clear_chr_" + str(self.chros[0]) + "_" + str(self.res) + "_piece_" + str(
                        self.piece_size) + ".npy")
                self.data = np.load(
                    "Data/Splits/GSE100569_noise_chr_" + str(self.chros[0]) + "_" + str(self.res) + "_piece_" + str(
                        self.piece_size) + ".npy")
                self.info = np.repeat(self.chros[0], self.data.shape[0])

        def __len__(self):
            return self.data.shape[0]

        def __getitem__(self, idx):
            return self.data[idx], self.target[idx], self.info[idx]

    def setup(self, stage = None):
        if stage in list(range(1, 21)):
            self.test_set = self.gse100569Dataset(full = True, tvt = stage, res = self.res, piece_size = self.piece_size)
        if stage == 'fit':
            self.train_set = self.gse100569Dataset(full = False, tvt = 'train', res = self.res, piece_size = self.piece_size)
            self.val_set = self.gse100569Dataset(full = False, tvt = 'val', res = self.res, piece_size = self.piece_size)
        if stage == 'test':
            self.test_set = self.gse100569Dataset(full = True, tvt = 'test', res = self.res, piece_size = self.piece_size)

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, num_workers = 12)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, num_workers = 12)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, num_workers = 12)

if __name__ == "__main__":

    obj = GSE100569Module()
    obj.prepare_data()
    obj.setup(stage = 'fit')
    obj.setup(stage = 'test')
    print("all thing is done!!!")
    # train dataset = (2605, 1, 65, 65); valid dataset = (590, 1, 65, 65); test dataset = (540, 1, 65, 65) 

