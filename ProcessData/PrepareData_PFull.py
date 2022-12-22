import matplotlib.pyplot as plt
import os
import sys
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
import cooler
from scipy.stats import pearsonr
from iced import normalization
from iced import filter


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
        ''' vals = len(np.unique(image))  # the first method to add poisson noise
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)'''
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

def loadBothConstraints(stria, strib, res, percent):
    contact_mapa  = np.loadtxt(stria)  # high resolution for our pre-train, the high and low are the same
    contact_mapb  = np.loadtxt(strib)  # low resolution

    print("============raw contact mapb shape: {}  and data length is {}".format(contact_mapb.shape, len(contact_mapb)))

    # method has similar
    slice_num = int(percent * len(contact_mapb))
    print('the downsample number is: {}'.format(slice_num))
    contact_mapb_dele = np.random.permutation(contact_mapb)[:slice_num]  # 9807 is 75% holds, 25% delete, this downsample will be used for GAN
    contact_mapb = contact_mapb_dele

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
    diaga         = np.diag(mata)  # np.diag() will give a 1-D array
    diagb         = np.diag(matb)

    removeidx = np.unique(np.concatenate((np.argwhere(diaga == 0)[:, 0], np.argwhere(np.isnan(diaga))[:,0])))
    print("\n the new removeidx shape: {} and its length: {}".format(removeidx.shape, len(removeidx)))
    #  input("Press Enter to continue...")  # input used to check some thing

    mata = np.delete(mata, removeidx, axis=0)
    # print("===========================================")
    # print("Shape : ", mata.shape)
    # input("Press Enter to continue...")   # input used to check some thing
    mata = np.delete(mata, removeidx, axis=1)
    gc.collect( )

    per_a       = np.percentile(mata, 99.9)
    mata        = np.clip(mata, 0, per_a)
    mata        = mata/per_a

    matb = np.delete(matb, removeidx, axis=0)
    matb = np.delete(matb, removeidx, axis=1)

    # below is about the ICE normalization for the mata
    # matb = normalization.ICE_normalization(matb) # add the normalization to matb

    per_b       = np.percentile(matb, 99.9)
    matb        = np.clip(matb, 0, per_b)
    matb        = matb/per_b

    return mata, matb


class GSE131811Module(pl.LightningDataModule):
    def __init__(self,
                 batch_size = 64,
                 res = 40000,
                 percent = 0.75,
                 piece_size = 77): # 65 + 12 == 77
        super( ).__init__( )
        self.batch_size = batch_size
        self.res = res
        self.step = 4  # here the parameter should be modified
        self.piece_size = piece_size
        self.percent = percent

    def extract_constraint_mats(self):
        if not os.path.exists("DataFull/Constraints"):
            subprocess.run("mkdir -p DataFull/Constraints", shell = True)

        filepath = '../../Datasets/Drosophila/GSM3820057_Cell1.10000.mcool'
        AllRes = cooler.fileops.list_coolers(filepath)
        # print(AllRes)

        c = cooler.Cooler(filepath + '::resolutions/' + str(self.res))
        c1 = c.chroms( )[:]  # c1 chromesize information in the list format
        print(c1.loc[0, 'name'], c1.index)
        # print('\n')

        for i in c1.index:
            print(i, c1.loc[i, 'name'])
            chro = c1.loc[i, 'name']  # chro is str
            # print(type(chro))
            c2 = c.matrix(balance = True, as_pixels = True, join = True).fetch(chro)
            # print(c2)
            c2 = c2[['start1', 'start2', 'balanced']]
            c2.fillna(0, inplace = True)
            # print(c2)
            if i == 6:
                pass
            else:
                c2.to_csv('DataFull/Constraints/chrom_' + str(i + 1) + '_' + str(self.res) + '.txt', sep = '\t', index = False, header = False)

    def extract_create_numpy(self):
        if not os.path.exists("DataFull/Full_Mats"):
            subprocess.run("mkdir -p DataFull/Full_Mats", shell = True)

        globs = glob.glob("DataFull/Constraints/chrom_1_" + str(self.res) + ".txt")
        if len(globs) == 0:
            print("wait.. first we need to extract mats and double check the mats")
            #  input("Press Enter to continue...")
            self.extract_constraint_mats()
        for i in range(1, 7):
            target, data = loadBothConstraints(
                "DataFull/Constraints/chrom_" + str(i) + "_" + str(self.res) + ".txt",
                "DataFull/Constraints/chrom_" + str(i) + "_" + str(self.res) + ".txt",
                self.res, self.percent)

            target = np.float32(target)
            data = np.float32(data)

            print("the second time to convert float64 to float32")
            print(target.dtype)
            #  print(target)
            print(data.dtype)
            if i == 5:
                print(target.shape)  # the chrom_5, its' mat is (26, 26), so the splits' number = 0;
                print(target)
            # input("Press Enter to continue...")
            np.save("DataFull/Full_Mats/GSE131811_mat_full_chr_" + str(i) + "_" + str(self.res), target)
            np.save("DataFull/Full_Mats/GSE131811_mat_" + str(self.percent) + "_chr_" + str(i) + "_" + str(self.res), data)

    def split_numpy(self):
        if not os.path.exists("DataFull/Splits"):
            subprocess.run("mkdir -p DataFull/Splits", shell = True)

        globs = glob.glob("DataFull/Full_Mats/GSE131811_mat_full_chr_1_" + str(self.res) + ".npy")
        if len(globs) == 0:
            self.extract_create_numpy( )

        for i in range(1, 7):
            target = splitPieces("DataFull/Full_Mats/GSE131811_mat_full_chr_" + str(i) + "_" + str(self.res) + ".npy",
                                    self.piece_size, self.step)
            data = splitPieces("DataFull/Full_Mats/GSE131811_mat_" + str(self.percent) + "_chr_" + str(i) + "_" + str(self.res) + ".npy",
                                  self.piece_size, self.step)

            if (i == 5):
                print("====chrom: {}, splits shape: {} and splits number: ".format(i, target.shape, len(target)))
                print(target)
                # input("Press Enter to continue...")
            np.save(
                "DataFull/Splits/GSE131811_full_chr_" + str(i) + "_" + str(self.res) + "_piece_" + str(self.piece_size),
                target)
            np.save(
                "DataFull/Splits/GSE131811_" + str(self.percent) + "_chr_" + str(i) + "_" + str(self.res) + "_piece_" + str(self.piece_size),
                data)

    def prepare_data(self):
        print("Preparing the Preparations ...")
        globs = glob.glob(
            "DataFull/Splits/GSE131811_full_chr_*_" + str(self.res) + "_piece_" + str(self.piece_size) + str(".npy"))
        if len(globs) > 5:
            print("Ready to go")
        else:
            print(".. wait, first we need to split the mats")
            self.split_numpy( )

    class gse131811Dataset(Dataset):
        def __init__(self, full, tvt, res, piece_size, percent):
            self.piece_size = piece_size
            self.tvt = tvt
            self.res = res
            self.full = full
            self.percent = percent
            if full == True:
                if tvt in list(range(1, 7)):
                    self.chros = [tvt]
                if tvt == "train":
                    self.chros = [1, 2, 3, 5]
                elif tvt == "val":
                    self.chros = [4]
                elif tvt == "test":
                    self.chros = [6]

                self.target = np.load(
                    "DataFull/Splits/GSE131811_full_chr_" + str(self.chros[0]) + "_" + str(self.res) + "_piece_" + str(
                        self.piece_size) + ".npy")
                self.data = np.load(
                    "DataFull/Splits/GSE131811_" + str(self.percent) + "_chr_" + str(self.chros[0]) + "_" + str(self.res) + "_piece_" + str(
                        self.piece_size) + ".npy")
                self.info = np.repeat(self.chros[0], self.data.shape[0])
                for c, chro in enumerate(self.chros[1:]):
                    temp = np.load(
                        "DataFull/Splits/GSE131811_full_chr_" + str(chro) + "_" + str(self.res) + "_piece_" + str(
                            self.piece_size) + ".npy")
                    print(self.target.shape, temp.shape, len(temp), chro)
                    # input("Press Enter to continue...")
                    if len(temp) == 0:
                        pass
                    else:
                        self.target = np.concatenate((self.target, temp))

                    temp = np.load(
                        "DataFull/Splits/GSE131811_" + str(self.percent) + "_chr_" + str(chro) + "_" + str(self.res) + "_piece_" + str(
                            self.piece_size) + ".npy")
                    if len(temp) == 0:
                        pass
                    else:
                        self.data = np.concatenate((self.data, temp))  # temp.shape[0] means how many pieces every load
                        self.info = np.concatenate((self.info, np.repeat(chro, temp.shape[0])))


                print("========================= the stage of training =====================\n", tvt)
                print(self.target.shape, self.data.shape)
                # print(self.target.shape[0], self.target.shape[1], self.target.shape[2], self.target.shape[3])
                # input("Press Enter to continue...")

            else:
                if tvt == "train":
                    self.chros = [3]
                elif tvt == "val":
                    self.chros = [4]  #
                elif tvt == "test":
                    self.chros = [6]
                self.target = np.load(
                    "DataFull/Splits/GSE131811_full_chr_" + str(self.chros[0]) + "_" + str(self.res) + "_piece_" + str(
                        self.piece_size) + ".npy")
                self.data = np.load(
                    "DataFull/Splits/GSE131811_" + str(self.percent) + "_chr_" + str(self.chros[0]) + "_" + str(self.res) + "_piece_" + str(
                        self.piece_size) + ".npy")
                self.info = np.repeat(self.chros[0], self.data.shape[0])

                print("========================= the stage of training =====================\n", tvt)
                print(self.target.shape, self.data.shape)
                # input("Press Enter to continue...")

        def __len__(self):
            return self.data.shape[0]

        def __getitem__(self, idx):
            return self.data[idx], self.target[idx], self.info[idx]

    def setup(self, stage = None):
        if stage in list(range(1, 7)):
            self.test_set = self.gse131811Dataset(full = True, tvt = stage, res = self.res, piece_size = self.piece_size, percent = self.percent)
        if stage == 'fit':
            self.train_set = self.gse131811Dataset(full = True, tvt = 'train', res = self.res, piece_size = self.piece_size, percent = self.percent)
            self.val_set = self.gse131811Dataset(full = True, tvt = 'val', res = self.res, piece_size = self.piece_size, percent = self.percent)
        if stage == 'test':
            self.test_set = self.gse131811Dataset(full = True, tvt = 'test', res = self.res, piece_size = self.piece_size, percent = self.percent)

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, num_workers = 12)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, num_workers = 12)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, num_workers = 12)

if __name__ == "__main__":

    obj = GSE131811Module()
    obj.prepare_data()
    obj.setup(stage = 'fit')
    obj.setup(stage = 'test')
    print("all thing is done!!!")

