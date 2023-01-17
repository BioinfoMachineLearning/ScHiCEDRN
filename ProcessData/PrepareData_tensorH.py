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
from scipy.sparse import coo_matrix
import pandas as pd
from scipy.stats import pearsonr
import torch
import gc
import cooler



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

#  below is used for downsample the single cell hic data:
def dense2tag(matrix):
    """
    Converts a square matrix (dense) to coo-based tag matrix.
    """
    matrix = np.triu(matrix)
    tag_len = np.sum(matrix)
    tag_mat = np.zeros((tag_len, 2), dtype='int')
    coo_mat = coo_matrix(matrix)
    row, col, data = coo_mat.row, coo_mat.col, coo_mat.data
    start_idx = 0
    for i in range(len(row)):
        end_idx = start_idx + data[i]
        tag_mat[start_idx:end_idx, :] = (row[i], col[i])
        start_idx = end_idx
    return tag_mat, tag_len

def tag2dense(tag, nsize):
    """
    Coverts a coo-based tag matrix to densed square matrix.
    """
    coo_data, data = np.unique(tag, axis=0, return_counts=True)
    row, col = coo_data[:, 0], coo_data[:, 1]
    dense_mat = coo_matrix((data, (row, col)), shape=(nsize, nsize)).toarray()
    dense_mat = dense_mat + np.triu(dense_mat, k=1).T
    return dense_mat

def downsampling(matrix, down_ratio, verbose=False):
    """
    Downsampling method.
    """
    if verbose: print(f"[Downsampling] Matrix shape is {matrix.shape}")
    tag_mat, tag_len = dense2tag(matrix)
    sample_idx = np.random.choice(tag_len, int(tag_len * down_ratio))
    sample_tag = tag_mat[sample_idx]
    if verbose: print(f'[Downsampling] Sampling 1/{down_ratio} of {tag_len} reads')
    down_mat = tag2dense(sample_tag, matrix.shape[0])
    return down_mat

def splitPieces(fn, piece_size, step):
    data   = np.load(fn)
    pieces = []
    bound  = data.shape[0]
    bound1 = data.shape[1]
    assert bound == bound1
    for i in range(0, bound, step): # for the entire map
        for j in range(0, bound, step):
            if abs(i - j) <= 121 and i + step < bound and j + step < bound:
                pieces.append(data[i:i+piece_size, j:j+piece_size])
    pieces = np.asarray(pieces)
    pieces = np.expand_dims(pieces,1)
    return pieces

def loadBothConstraints(stria, strib, res, percent):
    contact_mapa  = np.loadtxt(stria)  # high resolution for our pre-train, the high and low are the same
    contact_mapb  = np.loadtxt(strib)  # low resolution

    print("============raw contact mapb shape: {}  and data length is {}".format(contact_mapb.shape, len(contact_mapb)))

    # method  has similar function

    rowsa         = (contact_mapa[:,0]/res).astype(int)
    colsa         = (contact_mapa[:,1]/res).astype(int)
    valsa         = contact_mapa[:,2]
    rowsb         = (contact_mapb[:,0]/res).astype(int)
    colsb         = (contact_mapb[:,1]/res).astype(int)
    valsb         = contact_mapb[:,2].astype(int)
    bigbin        = np.max((np.max((rowsa, colsa)), np.max((rowsb, colsb))))
    smallbin      = np.min((np.min((rowsa, colsa)), np.min((rowsb, colsb))))
    mata          = np.zeros((bigbin-smallbin+1, bigbin-smallbin+1), dtype='float32')
    matb          = np.zeros((bigbin-smallbin+1, bigbin-smallbin+1), dtype= 'int')
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

    removeidx = np.unique(np.concatenate((np.argwhere(diaga == 0)[:, 0], np.argwhere(np.isnan(diaga))[:, 0])))
    print("\n the new removeidx shape: {} and its length: {}".format(removeidx.shape, len(removeidx)))
    #  input("Press Enter to continue...")  # input used to check some thing

    mata = np.delete(mata, removeidx, axis=0)
    mata = np.delete(mata, removeidx, axis=1)
    gc.collect()


    per_a       = np.percentile(mata, 99.99)
    print(np.percentile(mata, 99.99), np.percentile(mata, 99.9), np.max(mata))
    mata        = np.clip(mata, 0, per_a)
    mata        = mata/per_a

    matb_down = downsampling(matb, percent)
    matb = np.delete(matb, removeidx, axis=0)
    matb = np.delete(matb, removeidx, axis=1)
    per_b       = np.percentile(matb, 99.99)
    print(np.percentile(matb, 99.99), np.percentile(matb, 99.9), np.max(matb))
    matb        = np.clip(matb, 0, per_b)
    matb        = matb/per_b

    matb_down = np.delete(matb_down, removeidx, axis = 0)
    matb_down = np.delete(matb_down, removeidx, axis = 1)
    per_d = np.percentile(matb_down, 99.99)
    print(per_d, np.percentile(matb_down, 99.9), np.max(matb_down))
    matb_down = np.clip(matb_down, 0, per_d)
    matb_down = matb_down / per_d

    return matb, matb_down


class GSE130711Module(pl.LightningDataModule):
    def __init__(self,
                 batch_size = 64,
                 res = 40000,
                 percent = 0.75,
                 piece_size = 40,
                 cell_line = 'Human',
                 cell_No = 1
                 ): #64 is used for unet_model
        super( ).__init__( )
        self.batch_size = batch_size
        self.res = res
        self.step = 40  # here the parameter should be modified
        self.piece_size = piece_size
        self.percent = percent
        self.cellLine = cell_line
        self.cellNo = cell_No
        self.dirname = "DataFull_"+self.cellLine+"_cell"+str(self.cellNo)+"_"+str(self.percent)+"_"+str(self.res)

    def extract_constraint_mats(self):
        if not os.path.exists(self.dirname+"/Constraints"):
            subprocess.run("mkdir -p "+self.dirname+"/Constraints", shell = True)

        outdir = self.dirname+"/Constraints"
        file_inter = glob.glob('../Datasets/Human/'+'cell'+str(self.cellNo)+'_'+r'*.mcool')
        filepath = file_inter[0]
        AllRes = cooler.fileops.list_coolers(filepath)
        print(AllRes)

        c = cooler.Cooler(filepath + '::resolutions/' + str(self.res))
        c1 = c.chroms()[:]  # c1 chromesize information in the list format
        print(c1.loc[0, 'name'], c1.index)
        # print('\n')

        for i in c1.index:
            print(i, c1.loc[i, 'name'])
            chro = c1.loc[i, 'name']  # chro is str
            # print(type(chro))
            c2 = c.matrix(balance = True, as_pixels = True, join = True).fetch(chro)
            c3 = c2[['start1', 'start2', 'count']]
            # print(c2)
            c2 = c2[['start1', 'start2', 'balanced']]
            c2.fillna(0, inplace = True)
            # print(c2)
            if i >= 22:
                pass
            else:
                c2.to_csv(outdir+'/chrom_' + str(i + 1) + '_' + str(self.res) + '.txt', sep = '\t', index = False, header = False)
                c3.to_csv(outdir+'/chrom_' + str(i + 1) + '_' + 'count' + '.txt', sep = '\t', index = False, header = False)

    def extract_create_numpy(self):
        if not os.path.exists(self.dirname+"/Full_Mats"):
            subprocess.run("mkdir -p "+self.dirname+"/Full_Mats", shell = True)

        globs = glob.glob(self.dirname+"/Constraints/chrom_1_" + str(self.res) + ".txt")
        if len(globs) == 0:
            print("wait.. first we need to extract mats and double check the mats")
            #  input("Press Enter to continue...")
            self.extract_constraint_mats()
        for i in range(1, 23):
            target, data = loadBothConstraints(
                self.dirname+"/Constraints/chrom_" + str(i) + "_" + str(self.res) + ".txt",
                self.dirname+"/Constraints/chrom_" + str(i) + "_" + "count" + ".txt",
                self.res, self.percent)

            target = np.float32(target)
            data = np.float32(data)

            print("the second time to convert float64 to float32")
            print(target.dtype)
            print(data.dtype)

            np.save(self.dirname+"/Full_Mats/GSE131811_mat_full_chr_" + str(i) + "_" + str(self.res), target)
            np.save(self.dirname+"/Full_Mats/GSE131811_mat_" + str(self.percent) + "_chr_" + str(i) + "_" + str(self.res), data)

    def split_numpy(self):
        if not os.path.exists(self.dirname+"/Splits"):
            subprocess.run("mkdir -p "+self.dirname+"/Splits", shell = True)

        globs = glob.glob(self.dirname+"/Full_Mats/GSE131811_mat_full_chr_1_" + str(self.res) + ".npy")
        if len(globs) == 0:
            self.extract_create_numpy( )

        for i in range(1, 23):
            target = splitPieces(self.dirname+"/Full_Mats/GSE131811_mat_full_chr_" + str(i) + "_" + str(self.res) + ".npy",
                                    self.piece_size, self.step)
            data = splitPieces(self.dirname+"/Full_Mats/GSE131811_mat_" + str(self.percent) + "_chr_" + str(i) + "_" + str(self.res) + ".npy",
                                  self.piece_size, self.step)

            np.save(
                self.dirname+"/Splits/GSE131811_full_chr_" + str(i) + "_" + str(self.res) + "_piece_" + str(self.piece_size),
                target)
            np.save(
                self.dirname+"/Splits/GSE131811_" + str(self.percent) + "_chr_" + str(i) + "_" + str(self.res) + "_piece_" + str(self.piece_size),
                data)

    def prepare_data(self):
        print("Preparing the Preparations ...")
        globs = glob.glob(
            self.dirname+"/Splits/GSE131811_full_chr_*_" + str(self.res) + "_piece_" + str(self.piece_size) + str(".npy"))
        if len(globs) > 20:
            print("Ready to go")
        else:
            print(".. wait, first we need to split the mats")
            self.split_numpy()

    class gse131811Dataset(Dataset):
        def __init__(self, full, tvt, res, piece_size, percent, dir):
            self.piece_size = piece_size
            self.tvt = tvt
            self.res = res
            self.full = full
            self.percent = percent
            self.dir = dir

            if full == True:
                if tvt in list(range(1, 23)):
                    self.chros = [tvt]
                if tvt == "train":
                    self.chros = [1, 3, 5, 7, 8, 9, 11, 13, 15, 16, 17, 19, 21, 22]
                elif tvt == "val":
                    self.chros = [4, 14, 18, 20] #
                elif tvt == "test":
                    self.chros = [2, 6, 10, 12] #

                self.target = np.load(
                    self.dir+"/Splits/GSE131811_full_chr_" + str(self.chros[0]) + "_" + str(self.res) + "_piece_" + str(
                        self.piece_size) + ".npy")
                self.data = np.load(
                    self.dir+"/Splits/GSE131811_" + str(self.percent) + "_chr_" + str(self.chros[0]) + "_" + str(self.res) + "_piece_" + str(
                        self.piece_size) + ".npy")
                self.info = np.repeat(self.chros[0], self.data.shape[0])
                for c, chro in enumerate(self.chros[1:]):
                    temp = np.load(
                        self.dir+"/Splits/GSE131811_full_chr_" + str(chro) + "_" + str(self.res) + "_piece_" + str(
                            self.piece_size) + ".npy")
                    print(self.target.shape, temp.shape, len(temp), chro)
                    # input("Press Enter to continue...")
                    if len(temp) == 0:
                        pass
                    else:
                        self.target = np.concatenate((self.target, temp))

                    temp = np.load(
                        self.dir+"/Splits/GSE131811_" + str(self.percent) + "_chr_" + str(chro) + "_" + str(self.res) + "_piece_" + str(
                            self.piece_size) + ".npy")
                    if len(temp) == 0:
                        pass
                    else:
                        self.data = np.concatenate((self.data, temp))  # temp.shape[0] means how many pieces every load
                        self.info = np.concatenate((self.info, np.repeat(chro, temp.shape[0])))

                self.data = torch.from_numpy(self.data)
                self.target = torch.from_numpy(self.target)
                self.info = torch.from_numpy(self.info)
                print("========================= the stage of training =====================\n", tvt)
                print(self.target.shape, self.data.shape)

            else:
                if tvt == "train":
                    self.chros = [15]
                elif tvt == "val":
                    self.chros = [16]
                elif tvt == "test":
                    self.chros = [17]
                self.target = np.load(
                    self.dir+"/Splits/GSE131811_full_chr_" + str(self.chros[0]) + "_" + str(self.res) + "_piece_" + str(
                        self.piece_size) + ".npy")
                self.data = np.load(
                    self.dir+"/Splits/GSE131811_" + str(self.percent) + "_chr_" + str(self.chros[0]) + "_" + str(self.res) + "_piece_" + str(
                        self.piece_size) + ".npy")
                self.info = np.repeat(self.chros[0], self.data.shape[0])

                self.data = torch.from_numpy(self.data)
                self.target = torch.from_numpy(self.target)
                self.info = torch.from_numpy(self.info)
                print("========================= the stage of training =====================\n", tvt)
                print(self.target.shape, self.data.shape)

        def __len__(self):
            return self.data.shape[0]

        def __getitem__(self, idx):
            return self.data[idx], self.target[idx], self.info[idx]

    def setup(self, stage = None):
        if stage in list(range(1, 23)):
            self.test_set = self.gse131811Dataset(full = True, tvt = stage, res = self.res, piece_size = self.piece_size, percent = self.percent, dir = self.dirname)
        if stage == 'fit':
            self.train_set = self.gse131811Dataset(full = True, tvt = 'train', res = self.res, piece_size = self.piece_size, percent = self.percent, dir = self.dirname)
            self.val_set = self.gse131811Dataset(full = True, tvt = 'val', res = self.res, piece_size = self.piece_size, percent = self.percent, dir = self.dirname)
        if stage == 'test':
            self.test_set = self.gse131811Dataset(full = True, tvt = 'test', res = self.res, piece_size = self.piece_size, percent = self.percent, dir = self.dirname)

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, num_workers = 12, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, num_workers = 12, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, num_workers = 12, drop_last=True)


if __name__ == '__main__':
    obj = GSE130711Module(percent = 0.02, cell_No = 1)
    obj.prepare_data()
    obj.setup(stage = 'fit')
    obj.setup(stage = 'test')
    print("all thing is done!!!")

    aa = obj.test_dataloader().dataset.data
    print("\nTo check wether the data is tensor")
    print(type(aa))
    print(aa.shape)
    test_loader = obj.test_dataloader()
    i = 1
    for data, target, ind in test_loader:
        print(f'\nthe data batch id: {i} in test_loader')
        print(data.shape)
        print(target.shape)
        print(f'the chrom of index shape is {ind.shape}')
        i = i+1

    ds_out = obj.test_dataloader().dataset.data[8:9]
    #ds_out = torch.from_numpy(obj.test_dataloader().dataset.data[8:9])  #10:11
    len1 = obj.test_dataloader().dataset.data.shape
    ds_out = ds_out[0][0][:, :]
    tar_out = obj.test_dataloader( ).dataset.target[8:9]
    #tar_out = torch.from_numpy(obj.test_dataloader().dataset.target[8:9])
    len2 = obj.test_dataloader().dataset.target.shape
    tar_out = tar_out[0][0][:, :]
    print("\nThe test data length is:{} and test target is: {}".format(len1, len2))
    print(ds_out)

    fig, ax = plt.subplots(1, 2)  # just one row/colum this will think as one-dimensional
    '''for j in range(0, 2): # in order to set the x_ticks and y_ticks without any labels/digits
        ax[j].set_xticks([])
        ax[j].set_yticks([])'''

    ax[0].imshow(ds_out, cmap = "Reds")
    ax[0].set_title("data_downsampled")

    ax[1].imshow(tar_out, cmap = "Reds")
    ax[1].set_title("Target")
    plt.show()
