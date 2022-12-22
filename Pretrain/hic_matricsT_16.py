import os
import sys
sys.path.append(".")
import subprocess
import glob
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import shutil

#load data module
from ProcessData.PrepareData_tensor import GSE131811Module
from ProcessData.PrepareData_tensorH import GSE130711Module

#Our models
import Models.Hiedsr_ganT48 as hiedsr
#other models
import Models.hicsr   as hicsr
import Models.deephic as deephic
import Models.hicplus as hicplus
import Models.Unet_parts1 as unet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

'''
for Human, the test chros list = [2, 6, 10, 12]
for Dros_cell, the test chros list = [2, 6], besides that original only 6 as the test chromosome
'''

#single chromsome information for test
RES        = 40000
PIECE_SIZE = 40

#below is the cell information for test
cell_lin = "Human"
cell_no = 1
percentage = 0.02

#below two is used for pretrained models' paths
cell_not = 1
cell_lint = "HumanT48_"

file_inter = "Downsample_"+str(percentage)+"_"+cell_lint+str(cell_not)+"/"
#file_intert = "Downsample_0.75_Dros_cell21/"


#Our models on multicom
hiedsrMod  = hiedsr.Generator().to(device)
#file_temp = "../pretrained/"+"Downsample_"+str(percentage)+"_"+cell_lin+"T0.75_16"+str(cell_no)+"/"+"bestg_40kb_c40_s40_"+cell_lin+"T0.75_16"+str(cell_no)+"_hiedsr.pytorch"
#file_path1 = "../pretrained/Downsample_0.75/bestg_40kb_c40_s40_Dros_cell1_hiedsr.pytorch"
file_path1 = "../pretrained/"+file_inter+"bestg_40kb_c40_s40_"+cell_lint+str(cell_not)+"_hiedsr.pytorch"
hiedsrMod.load_state_dict(torch.load(file_path1))
hiedsrMod.eval()



'''
# other models on daisy
model_hicplus = hicplus.Net(40,28).to(device)
file_path1 = "../pretrained/"+file_inter+"finalg_40kb_c40_s40_"+cell_lint+str(cell_not)+"_hicplus.pytorch"
model_hicplus.load_state_dict(torch.load(file_path1))
model_hicplus.eval()

model_unet = unet.unet_2D().to(device)
file_path1 = "../pretrained/"+file_inter+"finalg_40kb_c40_s40_"+cell_lint+str(cell_not)+"_unet.pytorch"
model_unet.load_state_dict(torch.load(file_path1))
model_unet.eval()
'''

#prepare the input files for hicRep
def HicRepInput(file1, res):
    contact_mapa = np.loadtxt(file1)
    rowsa = (contact_mapa[:, 0] / res).astype(int)
    colsa = (contact_mapa[:, 1] / res).astype(int)
    print(type(rowsa))
    print(rowsa)

    print(colsa)
    valsa = contact_mapa[:, 2]
    bigbin = np.max((rowsa, colsa))
    smallbin = np.min((rowsa, colsa))
    print("\n============bigbin: {} and rowsa lenth: {}=============\n".format(bigbin, len(rowsa)))
    print("============smallbin: {} and closa lenth: {}=============\n".format(smallbin, len(colsa)))

    mata = np.zeros((bigbin - smallbin + 1, bigbin - smallbin + 1), dtype = 'int')
    for ra, ca, ia in zip(rowsa, colsa, valsa):
        mata[ra - smallbin, ca - smallbin] = abs(ia)
        mata[ca - smallbin, ra - smallbin] = abs(ia)

    #np.savetxt(file2, mata)
    return mata

root_dir = "../hicqc_inputs/"+cell_lint+str(cell_no)+"_"+str(percentage)+"_part100"
root_dirR = "../hicRep_inputs/"+cell_lint+str(cell_no)+"_"+str(percentage)+"_part100"


if cell_lin == "Human":
    chros_all = [2, 6, 10, 12]
else:
    chros_all = [2, 6]


for chro in chros_all:
    CHRO       = chro #single chromsome information for test

    #The below two step to make sure the two out directories are new without any additional old information
    if os.path.exists(root_dir):
        globs = glob.glob(root_dirR+"/*_"+str(CHRO)+".txt")
        if root_dirR+"/original_"+str(CHRO)+'.txt' in globs:
            shutil.rmtree(root_dir)
            shutil.rmtree(root_dirR)
            print("\n================Trim the old information in output directories=============\n")

    # if not exist, we should setup the new director to store the data
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir, exist_ok = True)
        os.makedirs(root_dirR, exist_ok = True)

    #Prepare files for 3DChromatin_ReplicateQC

    hiedsr_hic       = open(root_dir+"/hiedsr_"+str(CHRO), 'w')
    original_hic     = open(root_dir+"/original_"+str(CHRO), 'w')
    down_hic         = open(root_dir+"/down_"+str(CHRO), 'w')
    bins_file        = open(root_dir+"/bins_"+str(CHRO)+".bed",'w')

    '''
    unet_hic         = open(root_dir+"/unet_"+str(CHRO), 'w')
    hicplus_hic      = open(root_dir+"/hicplus_"+str(CHRO), 'w')
    '''

    #Prepare the .txt files for hicRep

    hiedsr_hicRep       = open(root_dirR+"/hiedsr_"+str(CHRO)+'.txt', 'w')
    original_hicRep     = open(root_dirR+"/original_"+str(CHRO)+'.txt', 'w')
    down_hicRep         = open(root_dirR+"/down_"+str(CHRO)+'.txt', 'w')

    '''
    unet_hicRep         = open(root_dirR+"/unet_"+str(CHRO)+'.txt', 'w')
    hicplus_hicRep      = open(root_dirR+"/hicplus_"+str(CHRO)+'.txt', 'w')
    '''
    #prepare the data to store
    if cell_lin == "Dros_cell":
        dm_test = GSE131811Module(batch_size = 1, percent = percentage, cell_No = cell_no)
    if cell_lin == "Human":
        dm_test = GSE130711Module(batch_size = 1, percent = percentage, cell_No = cell_no)

    dm_test.prepare_data()
    dm_test.setup(stage=CHRO)


    test_bar = tqdm(dm_test.test_dataloader())
    for s, sample in enumerate(test_bar):
        print(str(s)+"/"+str(dm_test.test_dataloader().dataset.data.shape[0]))
        if s >100:
            break

        data, target, _ = sample
        data = data.to(device)
        target = target.to(device)
        downs   = data[0][0]
        target  = target[0][0]
    
        #Pass through Models


        #Pass through hiedsr
        hiedsr_out = torch.zeros((PIECE_SIZE, PIECE_SIZE))
        hiedsr_out = hiedsrMod(data).cpu().detach()[0][0]

        '''
        # Pass through HicPlus,  the data should be padded first.
        hicplus_out = torch.zeros((PIECE_SIZE, PIECE_SIZE))
        temp = F.pad(data, (6, 6, 6, 6), mode = 'constant')
        hicplus_out = model_hicplus(temp).cpu().detach()[0][0]

        #Pass through Unet
        unet_out = torch.zeros((PIECE_SIZE, PIECE_SIZE))
        unet_out = model_unet(data).cpu().detach()[0][0]
        '''
        for i in range(0, 40):      #downs.shape[0]):
            bina = (40*s*RES)+(i*RES)
            bina_end = bina+RES
            bins_file.write(str(CHRO)+"\t"+str(bina)+"\t"+str(bina_end)+"\t"+str(bina)+"\n")
            for j in range(i, 40):  # downs.shape[1]):
                bina = (40*s*RES)+(i*RES)
                binb = (40*s*RES)+(j*RES)

                #down and target
                down_hic.write(str(CHRO)+"\t"+str(bina)+"\t"+str(CHRO)+"\t"+str(binb)+"\t"+str(int(downs[i,j]*100))+"\n")
                down_hicRep.write(str(bina)+"\t"+str(binb)+"\t"+str(int(downs[i,j]*100))+"\n")

                original_hic.write(str(CHRO)+"\t"+str(bina)+"\t"+str(CHRO)+"\t"+str(binb)+"\t"+str(int(target[i,j]*100))+"\n")
                original_hicRep.write(str(bina)+"\t"+str(binb)+"\t"+str(int(target[i,j]*100))+"\n")

                hiedsr_hic.write(str(CHRO)+"\t"+str(bina)+"\t"+str(CHRO)+"\t"+str(binb)+"\t"+str(int(hiedsr_out[i,j]*100))+"\n")
                hiedsr_hicRep.write(str(bina)+"\t"+str(binb)+"\t"+str(int(hiedsr_out[i,j]*100))+"\n")

                '''
                unet_hic.write(str(CHRO)+"\t"+str(bina)+"\t"+str(CHRO)+"\t"+str(binb)+"\t"+str(int(unet_out[i,j]*100))+"\n")
                unet_hicRep.write(str(bina)+"\t"+str(binb)+"\t"+str(int(unet_out[i,j]*100))+"\n")

                hicplus_hic.write(str(CHRO)+"\t"+str(bina)+"\t"+str(CHRO)+"\t"+str(binb)+"\t"+str(int(hicplus_out[i,j]*100))+"\n")
                hicplus_hicRep.write(str(bina)+"\t"+str(binb)+"\t"+str(int(hicplus_out[i,j]*100))+"\n")
                '''

    #close all the files for 3DChromatin_ReplicateQC
    down_hic.close()
    bins_file.close()
    original_hic.close()

    hiedsr_hic.close()

    '''
    unet_hic.close()
    hicplus_hic.close()
    '''

    #zip the produced files
    subprocess.run("gzip "+root_dir+"/original_"+str(CHRO), shell=True)
    subprocess.run("gzip "+root_dir+"/down_"+str(CHRO),     shell=True)
    subprocess.run("gzip "+root_dir+"/bins_"+str(CHRO)+".bed",     shell=True)

    subprocess.run("gzip "+root_dir+"/hiedsr_"+str(CHRO),  shell=True)

    '''
    subprocess.run("gzip "+root_dir+"/hicplus_"+str(CHRO),  shell=True)
    subprocess.run("gzip "+root_dir+"/unet_"+str(CHRO),  shell=True)
    '''

    #close all the .txt files for hicRep
    down_hicRep.close()
    original_hicRep.close()

    hiedsr_hicRep.close()

    '''
    unet_hicRep.close()
    hicplus_hicRep.close()
    '''

    tool_names   = ['hiedsr', 'down']
    #tool_names = ['down', 'hicplus', 'unet']
    BASE_STR = '/home/yw7bh/Projects/Denoise/GSE131811/hicqc_inputs/'
    sample_files = [
            root_dir+'/metric_hiedsr_'+str(CHRO)+".samples",
            root_dir+'/metric_down_'+str(CHRO)+".samples"
            #root_dir+'/metric_hicplus_'+str(CHRO)+".samples",
            #root_dir+'/metric_unet_'+str(CHRO)+".samples"
            ]

    pair_files  = [
            root_dir+'/metric_hiedsr_'+str(CHRO)+".pairs",
            root_dir+'/metric_down_'+str(CHRO)+".pairs"
            #root_dir+'/metric_hicplus_'+str(CHRO)+".pairs",
            #root_dir+'/metric_unet_'+str(CHRO)+".pairs"
            ]

    for tool_name, sample_fn, pair_fn in zip(tool_names, sample_files, pair_files):
        hic_metric_sample = open(sample_fn, 'w')
        hic_metric_pair   = open(pair_fn, 'w')
        SAMPLE_STRING="original\t"+BASE_STR+"original_"+str(CHRO)+".gz\n"+str(tool_name)+"\t"+BASE_STR+str(tool_name)+"_"+str(CHRO)+".gz"
        PAIR_STRING  = "original\t"+str(tool_name)
        hic_metric_sample.write(SAMPLE_STRING)
        hic_metric_pair.write(PAIR_STRING)

        #close the already written files.
        hic_metric_sample.close()
        hic_metric_pair.close()

    # below convert the .txt to the .matrix for hicrep to use:
    #Tools = ['down', 'original', 'hicplus', 'unet']
    Tools = ['hiedsr', 'down', 'original']
    res = RES
    for tool in Tools:
        file1 = root_dirR+"/"+tool+'_'+str(CHRO)+'.txt'
        file2 = root_dirR+"/"+tool+'_'+str(CHRO)+'.matrix'
        data = HicRepInput(file1, res)
        np.savetxt(file2, data)


