import sys
sys.path.append(".")
from Utils.loss import insulation as ins
import pdb
import os
import glob
import numpy as np
from numpy import inf
from tqdm import tqdm

import torch
import torch.nn.functional as F

#load data module
from ProcessData.PrepareData_tensor import GSE131811Module
from ProcessData.PrepareData_tensorH import GSE130711Module

#Load Models
import Models.Hiedsr_gan as hiedsr
#other models
import Models.hicsr   as hicsr
import Models.deephic as deephic
import Models.hicplus as hicplus
import Models.Unet_parts1 as unet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

methods = ['downs', 'hicplus', 'deephic', 'hicsr', 'unet',  'hiedsr', 'hiedsrgan']
colors  = ['black', 'silver', 'blue', 'darkviolet', 'red',  'coral', 'green']

getIns   = ins.computeInsulation().to(device)

#single chromsome information for test
RES        = 40000
PIECE_SIZE = 40

#below is the cell information for test
cell_lin = "Dros_cell"  # "Human" or "Dros_cell"
cell_no = 2
percentage = 0.75

#below two is used for pretrained models' paths
cell_not = 1
cell_lint = "Human"

file_inter = "Downsample_"+str(percentage)+"_"+cell_lint+str(cell_not)+"/"
#file_inter = "Downsample_"+str(percentage)+"/"

#Our models on multicom
hiedsrMod  = hiedsr.Generator().to(device)
file_path1 = "../pretrained/"+file_inter+"bestg_40kb_c40_s40_"+cell_lint+str(cell_not)+"_hiedsr.pytorch"
hiedsrMod.load_state_dict(torch.load(file_path1))
hiedsrMod.eval()

hiedsrganMod = hiedsr.Generator().to(device)
file_path1 = "../pretrained/"+file_inter+"bestg_40kb_c40_s40_"+cell_lint+str(cell_not)+"_hiedsrgan.pytorch"
hiedsrganMod.load_state_dict(torch.load(file_path1))
hiedsrganMod.eval()


model_hicsr   = hicsr.Generator(num_res_blocks=15).to(device)
file_path1  = "../pretrained/"+file_inter+"bestg_40kb_c40_s40_"+cell_lint+str(cell_not)+"_hicsr.pytorch"
model_hicsr.load_state_dict(torch.load(file_path1))
model_hicsr.eval()

model_deephic = deephic.Generator(scale_factor=1, in_channel=1, resblock_num=5).to(device)
file_path1 = "../pretrained/"+file_inter+"bestg_40kb_c40_s40_"+cell_lint+str(cell_not)+"_deephic.pytorch"
model_deephic.load_state_dict(torch.load(file_path1))
model_deephic.eval()

# other models on daisy
model_hicplus = hicplus.Net(40,28).to(device)
file_path1 = "../pretrained/"+file_inter+"bestg_40kb_c40_s40_"+cell_lint+str(cell_not)+"_hicplus.pytorch"
model_hicplus.load_state_dict(torch.load(file_path1))
model_hicplus.eval()

model_unet = unet.unet_2D().to(device)
file_path1 = "../pretrained/"+file_inter+"bestg_40kb_c40_s40_"+cell_lint+str(cell_not)+"_unet.pytorch"
model_unet.load_state_dict(torch.load(file_path1))
model_unet.eval()


if cell_lin == "Human":
    chros_all = [2, 6, 10, 12]  #[2, 6, 10, 12] for all test chromsomes
else:
    chros_all = [2, 6]

#filetering some unusual data
def filterNum(arr1):
    arr1 = np.array(arr1)
    arr1[arr1 == inf] = 0
    arr1[arr1 == -inf] = 0
    arr2 = np.nan_to_num(arr1)

    return arr2

#average across all the test chromosome
hicsr_all = []
deephic_all = []
hiedsr_all = []
hiedsrgan_all = []
down_all = []
hicplus_all = []
unet_all = []

for CHRO in chros_all:
    RES        = 40000
    PIECE_SIZE = 40
    # prepare the data to store
    if cell_lin == "Dros_cell":
        dm_test = GSE131811Module(batch_size = 1, percent = percentage, cell_No = cell_no)
    if cell_lin == "Human":
        dm_test = GSE130711Module(batch_size = 1, percent = percentage, cell_No = cell_no)

    dm_test.prepare_data( )
    dm_test.setup(stage = CHRO)

    full_insulation_dist = {
            'hicsr':[],
            'deephic':[],
            'hiedsr':[],
            'hiedsrgan':[],
            'down':[],
            'target':[],
            'hicplus':[],
            'unet':[]
            }

    directionality_comp = {
            'hicsr':[],
            'deephic':[],
            'hiedsr':[],
            'hiedsrgan':[],
            'down':[],
            'target':[],
            'hicplus':[],
            'unet':[]
            }

    def getTadBorderDists(x,y):
        nearest_distances = []
        for border1 in x:
            if border1 >50 and border1 <101:
                nearest = 9999
                for border2 in y:
                    dist = abs(border1-border2)
                    if dist < nearest:
                        nearest = dist
                nearest_distances.append(nearest)

        return nearest_distances

    STEP_SIZE = 40
    BUFF_SIZE = 40
    #pdb.set_trace()
    NUM_ITEMS = dm_test.test_dataloader().dataset.data.shape[0]
    test_bar = tqdm(dm_test.test_dataloader())
    for s, sample in enumerate(test_bar):
        print(str(s)+"/"+str(NUM_ITEMS))
        data, target, _ = sample
        data = data.to(device)
        target = target.to(device)
        downs   = data[0][0]
        target  = target[0][0]
        if target.sum()==0 or data.sum()==0:
            continue

        # Pass through Models

        #Pass through hiedsr
        hiedsr_out = torch.zeros((PIECE_SIZE, PIECE_SIZE))
        hiedsr_out = hiedsrMod(data).cpu().detach()[0][0]

        #Pass through hiedsrgan
        hiedsrgan_out = torch.zeros((PIECE_SIZE, PIECE_SIZE))
        hiedsrgan_out = hiedsrganMod(data).cpu().detach()[0][0]

        #Pass through Deephic
        deephic_out = torch.zeros((PIECE_SIZE, PIECE_SIZE))
        deephic_out = model_deephic(data).cpu().detach()[0][0]

        #Pass through HiCSR,  the data should be padded first.
        hicsr_out = torch.zeros((PIECE_SIZE, PIECE_SIZE))
        temp = F.pad(data, (6, 6, 6, 6), mode = 'constant')
        hicsr_out = model_hicsr(temp).cpu().detach()[0][0]

        # Pass through HicPlus,  the data should be padded first.
        hicplus_out = torch.zeros((PIECE_SIZE, PIECE_SIZE))
        temp = F.pad(data, (6, 6, 6, 6), mode = 'constant')
        hicplus_out = model_hicplus(temp).cpu().detach()[0][0]

        # Pass through Unet
        unet_out = torch.zeros((PIECE_SIZE, PIECE_SIZE))
        unet_out = model_unet(data).cpu().detach()[0][0]

        directionality_comp['down'].extend(getIns.forward(downs.reshape(1,1,40,40))[1][0][0][:].tolist())
        directionality_comp['target'].extend(getIns.forward(target.reshape(1,1,40,40))[1][0][0][:].tolist())


        directionality_comp['hicplus'].extend(getIns.forward(hicplus_out.reshape(1,1,40,40))[1][0][0][:].tolist())
        directionality_comp['unet'].extend(getIns.forward(unet_out.reshape(1,1,40,40))[1][0][0][:].tolist())


        directionality_comp['deephic'].extend(getIns.forward(deephic_out.reshape(1,1,40,40))[1][0][0][:].tolist())
        directionality_comp['hicsr'].extend(getIns.forward(hicsr_out.reshape(1,1,40,40))[1][0][0][:].tolist())
        directionality_comp['hiedsr'].extend(getIns.forward(hiedsr_out.reshape(1, 1, 40, 40))[1][0][0][:].tolist())
        directionality_comp['hiedsrgan'].extend(getIns.forward(hiedsrgan_out.reshape(1, 1, 40, 40))[1][0][0][:].tolist())


    directionality_comp['down'] = filterNum(directionality_comp['down'])
    directionality_comp['target'] = filterNum(directionality_comp['target'])

    directionality_comp['hicplus'] = filterNum(directionality_comp['hicplus'])
    directionality_comp['unet'] = filterNum(directionality_comp['unet'])

    directionality_comp['deephic'] = filterNum(directionality_comp['deephic'])
    directionality_comp['hicsr'] = filterNum(directionality_comp['hicsr'])
    directionality_comp['hiedsr'] = filterNum(directionality_comp['hiedsr'])
    directionality_comp['hiedsrgan'] = filterNum(directionality_comp['hiedsrgan'])

    down_direction    = np.linalg.norm(directionality_comp['down']-directionality_comp['target'])

    hicplus_direction    = np.linalg.norm(directionality_comp['hicplus']-directionality_comp['target'])
    unet_direction    = np.linalg.norm(directionality_comp['unet']-directionality_comp['target'])

    deephic_direction    = np.linalg.norm(directionality_comp['deephic']-directionality_comp['target'])
    hicsr_direction    = np.linalg.norm(directionality_comp['hicsr']-directionality_comp['target'])
    hiedsr_direction = np.linalg.norm(directionality_comp['hiedsr']-directionality_comp['target'])
    hiedsrgan_direction = np.linalg.norm(directionality_comp['hiedsrgan']-directionality_comp['target'])


    print("------"+str(cell_lin)+str(cell_no)+"_"+str(percentage)+"--Chro:"+str(CHRO)+"-------")
    print("down direction: "     +str(down_direction)+"\n"\
            "hicplus direction: "+str(hicplus_direction)+"\n"\
            "unet_direction: "+str(unet_direction)+"\n"\
            "deephic_direction: "+str(deephic_direction)+"\n"\
            "hicsr_direction: "  +str(hicsr_direction)+"\n"\
            "hiedsr_direction: " + str(hiedsr_direction)+"\n"\
            "hiedsrgan_direction: " + str(hiedsrgan_direction)+"\n"
          )
    down_all.append(down_direction)

    hicplus_all.append(hicplus_direction)
    unet_all.append(unet_direction)

    deephic_all.append(deephic_direction)
    hicsr_all.append(hicsr_direction)
    hiedsr_all.append(hiedsr_direction)
    hiedsrgan_all.append(hiedsrgan_direction)


down_o = sum(down_all)/len(down_all)

hicplus_o = sum(hicplus_all)/len(hicplus_all)
unet_o = sum(unet_all)/len(unet_all)

deephic_o = sum(deephic_all)/len(deephic_all)
hicsr_o = sum(hicsr_all)/len(hicsr_all)
hiedsr_o = sum(hiedsr_all)/len(hiedsr_all)
hiedsrgan_o = sum(hiedsrgan_all)/len(hiedsrgan_all)


outdir = "../Insolation_score"
outfile = outdir + "/multicom_" + cell_lin + str(cell_no) + "_" + str(percentage) + ".txt"

#The below two step to make sure the two out directories are new without any additional old information
if os.path.exists(outdir):
    globs = glob.glob(outfile)
    if outfile in globs:
        os.remove(outfile)
        print("\n================Trim the old information in output directories=============\n")

if not os.path.isdir(outdir):
    os.makedirs(outdir, exist_ok = True)

results = open(outfile, "w")

Tools = ['down', 'hicplus', 'unet', 'deephic', 'hicsr', 'hiedsr', 'hiedsrgan']
All = [str(down_all), str(hicplus_all), str(unet_all), str(deephic_all), str(hicsr_all), str(hiedsr_all), str(hiedsrgan_all)]
Ave = [str(down_o), str(hicplus_o), str(unet_o), str(deephic_o), str(hicsr_o), str(hiedsr_o), str(hiedsrgan_o)]

for tool, all, av in zip(Tools, All, Ave):
    results.write(tool+"All_chroms:\t"+all+"\t"+"Average: \t"+av+"\n")

results.close()




