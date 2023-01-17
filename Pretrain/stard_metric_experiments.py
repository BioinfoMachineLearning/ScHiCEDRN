import sys
sys.path.append(".")
import numpy as np
import os
import matplotlib.pyplot as plt
import torch

#Our models
import Models.schicedrn_gan as hiedsr
#other models
import Models.hicsr   as hicsr
import Models.deephic as deephic
import Models.hicplus as hicplus
import Models.Loopenhance_parts1 as unet

from Utils import stard_metrics as vm
from ProcessData.PrepareData_tensor import GSE131811Module
from ProcessData.PrepareData_tensorH import GSE130711Module
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#below is the cell information for test
cell_lin ="Dros_cell"
cell_no = 2
percentage = 0.02

#below two is used for pretrained models' paths
cell_not = 1
cell_lint = "Human"

file_inter = "Downsample_"+str(percentage)+"_"+cell_lint+str(cell_not)+"/"
#file_inter = "Downsample_"+str(percentage)+"/"

#load data
if cell_lin == "Dros_cell":
    dm_test = GSE131811Module(batch_size = 1, percent = percentage, cell_No=cell_no)
if cell_lin == "Human":
    dm_test = GSE130711Module(batch_size=1, percent=percentage, cell_No=cell_no)
dm_test.prepare_data()
dm_test.setup(stage='test')

ds     = dm_test.test_dataloader().dataset.data[65:66]
target = dm_test.test_dataloader().dataset.target[65:66]


#Our models on multicom
hiedsrMod  = hiedsr.Generator().to(device)
file_path1 = "../pretrained/"+file_inter+"bestg_40kb_c40_s40_"+cell_lint+str(cell_not)+"_hiedsr.pytorch"
hiedsrMod.load_state_dict(torch.load(file_path1))
hiedsrMod.eval()

hiedsrganMod = hiedsr.Generator().to(device)
file_path1 = "../pretrained/"+file_inter+"bestg_40kb_c40_s40_"+cell_lint+str(cell_not)+"_hiedsrgan.pytorch"
hiedsrganMod.load_state_dict(torch.load(file_path1))
hiedsrganMod.eval()

# otheir models on muticom  (hicsr and nicplus these two will lead Hout = Hin - 12, the rest models will have Hout = Hin)
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
file_path1 = "../pretrained/"+file_inter+"bestg_40kb_c40_s40_"+cell_lin+str(cell_not)+"_hicplus.pytorch"
model_hicplus.load_state_dict(torch.load(file_path1))
model_hicplus.eval()

model_unet = unet.unet_2D().to(device)
file_path1 = "../pretrained/"+file_inter+"bestg_40kb_c40_s40_"+cell_lin+str(cell_not)+"_unet.pytorch"
model_unet.load_state_dict(torch.load(file_path1))
model_unet.eval()

#pass through models

v_m ={}
chro = "test"
chro1 = 1
#compute vision metrics

print("hiedsr")
visionMetrics = vm.VisionMetrics()
visionMetrics.setDataset(chro, percent=percentage, cellN=cell_no, cell_line=cell_lin)
v_m[chro1, 'hiedsr']=visionMetrics.getMetrics(model=hiedsrMod, spliter="hiedsr")

print("hiedsrgan")
visionMetrics = vm.VisionMetrics()
visionMetrics.setDataset(chro, percent=percentage, cellN=cell_no, cell_line=cell_lin)
v_m[chro1, 'hiedsrgan']=visionMetrics.getMetrics(model=hiedsrganMod, spliter="hiedsrgan")

print("deephic")
visionMetrics = vm.VisionMetrics()
visionMetrics.setDataset(chro, percent=percentage, cellN=cell_no, cell_line=cell_lin)
v_m[chro1, 'deephic']=visionMetrics.getMetrics(model=model_deephic, spliter="deephic")
 
print("HiCSR")
visionMetrics = vm.VisionMetrics()
visionMetrics.setDataset(chro, percent=percentage, cellN=cell_no, cell_line=cell_lin)
v_m[chro1, 'hicsr']=visionMetrics.getMetrics(model=model_hicsr, spliter="hicsr")

print("unet")
visionMetrics = vm.VisionMetrics()
visionMetrics.setDataset(chro, percent=percentage, cellN=cell_no, cell_line=cell_lin)
v_m[chro1, 'unet']=visionMetrics.getMetrics(model=model_unet, spliter="unet")

print("hicplus")
visionMetrics = vm.VisionMetrics()
visionMetrics.setDataset(chro, percent=percentage, cellN=cell_no, cell_line=cell_lin)
v_m[chro1, 'hicplus']=visionMetrics.getMetrics(model=model_hicplus, spliter="hicplus")

model_names  = ['hiedsr', 'hiedsrgan',  'deephic', 'hicsr', 'unet', 'hicplus']
#model_names = ['unet', 'hicplus']
metric_names = ['ssim', 'psnr', 'mse', 'snr', 'gds']


# below is to record the gds values for each cell
gds_path = cell_lin+str(cell_no)+"_"+str(percentage)
gds_dir = "../GenomeDISCO"
if not os.path.exists(gds_dir):
    os.makedirs(gds_dir, exist_ok = True)
record_gds = open(gds_dir+"/"+"GenomeDISCO_multicom.txt", "a")
record_gds.write("\n"+gds_path+":\n")

cell_text = []
for mod_nm in model_names:
    met_list = []
    for met_nm in metric_names:
        met_list.append("{:.4f}".format(np.mean(v_m[chro1, mod_nm]['pas_'+str(met_nm)])))
        if met_nm == 'gds':
            record_gds.write(mod_nm + ":\t" + str(np.mean(v_m[chro1, mod_nm]['pas_' + str(met_nm)])) + "\n")
    cell_text.append(met_list)
record_gds.close()

plt.subplots_adjust(left=0.2, top=0.8)
plt.table(cellText=cell_text, rowLabels=model_names, colLabels=metric_names, loc='top')
#plt.title(chro)
plt.show()



