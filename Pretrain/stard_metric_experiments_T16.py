import sys
sys.path.append(".")
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

#Our models
import Models.schicedrn_ganT as hiedsr
#other models
from Utils import stard_metrics as vm
from ProcessData.PrepareData_tensor import GSE131811Module
from ProcessData.PrepareData_tensorH import GSE130711Module
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#below is the cell information for test
cell_lin = 'Human'
cell_no = 1
percentage = 0.02

#below two is used for pretrained models' paths
cell_not = 1
cell_lint = "HumanT16_"
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


file_temp = "../pretrained/"+file_inter+"bestg_40kb_c40_s40_"+cell_lint+str(cell_not)
#Our models on multicom
hiedsrMod  = hiedsr.Generator().to(device)
#file_path1 = "../pretrained/"+file_inter+"bestg_40kb_c40_s40_"+cell_lin+str(cell_not)+"_hiedsr.pytorch"
file_path1 = file_temp+"_hiedsr.pytorch"
hiedsrMod.load_state_dict(torch.load(file_path1))
hiedsrMod.eval()

v_m ={}
chro = "test"
chro1 = 1
#compute vision metrics

print("hiedsr")
visionMetrics = vm.VisionMetrics()
visionMetrics.setDataset(chro, percent=percentage, cellN=cell_no, cell_line=cell_lin)
v_m[chro1, 'hiedsr']=visionMetrics.getMetrics(model=hiedsrMod, spliter="hiedsr")

#model_names  = ['hiedsr', 'hiedsrgan', 'hicarn', 'deephic', 'hicsr']
model_names = ['hiedsr']
metric_names = ['ssim', 'psnr', 'mse', 'snr', 'pcc', 'spc', 'gds']

# below is to record the gds values for each cell
gds_path = cell_lint+str(cell_no)+"_"+str(percentage)
gds_dir = "../GenomeDISCO_Test"
if not os.path.exists(gds_dir):
    os.makedirs(gds_dir, exist_ok = True)
record_gds = open(gds_dir+"/"+"GenomeDISCO_multicom.txt", "a")
record_gds.write("\n"+gds_path+":\n")

cell_text = []
for mod_nm in model_names:
    met_list = []
    for met_nm in metric_names:
        met_list.append("{:.4f}".format(np.mean(v_m[chro1, mod_nm]['pas_'+str(met_nm)])))
        record_gds.write(mod_nm + ":\t" + str(np.mean(v_m[chro1, mod_nm]['pas_' + str(met_nm)])) + "\n")
    cell_text.append(met_list)
record_gds.close()

plt.subplots_adjust(left=0.2, top=0.8)
plt.table(cellText=cell_text, rowLabels=model_names, colLabels=metric_names, loc='top')
#plt.title(chro)
plt.show()



