import sys
sys.path.append(".")

#our models
import Models.schicedrn_gan as hiedsr

#other models
import Models.hicplus as hicplus
import Models.hicsr   as hicsr
import Models.deephic as deephic

import Models.Loopenhance_parts1 as unet

import os
import tmscoring
import glob
import subprocess
import shutil
import pdb
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from ProcessData.PrepareData_tensor import GSE131811Module
from ProcessData.PrepareData_tensorM import GSE130711Module
import torch
import torch.nn.functional as F

device = torch.device('cpu')
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

PIECE_SIZE = 40

def buildFolders():
    if not os.path.exists('3D_Mod'):
        os.makedirs('3D_Mod')
    if not os.path.exists('3D_Mod/Constraints'):
        os.makedirs('3D_Mod/Constraints')
    if not os.path.exists('3D_Mod/output'):
        os.makedirs('3D_Mod/output')
    if not os.path.exists('3D_Mod/Parameters'):
        os.makedirs('3D_Mod/Parameters')

def convertChroToConstraints(chro,
                            cell_line="Human",
                            res=40000,
                            plotmap = True):
    #bin_num = int(CHRO_LENGTHS[chro]/res)
    #print(bin_num)

    if cell_line == "Dros_cell":
        dm_test = GSE131811Module(batch_size = 1, percent = 0.75, cell_No = 1)
    if cell_line == "Human":
        dm_test = GSE130711Module(batch_size = 1, percent = 0.75, cell_No = 1)
    dm_test.prepare_data()
    dm_test.setup(stage=chro)

    # without utility
    #target_chro    = np.zeros((bin_num, bin_num))
    #down_chro      = np.zeros((bin_num, bin_num))
    #hiedsrgan_chro   = np.zeros((bin_num, bin_num))

    # below two is used for pretrained models' paths
    cell_not = 1
    cell_lint = "Human"
    percentage = 0.75
    file_inter = "Downsample_" + str(percentage) + "_" + cell_lint + str(cell_not) + "/"


    #Our models on multicom
    hiedsrMod  = hiedsr.Generator().to(device)
    file_path1 = "../pretrained/"+file_inter+"bestg_40kb_c40_s40_"+cell_lint+str(cell_not)+"_hiedsr.pytorch"
    hiedsrMod.load_state_dict(torch.load(file_path1))
    hiedsrMod.eval()

    hiedsrganMod = hiedsr.Generator().to(device)
    #file_path1 = "../pretrained/Downsample_0.75/bestg_40kb_c40_s40_Dros_cell1_hiedsrgan.pytorch"
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
    model_hicplus = hicplus.Net(40, 28).to(device)
    file_path1 = "../pretrained/" + file_inter + "bestg_40kb_c40_s40_" + cell_lint + str(cell_not) + "_hicplus.pytorch"
    model_hicplus.load_state_dict(torch.load(file_path1))
    model_hicplus.eval()

    model_unet = unet.unet_2D().to(device)
    file_path1 = "../pretrained/" + file_inter + "bestg_40kb_c40_s40_" + cell_lint + str(cell_not) + "_unet.pytorch"
    model_unet.load_state_dict(torch.load(file_path1))
    model_unet.eval()


    NUM_ENTRIES = dm_test.test_dataloader().dataset.data.shape[0]
    test_bar = tqdm(dm_test.test_dataloader())
    block = 0
    region = 0
    for s, sample in enumerate(test_bar):
        if s > 7:
            break

        print(str(s)+"/"+str(NUM_ENTRIES))
        data, target, _ = sample
        data = data.to(device)
        target = target.to(device)

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


        data   = data[0][0]
        target = target[0][0]

        if s in range(1, 8) and s % 6 == 0:
            if plotmap:
                region = region + 1
                region_start = region*(1.6*6)
                region_end = region_start+1.6

                fig, ax = plt.subplots(2, 6, figsize = (30, 10))

                for i in range(0, 2):
                    for j in range(0, 6):
                        ax[i, j].set_xticks([])
                        ax[i, j].set_yticks([])

                colorN = ['Reds', 'Blues', 'plasma', 'viridis']
                ax[0, 0].imshow(data, cmap = colorN[2])
                ax[0, 0].set_title("DownSampled", fontsize=20)
                ax[0, 0].set_ylabel("Chro"+str(chro)+" "+"{:.2f}".format(region_start)+"-"+"{:.2f}".format(region_end), fontsize=20)

                ax[0, 1].imshow(deephic_out, cmap=colorN[2])
                ax[0, 1].set_title("DeepHiC", fontsize=20)

                ax[0, 2].imshow(hicsr_out, cmap = colorN[2])
                ax[0, 2].set_title("HiCSR", fontsize=20)

                ax[0, 3].imshow(hiedsr_out, cmap = colorN[2])
                ax[0, 3].set_title("ScHiCedsr(ours)", fontsize = 20)

                ax[0, 4].imshow(hiedsrgan_out, cmap = colorN[2])
                ax[0, 4].set_title("ScHiCedsrgan(ours)", fontsize = 20)

                ax[0, 5].imshow(target, cmap=colorN[2])
                ax[0, 5].set_title("Target", fontsize=20)

                ax[1, 0].imshow(data[8:21, 8:21], cmap = colorN[2])
                ax[1, 0].set_ylabel("Chro" + str(chro) + " " + "{:.2f}".format(region_start+0.32) + "-" + "{:.2f}".format(region_start+0.8), fontsize=20)
                ax[1, 1].imshow(deephic_out[8:21, 8:21], cmap = colorN[2])
                ax[1, 2].imshow(hicsr_out[8:21, 8:21], cmap = colorN[2])
                ax[1, 3].imshow(hiedsr_out[8:21, 8:21], cmap = colorN[2])
                ax[1, 4].imshow(hiedsrgan_out[8:21, 8:21], cmap = colorN[2])
                ax[1, 5].imshow(target[8:21, 8:21], cmap = colorN[2])

                plt.show()

                #plot difference:
                fig, ax = plt.subplots(1, 5)
                Diff_Down = abs(target - data)
                Diff_Deephic = abs(target - deephic_out)
                Diff_Hicsr = abs(target - hicsr_out)
                Diff_Hiedsr = abs(target - hiedsr_out)
                Diff_Hiedsrgan = abs(target - hiedsrgan_out)

                for i in range(0, 5):
                    ax[i].set_xticks([])
                    ax[i].set_yticks([])

                ax[0].imshow(Diff_Down, cmap = colorN[3])
                ax[0].set_ylabel("Chro" + str(chro) + " " + "{:.2f}".format(region_start) + "-" + "{:.2f}".format(region_end))
                ax[0].set_title("DownSampled")

                ax[1].imshow(Diff_Deephic, cmap = colorN[3])
                ax[1].set_title("HiCPlus")

                ax[2].imshow(Diff_Hicsr, cmap = colorN[3])
                ax[2].set_title("HiCSR")

                ax[3].imshow(Diff_Hiedsr, cmap = colorN[3])
                ax[3].set_title("ScHiCedsr(ours)")

                ax[4].imshow(Diff_Hiedsrgan, cmap = colorN[3])
                ax[4].set_title("ScHiCedsrgan(ours)")

                plt.show()

            else:
                print("\n@@@@@@@@@@@@@ Nothing to plot @@@@@@@@@@@@@@@@@@@\n")


        if s > 100000:  # every 4.8MB for chro12
            block = block+1

            target_const_name   = "3D_Mod/Constraints/chro_"+str(chro)+"_target_"+str(block-1)+"_"
            data_const_name     = "3D_Mod/Constraints/chro_"+str(chro)+"_data_"+str(block-1)+"_"

            # training on multicom
            hiedsr_const_name = "3D_Mod/Constraints/chro_" + str(chro) + "_hiedsr_" + str(block-1) + "_"
            hiedsrgan_const_name  = "3D_Mod/Constraints/chro_"+str(chro)+"_hiedsrgan_"+str(block-1)+"_"
            deephic_const_name = "3D_Mod/Constraints/chro_" + str(chro) + "_deephic_" + str(block-1) + "_"
            hicarn_const_name = "3D_Mod/Constraints/chro_" + str(chro) + "_hicarn_" + str(block-1) + "_"
            hicsr_const_name = "3D_Mod/Constraints/chro_" + str(chro) + "_hicsr_" + str(block-1) + "_"


            hicplus_const_name = "3D_Mod/Constraints/chro_" + str(chro) + "_hicplus_" + str(block - 1) + "_"
            unet_const_name = "3D_Mod/Constraints/chro_" + str(chro) + "_unet_" + str(block - 1) + "_"


            if os.path.exists(target_const_name):
                print("\n=================Trim the exists same files ==================\n")
                shutil.rmtree("3D_Mod/Constraints")
            if not os.path.exists("3D_Mod/Constraints"):
                print("@@@@@@@@@@@@@@@@@@@ build the storing directory @@@@@@@@@@@@@@@@@@\n")
                os.makedirs("3D_Mod/Constraints", exist_ok = True)

            target_constraints  = open(target_const_name, 'w')
            data_constraints    = open(data_const_name, 'w')


            hiedsr_constraints = open(hiedsr_const_name, 'w')
            hiedsrgan_constraints = open(hiedsrgan_const_name, 'w')
            deephic_constraints = open(deephic_const_name, 'w')
            hicarn_constraints = open(hicarn_const_name, 'w')
            hicsr_constraints = open(hicsr_const_name, 'w')


            hicplus_constraints = open(hicplus_const_name, 'w')
            unet_constraints = open(unet_const_name, 'w')


            for i in range(0, data.shape[0]):
                for j in range(i, data.shape[1]):
                    data_constraints.write(str(i)+"\t"+str(j)+"\t"+str(data[i,j].item())+"\n")
                    target_constraints.write(str(i)+"\t"+str(j)+"\t"+str(target[i,j].item())+"\n")


                    hiedsr_constraints.write(str(i)+"\t"+str(j)+"\t"+str(hiedsr_out[i,j].item())+"\n")
                    hiedsrgan_constraints.write(str(i)+"\t"+str(j)+"\t"+str(hiedsrgan_out[i,j].item())+"\n")
                    deephic_constraints.write(str(i)+"\t"+str(j)+"\t"+str(deephic_out[i,j].item())+"\n")
                    hicsr_constraints.write(str(i)+"\t"+str(j)+"\t"+str(hicsr_out[i,j].item())+"\n")


                    hicplus_constraints.write(str(i)+"\t"+str(j)+"\t"+str(hicplus_out[i,j].item())+"\n")
                    unet_constraints.write(str(i)+"\t"+str(j)+"\t"+str(unet_out[i, j].item()) + "\n")


            target_constraints.close()
            data_constraints.close()


            hiedsr_constraints.close()
            hiedsrgan_constraints.close()
            deephic_constraints.close()
            hicarn_constraints.close()
            hicsr_constraints.close()


            hicplus_constraints.close()
            unet_constraints.close()


def buildParameters(chro,
                cell_line="Human",
                res=40000):
    constraints  = glob.glob("3D_Mod/Constraints/chro_"+str(chro)+"_*")
    for constraint in  constraints:
        suffix = constraint.split("/")[-1]
        stri = """NUM = 3\r
OUTPUT_FOLDER = 3D_Mod/output/\r
INPUT_FILE = """+constraint+"""\r
CONVERT_FACTOR = 0.6\r
VERBOSE = true\r
LEARNING_RATE = 1\r
MAX_ITERATION = 10000\n"""
        param_f = open("3D_Mod/Parameters/"+suffix, 'w')
        param_f.write(stri)


JAR_LOCATION = "other_tools/examples/3DMax.jar"
def runSegmentParams(chro, position_index):
    for struc in ['data', 'target', 'hiedsrgan']:
        subprocess.run("java -Xmx5000m -jar "+JAR_LOCATION+" 3D_Mod/Parameters/chro_"+str(chro)+"_"+struc+"_"+str(position_index)+"_", shell=True)

def runParams(chro):
    if not os.path.exists(JAR_LOCATION):
        subprocess.run("git clone https://github.com/BDM-Lab/3DMax.git other_tools", shell = True)

    PdbPath = "3D_Mod/output/chro_"+str(chro)+"_target_0"+"_*.pdb"
    if os.path.exists(PdbPath):
        print("\n=================Trim the exists output files ==================\n")
        shutil.rmtree("3D_Mod/output")

    if not os.path.exists("3D_Mod/output"):
        print("@@@@@@@@@@@@@@@@@@@ build the output directory @@@@@@@@@@@@@@@@@@\n")
        os.makedirs("3D_Mod/output", exist_ok = True)

    params = glob.glob("3D_Mod/Parameters/chro_"+str(chro)+"_*")
    for par in params:
        subprocess.run("java -Xmx5000m -jar "+JAR_LOCATION+" "+par, shell=True)

def getSegmentTMScores(chro, position_index):
    data_strucs     = glob.glob("3D_Mod/output/chro_"+str(chro)+"_data_"+str(position_index)+"_*.pdb")
    target_strucs   = glob.glob("3D_Mod/output/chro_"+str(chro)+"_target_"+str(position_index)+"_*.pdb")

    hiedsr_strucs = glob.glob("3D_Mod/output/chro_" + str(chro) + "_hiedsr_" + str(position_index) + "_*.pdb")
    hiedsrgan_strucs  = glob.glob("3D_Mod/output/chro_"+str(chro)+"_hiedsrgan_"+str(position_index)+"_*.pdb")
    deephic_strucs = glob.glob("3D_Mod/output/chro_" + str(chro) + "_deephic_" + str(position_index) + "_*.pdb")
    hicsr_strucs = glob.glob("3D_Mod/output/chro_" + str(chro) + "_hicsr_" + str(position_index) + "_*.pdb")

    hicplus_strucs = glob.glob("3D_Mod/output/chro_" + str(chro) + "_hicplus_" + str(position_index) + "_*.pdb")
    unet_strucs = glob.glob("3D_Mod/output/chro_" + str(chro) + "_unet_" + str(position_index) + "_*.pdb")

    struc_types      = [data_strucs, hiedsr_strucs, hiedsrgan_strucs, deephic_strucs, hicsr_strucs, hicplus_strucs, unet_strucs, target_strucs]
    #struc_types = [data_strucs,  hicplus_strucs, unet_strucs, target_strucs]

    struc_type_names = ['data_strucs', 'hiedsr_strucs', 'hiedsrgan_strucs', 'deephic_strucs',  'hicsr_strucs', 'hicplus_strucs', 'unet_strucs', 'target_strucs']
    #struc_type_names = ['data_strucs', 'hicplus_strucs', 'unet_strucs', 'target_strucs']


    internal_scores = {'data_strucs':[],
                    'hiedsr_strucs':[],
                    'hiedsrgan_strucs':[],
                    'deephic_strucs':[],
                    'hicsr_strucs':[],
                    'hicplus_strucs':[],
                    'unet_strucs':[],
                    'target_strucs':[]
                    }

    '''
    internal_scores = {'data_strucs':[],
                    'hicplus_strucs':[],
                    'unet_strucs':[],
                    'target_strucs':[]}
    '''

    for struc_type, struc_type_name in zip(struc_types, struc_type_names):
        for i, data_a in enumerate(struc_type):
            for j, data_b in enumerate(struc_type):
                if not struc_type_name in internal_scores.keys():
                    internal_scores[struc_type_name] = []
                if i>=j:
                    continue
                else:
                    alignment = tmscoring.TMscoring(data_a, data_b)
                    alignment.optimise()
                    indiv_tm = alignment.tmscore(**alignment.get_current_values())

                    indiv_tm = np.array(indiv_tm)
                    indiv_tm = np.nan_to_num(indiv_tm)

                    internal_scores[struc_type_name].append(indiv_tm)


    relative_scores = {'data_strucs':[],
                    'hiedsr_strucs':[],
                    'hiedsrgan_strucs':[],
                    'deephic_strucs':[],
                    'hicsr_strucs':[],
                    'hicplus_strucs':[],
                    'unet_strucs':[]}



    for struc_type, struc_type_name in zip(struc_types, struc_type_names):
        if struc_type_name == 'target_strucs':
             continue
        for i, data_a in enumerate(struc_type):
            for j, data_b in enumerate(target_strucs):
                alignment = tmscoring.TMscoring(data_a, data_b)
                alignment.optimise()
                indiv_tm  = alignment.tmscore(**alignment.get_current_values())

                indiv_tm = np.array(indiv_tm)
                indiv_tm = np.nan_to_num(indiv_tm)

                relative_scores[struc_type_name].append(indiv_tm)
    return relative_scores, internal_scores

def getTMScores(chro):


    internal_scores = {'data_strucs':[],
                      'hiedsr_strucs':[],
                      'hiedsrgan_strucs':[],
                      'deephic_strucs':[],
                      'hicsr_strucs':[],
                      'hicplus_strucs':[],
                      'unet_strucs':[],
                      'target_strucs':[]}

    '''
    internal_scores = {'data_strucs': [],
                       'hicplus_strucs': [],
                       'unet_strucs': [],
                       'target_strucs':[]}
    '''


    relative_scores = {'data_strucs':[],
                       'hiedsr_strucs':[],
                       'hiedsrgan_strucs':[],
                       'deephic_strucs':[],
                       'hicsr_strucs':[],
                       'hicplus_strucs':[],
                       'unet_strucs':[]}



    getSampleNum = lambda a: a.split("_")[-2]  # to find the postion we want to compare
    for position_index in list(map(getSampleNum, glob.glob("3D_Mod/Parameters/chro_"+str(chro)+"_*"))):
        temp_relative_scores, temp_internal_scores = getSegmentTMScores(chro, position_index)
        for key in temp_relative_scores.keys():
            relative_scores[key].extend(temp_relative_scores[key])
        for key in temp_internal_scores.keys():
            internal_scores[key].extend(temp_internal_scores[key])

    if not os.path.exists("3D_Mod/Scores"):
        print("@@@@@@@@@@@@@@@@@@@ build the Score directory @@@@@@@@@@@@@@@@@@\n")
        os.makedirs("3D_Mod/Scores", exist_ok = True)
    record_scores = open("3D_Mod/Scores/chro_"+str(chro)+".txt", "w")
    record_scores.write("INTERNAL SCORES\n")
    print("INTERNAL SCORES")
    for key in internal_scores.keys():
        print(key+":\t"+str(np.mean(internal_scores[key])))
        record_scores.write("\t"+key+":\t"+str(np.mean(internal_scores[key]))+"\n")

    print("RELATIVE SCORES")
    record_scores.write("RELATIVE SCORES\n")
    for key in relative_scores.keys():
        print(key+":\t"+str(np.mean(relative_scores[key])))
        record_scores.write("\t"+key+":\t"+str(np.mean(relative_scores[key]))+"\n")

    return relative_scores, internal_scores

def viewModels():
    struc_index=0
    chro=2  # 2, 6, 12(the chrom best to plot)
    models = glob.glob("3D_Mod/output/chro_"+str(chro)+"_*_"+str(struc_index)+"_*.pdb")
    subprocess.run("pymol "+' '.join(models),  shell=True)

def parallelScatter(chrom):
    colorlist = ['crimson', 'forestgreen', 'lightseagreen', 'bisque', 'yellowgreen', 'teal', 'steelblue', 'violet', 'orange']
    #relative, internal = getTMScores(4)
    chros = [chrom]     # here we first test on chrom 2, all the test list chros = [2, 6, 10, 12]
    relative_data = []
    internal_data = []
    for chro in chros:
        relative, internal = getTMScores(chro)
        for key in relative.keys():

            if key == "data_strucs":
                continue
            else:
                relative_data.append(relative[key])
        for key in internal.keys():

            if key == "data_strucs":
                continue
            elif key == "target_strucs":
                continue
            else:
                internal_data.append(internal[key])
    # pdb.set_trace()

    #relative
    fig, ax = plt.subplots()   # plot by box with std
    bp = ax.boxplot(relative_data, 
            positions=[1,2,3,4,5,6],    #for all chroms [1,2,4,5, 7,8, 10,11]
            patch_artist=True)

    for b, box in enumerate(bp['boxes']):
        box.set(facecolor = colorlist[b])

    ax.set_xticks([3.5])  #4.5, 7.5, 10.5
    ax.set_xticklabels(['Chro'+str(chrom)]) #'Chro6', 'Chro10', 'Chro12'
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    ax.set_ylabel("TM-Score")
    ax.set_title("Similarity to Target")
    plt.show()

    fig, ax = plt.subplots()
    bp = ax.boxplot(internal_data, 
            positions= [1,2,3,4,5,6],
            patch_artist=True)

    boxLenght = len(bp['boxes'])
    for b, box in enumerate(bp['boxes']):
        #if b != (boxLenght-1):
        box.set(facecolor = colorlist[b])
        #else:
            #box.set(facecolor = colorlist[-1])

    ax.set_xticks([3.5])   #[2,6,10,12]
    ax.set_xticklabels(['Chro'+str(chrom)])  #['Chro2', 'Chro6', 'Chro10', 'Chro12']
    #ax.spines['top'].set_visible(False) #the line noting the data area boundaries, a figures has four boundary lines
    #ax.spines['right'].set_visible(False)
    ax.set_ylabel("TM-Score")
    ax.set_title("Model Consistency")
    plt.show()

    # pdb.set_trace()

if __name__ == "__main__":

    #buildFolders()
    for chro in [2, 6, 10, 12]: # 2, 6,10,12
        #convertChroToConstraints(chro, plotmap = True)
        #buildParameters(chro)
        #runParams(chro)
        #getTMScores(2)
        parallelScatter(chrom = chro)
    #viewModels()














































































































'''
CHRO_LENGTHS={
        1:249250621,
        2:243199373,
        3:198022430,
        4:191154276,
        5:180915260,
        6:171115067,
        7:159138663,
        8:146364022,
        9:141213431,
        10:135534747,
        11:135006516,
        12:133851895,
        13:115169878,
        14:107349540,
        15:102531392,
        16:90354753,
        17:81195210,
        18:78077248,
        19:59128983,
        20:63025520,
        21:48129895,
        22:51304566
        }
'''
