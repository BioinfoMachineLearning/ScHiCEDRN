import sys
sys.path.append('.')

import torch
from tqdm import tqdm
import Models.schicedrn_gan as hiedsr
from ProcessData.PrepareData_tensor import GSE131811Module
from ProcessData.PrepareData_tensorH import GSE130711Module
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Training process for ScHiCedsr or ScHiCedsrgan.')
    parser.add_argument('-g', '--gan', type = int, choices=[0, 1], default=0)
    parser.add_argument('-e', '--epoch', type = int, default=300)
    parser.add_argument('-b', '--batch_size', type = int, default=1)
    parser.add_argument('-n', '--celln', type = int, default=1, help='The cell number of the cell-line you want to train')
    parser.add_argument('-l', '--celline', type = str, default='Human')
    parser.add_argument('-p', '--percent', type = float, default='0.75', help = 'The down-sampling ratio for the raw input dataset, the value should be equal or larger than 0.02 but not larger than 1.0')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # load data
    dm_test = None
    if args.celline == "Dros_cell":
        dm_test = GSE131811Module(batch_size=args.batch_size, percent=args.percent, cell_No=args.celln)
    if args.celline == "Human":
        dm_test = GSE130711Module(batch_size=args.batch_size, percent=args.percent, cell_No=args.celln)
    dm_test.prepare_data()
    dm_test.setup(stage='test')


    file_inter = "Downsample_" + str(args.percent) + "_" + "Human" + str(1) + "/"
    if args.gan == 0:
        hiedsrMod = hiedsr.Generator().to(device)
        file_path1 ="./Model_Weight/" + file_inter + "bestg_40kb_c40_s40_" + "Human" + str(1) + "_hiedsr.pytorch"
        hiedsrMod.load_state_dict(torch.load(file_path1))
        hiedsrMod.eval()
    else:
        hiedsrMod = hiedsr.Generator().to(device)
        file_path1 = "./Model_Weight/" + file_inter + "bestg_40kb_c40_s40_" + "Human" + str(1)  + "_hiedsrgan.pytorch"
        hiedsrMod.load_state_dict(torch.load(file_path1))
        hiedsrMod.eval()

    test_loader = dm_test.test_dataloader()
    test_bar = tqdm(test_loader)
    with torch.no_grad():
        for lr, hr, inds in test_bar:
            batch_size = lr.size(0)
            lr = lr.to(device)
            hr = hr.to(device)
            # data, full_target, info = epoch
            out = hiedsrMod(lr)