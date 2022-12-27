import sys
sys.path.append('.')

from Pretrain.train_hiedsr_vH import hiedsr
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Training process for ScHiCedsr or ScHiCedsrgan.')
    parser.add_argument('-g', '--gan', type = int, choices=[0, 1], default=0)
    parser.add_argument('-e', '--epoch', type = int, default=300)
    parser.add_argument('-b', '--batch_size', type = int, default=64)
    parser.add_argument('-n', '--celln', type = int, default=1, help='The cell number of the cell-line you want to train')
    parser.add_argument('-l', '--celline', type = str, default='Human')
    parser.add_argument('-p', '--percent', type = float, default='0.75', help = 'The down-sampling ratio for the raw input dataset, the value should be equal or larger than 0.02 but not larger than 1.0')
    args = parser.parse_args()

    if args.gan == 0:
        train_model = hiedsr(Gan=False, epoch=args.epoch, batch_s=args.batch_size, cellN=args.celln, celline=args.celline, percentage=args.percent)
        train_model.fit_model()
        print("\n\nTraining ScHiCedsr is done!!!\n")
    else:
        train_model = hiedsr(Gan=True, epoch=args.epoch, batch_s=args.batch_size, cellN=args.celln, celline=args.celline, percentage=args.percent)
        train_model.fit_model()
        print("\n\nTraining ScHiCedsrgan is done!!!\n")


