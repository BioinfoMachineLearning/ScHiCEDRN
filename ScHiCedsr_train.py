import sys
sys.path.append('.')
sys.path.append('../')

from Pretrain.train_hiedsr_vH import hiedsr
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Training process for ScHiCedsr or ScHiCedsrgan.')
