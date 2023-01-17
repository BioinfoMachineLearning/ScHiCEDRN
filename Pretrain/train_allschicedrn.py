import sys
sys.path.append(".")
sys.path.append("../")
import numpy as np

#models need to be trained

from Pretrain.train_schicedrn_vH import hiedsr

if __name__ == "__main__":

    batch_size = 1
    percenList = [0.75, 0.45, 0.1, 0.02]
    celln = 1
    for percen in percenList:
    #train on hiedsr
        train_model = hiedsr(Gan = False, epoch = 400, batch_s = batch_size, cellN = celln, percentage = percen)
        train_model.fit_model()
        print("\n\nTraining hiedsr is done!!!\n")

        train_model = hiedsr(Gan = True, epoch = 400, batch_s = batch_size, cellN = celln, percentage = percen)
        train_model.fit_model()
        print("\n\nTraining hiedsrgan is done!!!\n")















