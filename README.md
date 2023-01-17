# Single Cell HiC Data Enhancement
Single cell 3D genome modeling tools developed in the Bioinformatics and Machine Learning Lab

<h1 align="center">ScHiCEDRN</h1>

<img
  src="./showing.jpg"
  alt="The method used in our paper"
  title="ScHiCedsr and ScHiCedsrgan"
  style="border: 1px solid #ddd; border-radius: 4px; padding: 5px; max-width: 150px">


## Description
The directory contains the code used to run the experiments and our own models for the paper

## Developer

```
Yanli Wang
Deparment of Computer Science
Email yw7bh@missouri.edu
```

## Contact

```
Jianlin (Jack) Cheng, PhD
William and Nancy Thompson Distinguished Professor
Department of Electrical Engineering and Computer Science
University of Missouri
Columbia, MO 65211, USA
Email: chengji@missouri.edu
```

## Content of Folders

```
Model_Weights: Trained weights of all the models used in the paper
Models: Pytorch implementation of models used in experiments
ProcessData: The Raw Data, Data Loaders, and preprcoessing scripts
Pretrain: Scripts used to run experiments and to analyze experiemnt outputs
Utils: Scripts used for loss function and analyze outputs 
```

## Single cell HiC dataset used in the paper

```
The Cooler file dataset for Human cells with GEO number GSE130711 can be get from https://salkinstitute.app.box.com/s/fp63a4j36m5k255dhje3zcj5kfuzkyj1 or more detailed Human single-cell data at https://salkinstitute.app.box.com/s/fp63a4j36m5k255dhje3zcj5kfuzkyj1/folder/82405563291
The Cooler file format dataset for Drosophila was obtained from GEO with code GSE131811 can be get from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE131811 
```

## Dependencies

ScHiCedsr is written in Python3 and uses the Pytorch module. 
The dependencies can be installed by the following command:

```
# create conda environment
conda env create -f ScHiCedsr.yml

# active the environment
conda active ScHiCedsr
```

## Running ScHiCedsr

If you want to retrain your dataset, You can run ScHiCedsr by the following command:

```
python ScHiCedsr_train.py -g [boolean_value] -e [epoch_number] -b [batch_size] -n [cell_number] -l [cell_line] -p [percentage]
```

Optional Parameters:

```
-g, --gan            Choose the model you want to use, '1' means gan that you will use ScHiCedsrgan model to train, '0' indicates you will use ScHiCedsr to train.
-e, --epoch          How many epoches that you want to train.
-b, --batch_size     The batch size you want to use in you model.
-n, --celln          Cell number in the dataset you want to feed in you model.
-l, --celline        Which cell line you want to choose for your dataset, default is 'Human'.
-p, --percent        The downsampling ratio for the raw dataset, it should be equal or larger than 0.02 but not larger than 1.0, '1.0' means the input data without any downsampling. 
```
## License
This project is covered under the MIT License.

## Reference
Yanli Wang, Zhiye Guo, & Jianlin Cheng. ScHiCEDRN: Single-cell Hi-C data Enhance-ment with Deep Residual and Generative Adversarial Networks. (Submitted).




