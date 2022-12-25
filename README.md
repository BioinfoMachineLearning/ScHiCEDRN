# Single Cell HiC Data Enhancement
Single cell 3D genome modeling tools developed in the Bioinformatics and Machine Learning Lab

<h1 align="center">ScHiCedsr</h1>

![github_fig](https://user-images.githubusercontent.com/98677544/209075059-a7aca34b-5ee3-4857-a512-5f747da8b451.png)

## Description
The directory contains the code used to run the experiments and our own models for the paper

## Developer:

```
Yanli Wang
Deparment of Computer Science
Email yw7bh@missouri.edu
```

## Contact:

```
Jianlin (Jack) Cheng, PhD
William and Nancy Thompson Distinguished Professor
Department of Electrical Engineering and Computer Science
University of Missouri
Columbia, MO 65211, USA
Email: chengji@missouri.edu
```

## Content of Folders:

```
Model_Weights: Trained weights of all the models used in the paper
Models: Pytorch implementation of models used in experiments
ProcessData: The Raw Data, Data Loaders, and preprcoessing scripts
Pretrain: Scripts used to run experiments and to analyze experiemnt outputs
Utils: Scripts used for loss function and analyze outputs 
```

## Single cell Hic dataset used in the paper

```
The Cooler file dataset for Human cells with GEO number GSE130711 can be get from https://salkinstitute.app.box.com/s/fp63a4j36m5k255dhje3zcj5kfuzkyj1
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



