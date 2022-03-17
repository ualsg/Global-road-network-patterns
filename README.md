# Global Urban Road Network Patterns

This repository is the official implementation of [Global Urban Road Network Patterns](#ADD#). It includes the codes of the custom methods (written in Python) involved in the paper. We provide two major modules, `CRHD generator v2` and `Morphoindex generator v2`, which are the upgraded verisons of those proposed in [Classification of Urban Morphology with Deep Learning: Application on Urban Vitality](https://arxiv.org/abs/2105.09908). Functions of previous version are all kept and the tutorials could be found at [Road-Network-Classification](https://github.com/ualsg/Road-Network-Classification). 

Apart from leveraging online data from OpenStreetMap (OSM), `CRHD generator v2` enables using local road network data to generate CRHDs, which is useful for regions with unfavorable quality of OSM street data. `Morphoindex generator v2` adds the function of predicting the probabilities of road network patterns at multiple scales (0.5km, 1km, 2km). The CRHD operations mentioned in the paper are embedded in the module, so only the CRHDs at mid scale (1km resolusion) should be prepared in advance. 

You could find the list of cities of interest and the outcome data of their multiscale road network patterns at [figshare](https://doi.org/10.6084/m9.figshare.19375103.v2).

## Requirements

To use `CRHD generator v2`, you need to install the requirements below. We would recommend create a clean new virtual environment with Python 3.7 to install `osmnx` and the other packages.

```setup
pip install osmnx==1.1.2
pip install geopandas==0.10.2
pip install matplotlib==3.5.1
pip install pyproj==3.2.1
```
To use `Morphoindex generator v2`, you need to install the additional requirements:

```setup
pip install tensorflow==2.7.0
pip install opencv-python==3.4.2
pip install numpy==1.21.4
```
To calculate pattern probabilities, you should also download `config.py`, `MODEL.py` and `Build_model.py` togehther with `morphoindex_generator_v2.py`, and put them in the same filepath. Also, make sure you have downloaded our pretrained model which you can find below.

## Pre-trained Model

You can download our pretrained models here:

- [Model_v1_4class](https://drive.google.com/file/d/1N7T9lN4TL5r8EqduZfWv22ROZO4zp_FN/view?usp=sharing) trained with ResNet-34 architecture, learning rate as 0.0005, and batch size as 2. This model supports four-class classification of road network patterns.
- [Model_v2_6class](https://drive.google.com/file/d/1J7_LlgmuXiJkAD3uuEgmf5x96x1SJZSO/view?usp=sharing) (used in this paper) trained with ResNet-34 architecture, learning rate as 0.00005, batch size as 16, and data augmentation on. Details of the training hyperparameters are available in the training codes in 'train' file. This model supports six-class classification of road network patterns.


## Paper

A [preprint](#ADD#) about the work is available.

If you use this work in a scientific context, please cite this article.

xxxx

```
xxx
```

## Contact

[Chen Wangyang](https://ual.sg/authors/wangyang/), [Urban Analytics Lab](https://ual.sg), National University of Singapore, Singapore

