# NAGCN: Node-aligned Graph Convolutional Network for Whole-slide Image Representation and Classification (CVPR2022) 


## Pre-requisites

* NumPy
* Pandas
* Scikit-learn
* Tqdm
* PyTorch
* PyTorch Geometric

## Data Preparation

First, download [TCGA data](https://www.cancer.gov/tcga). You need to cut whole-slide images (WSIs) into patches (e.g. size $256\times 256$ at $20\times$ magnification) , and then generate the instance-level features using an encoder network. The instance-level features of each WSI are used to generate the WSI bag. Each WSI bag is stored as a `.pt` file.

After that you need to generate an index file (in `.pth` format), which stores a python `list`, and each item of the list is a `dict`, which corresponds to one WSI information.

```python
[{'slide': 'xxx.pt', 'shape': torch.Size([N, D]), 'target': 'y'}]
```

* slide: the file of WSI bag
* shape: the size of WSI bag
* target: the label string of WSI



## Clustering Sampling

Perform hierarchical global-to-local sampling strategy to build visual codebook and generate the sampled WSI bags.

```shell
$ python codebook_generation.py
```



## Graph Construction

Construct WSI graphs using the sampled WSI bags.

```shell
$ python graph_construction.py
```



## Experiment on TCGA Dataset

```shell
$ CUDA_VISIBLE_DEVICES=0 python main_nsclc.py 
```

***





## Citation

If you want to cite this work, please use the following BibTeX entry.

```latex
@InProceedings{Guan_2022_CVPR,
    author    = {Guan, Yonghang and Zhang, Jun and Tian, Kuan and Yang, Sen and Dong, Pei and Xiang, Jinxi and Yang, Wei and Huang, Junzhou and Zhang, Yuyao and Han, Xiao},
    title     = {Node-Aligned Graph Convolutional Network for Whole-Slide Image Representation and Classification},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {18813-18823}
}
```

