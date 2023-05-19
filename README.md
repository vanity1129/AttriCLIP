# AttriCLIP

This is the pytorch version code of "AttriCLIP: A Non-Incremental Learner for Incremental Knowledge Learning" in CVPR2023.

## Content

- [AttriCLIP](#attriclip)
  - [Content](#content)
  - [Introduce of AttriCLIP](#introduce-of-attriclip)
  - [Datasets](#datasets)
  - [Pretrained CLIP](#pretrained-clip)
  - [Insturction](#insturction)

## [Introduce of AttriCLIP](#Content)

AttriCLIP is introduced from《AttriCLIP: A Non-Incremental Learner for Incremental Knowledge Learning》

Paper ：Runqi Wang, Xiaoyue Duan, Guoliang Kang, Jianzhuang Liu, Shaohui Lin, Songcen Xu, Jinhu Lv, Baochang Zhang. "Few-Shot Learning with Visual Distribution Calibration and Cross-Modal Distribution Alignment". In CVPR, 2023.

## [Datasets](#Content)

The test Datasets：CIFAR100, [Download link](https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz)  
Datasets size：100 classes and 32*32 pixels for each image. 

The test Datasets：a subset of ImageNet, [Download link](https://www.image-net.org/)  
Datasets size：100 classes. The chosen classes are shown in supplementary materials of the paper and dataset/imagenet100.py of this project.


## [Pretrained CLIP](#Content)

We use the pretrained CLIP model from [here](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt)

## [Insturction](#Content)

```python
pip install -r requirements.txt
python main_incremental_sumbit.py --root data_path
```