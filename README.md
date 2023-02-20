# CMVAE: Causal Meta VAE For Unsupervised Meta-Learning

This is the Pytorch implementation for the CMVAE

## Dependencies
This code is written in Python. Dependencies include
* python >= 3.6
* pytorch = 1.4 or 1.7
* tqdm, wandb

## Data
* Download Omniglot data from [here](https://drive.google.com/file/d/1aipkJc4JDj91KuiI_VuHj752rdmNXyf_/view?usp=sharing). 
* Download pretrained features for Mini-ImageNet from [here](https://drive.google.com/file/d/1NKYDSHEIQgeTlcrB37ZOZ40N309vcNT8/view?usp=sharing).
* Download pretrained features for CelebA from [here](https://drive.google.com/file/d/1QNbMfAqgdWiP5DzaI8x1a6SZC6IlnDTi/view?usp=sharing).
* (Optional) If you want to train SimCLR from scratch, download images for ImageNet from [here](https://drive.google.com/file/d/1p7Rd59AtM2Faldzv-ju934zPeJuVXqGh/view?usp=sharing) amd CelebA [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

data directory should be look like this:
```shell
data/

├── omiglot/
  ├── train.npy
  ├── val.npy
  └── test.npy
  
├── mimgnet/
  ├── train_features.npy
  ├── val_features.npy
  └── test_features.npy
 
├── celeba/
  ├── train_features.npy
  ├── val_features.npy
  └── test_features.npy
    
└── imgnet or celeba_imgs/ -> (optional) if you want to train SimCLR from scratch
  ├── images/
    ├── n0210891500001298.jpg  
    ├── n0287152500001298.jpg 
	       ...
    └── n0236282200001298.jpg 
  ├── train.csv
  ├── val.csv
  └── test.csv
```

## Experiment
To reproduce **Omniglot 5-way experiment** for CMVAE, run the following code:
```bash
cd omniglot
python main.py --data-dir DATA DIRECTORY   --save-dir SAVE DIRECTORY  --way 5 --sample-size 200
```

To reproduce **Omniglot 20-way experiment** for CMVAE, run the following code:
```bash
cd omniglot
python main.py --data-dir DATA DIRECTORY   --save-dir SAVE DIRECTORY   --way 20 --sample-size 300
```

To reproduce **Mini-ImageNet 5-way experiment** for CMVAE, run the following code:
```bash
cd mimgnet
python main.py --data-dir DATA DIRECTORY   --save-dir SAVE DIRECTORY  
```

To reproduce **CelebA 5-way experiment** for CMVAE, run the following code:
```bash
cd celeba
python main.py --data-dir DATA DIRECTORY   --save-dir SAVE DIRECTORY  
```



(Optional) To reproduce SimCLR features for Mini-ImageNet, run the following code:
```bash
cd simclr
python main.py --data-dir DATA DIRECTORY  --save-dir SAVE DIRECTORY   --feature-save-dir FEATURE SAVE DIRECTORY  
```

## Acknowledgments
Our work and code benefit from two existing works, which we are very grateful.\
[Meta-GMVAE](https://github.com/db-Lee/Meta-GMVAE) \
[notears](https://github.com/xunzheng/notears)