# [Asynchronous Feedback Network for Perceptual Point Cloud Quality Assessment (TCSVT2024)](https://arxiv.org/abs/2407.09806)
by Yujie Zhang, Qi Yang, Ziyu Shan, Yiling Xu

This respository is about a no-reference point cloud quality metric based on asynchronous learning. The key idea is **using global feature to directly guide the "generation" of local feature**.

<p align="center">
<img src="https://github.com/zhangyujie-1998/AFQ-Net/blob/main/fig/motivation.png" width='70%' height='70%'>
</p>

## 🎦 Introduction

Recent years have witnessed the success of the deep learning-based technique in research of no-reference point cloud quality assessment (NR-PCQA). For a more accurate quality prediction, many previous studies have attempted to capture global and local features in a bottom-up manner, but ignored the interaction and promotion between them. To solve this problem, we propose a novel asynchronous feedback quality prediction network (AFQ-Net). Motivated by human visual perception mechanisms, AFQ-Net employs a dual-branch structure to deal with global and local features, simulating the left and right hemispheres of the human brain, and constructs a feedback module between them. Specifically, the input point clouds are first fed into a transformer-based global encoder to generate the attention maps that highlight these semantically rich regions, followed by being merged into the global feature. Then, we utilize the generated attention maps to perform dynamic convolution for different semantic regions and obtain the local feature. Finally, a coarse-to-fine strategy is adopted to merge the two features into the final quality score. We conduct comprehensive experiments on three datasets and achieve superior performance over the state-of-the-art approaches on all of these datasets. 

## 🔧 Dependencies

* Python 3.7.16
* PyTorch 1.8.0
* TorchVision
* scipy

## 📦 Data Preparation

We use 2D projections of point clouds as the network input, including texture, depth and occupancy images. To convert point clouds into images, we  create a independent environment  that relies on Pytorch3D and Open3D, following the instruction in [link](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md). Then, we run the MyProjection.py in the  ```projection``` folder to get projections. *We provide the download links for the projections of the used databases including SJTU-PCQA, WPC, LS-PCQA, BASICS and M-PCCD, which can be accessed here  [Onedrive](https://1drv.ms/f/c/669676c02328fc1b/Eonj9bAnDT5NrXkHMzTKuDABIAz6VdX-dHi8JvfTMi_Tiw?e=mHKXZV)*.

If you want to create projections for new databases, you need to replace the path of  'data_dir' in the MyProjection.py with the path of data on your computer. The file structure of used data should be like:

`\````

`├── SJTU-PCQA`

`│  ├── hhi_0.ply`

`│  ├── hhi_1.ply`

`...`

`\````

## 🚆 Training

AFQ-Net uses pre-trained vision transformer (ViT) as the backbone and we provide the pre-trained weight of ViT in  [Onedrive](https://1drv.ms/f/c/669676c02328fc1b/Eonj9bAnDT5NrXkHMzTKuDABIAz6VdX-dHi8JvfTMi_Tiw?e=mHKXZV).  You need to put the pre-trained weights in the ```checkpoint``` folder. Then, you can simply train the AFQ-Net by referring to train.sh. For example, you can train AFQ-Net on the WPC database with the following command:

`CUDA_VISIBLE_DEVICES=0 nohup python -u train.py \`

`--save_flag True \`

`--num_epochs 50 \`

`--batch_size 8 \`

`--test_patch_num 10 \`

`--learning_rate 0.00002 \`

`--decay_rate 5e-4 \`

`--database WPC \`

`--data_dir_texture ./database/WPC/proj_6view_512_texture \`

`--data_dir_depth ./database/WPC/proj_6view_512_depth \`

`--data_dir_mask ./database/WPC/proj_6view_512_mask \`

`--output_dir ./results/WPC/ \`

`--k_fold_num 5 \`

`> logs/log_WPC.txt 2>&1 &`

You only need to replace the path of 'data_dir_texture', 'data_dir_depth' and 'data_dir_mask' with the path of projections on your computer. If you want to use the databases adopted in the paper, you can unzip the provided .zip file in the  ```database``` folder.


## 📖 Citation

If you find this work is helpful, please consider citing:
```
@article{zhang2024asynchronous,
  title={Asynchronous Feedback Network for Perceptual Point Cloud Quality Assessment},
  author={Zhang, Yujie and Yang, Qi and Shan, Ziyu and Xu, Yiling},
  journal={arXiv preprint arXiv:2407.09806},
  year={2024}
}
```



