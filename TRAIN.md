# PoseAug

# Installation 

### Environment
* **Supported OS:** Ubuntu 16.04
* **Packages:**
    * Python: 3.6.9
    * PyTorch: 1.0.1.post2 ([https://pytorch.org/](https://pytorch.org/))
* **build the environment:**   
```sh
cd PoseAug
conda create -n poseaugEnv python=3.6.9
conda activate poseaugEnv
pip install -r requirements.txt
```

### Dataset
* Please follow the `DATASETS.md` to get the data ready.

# Train  
* There are 32 experiments in total (16 for baseline training, 16 for PoseAug training), 
including four pose estimators ([SemGCN](https://github.com/garyzhao/SemGCN), [SimpleBaseline](https://github.com/una-dinosauria/3d-pose-baseline), [ST-GCN](https://github.com/vanoracai/Exploiting-Spatial-temporal-Relationships-for-3D-Pose-Estimation-via-Graph-Convolutional-Networks), [VideoPose](https://github.com/facebookresearch/VideoPose3D))
and four 2D pose settings (Ground Truth, CPN, DET, HR-Net).

### Pretrain
```sh
# gcn
python3 run_baseline.py --note pretrain --dropout 0 --lr 2e-2 --epochs 100 --posenet_name 'gcn' --checkpoint './checkpoint/pretrain_baseline' --keypoints gt
python3 run_baseline.py --note pretrain --dropout 0 --lr 2e-2 --epochs 100 --posenet_name 'gcn' --checkpoint './checkpoint/pretrain_baseline' --keypoints cpn_ft_h36m_dbb
python3 run_baseline.py --note pretrain --dropout 0 --lr 2e-2 --epochs 100 --posenet_name 'gcn' --checkpoint './checkpoint/pretrain_baseline' --keypoints detectron_ft_h36m
python3 run_baseline.py --note pretrain --dropout 0 --lr 2e-2 --epochs 100 --posenet_name 'gcn' --checkpoint './checkpoint/pretrain_baseline' --keypoints hr

# videopose
python3 run_baseline.py --note pretrain --lr 1e-3 --posenet_name 'videopose' --checkpoint './checkpoint/pretrain_baseline' --keypoints gt
python3 run_baseline.py --note pretrain --lr 1e-3 --posenet_name 'videopose' --checkpoint './checkpoint/pretrain_baseline' --keypoints cpn_ft_h36m_dbb
python3 run_baseline.py --note pretrain --lr 1e-3 --posenet_name 'videopose' --checkpoint './checkpoint/pretrain_baseline' --keypoints detectron_ft_h36m
python3 run_baseline.py --note pretrain --lr 1e-3 --posenet_name 'videopose' --checkpoint './checkpoint/pretrain_baseline' --keypoints hr

# mlp
python3 run_baseline.py --note pretrain --lr 1e-3 --stages 2 --posenet_name 'mlp' --checkpoint './checkpoint/pretrain_baseline' --keypoints gt
python3 run_baseline.py --note pretrain --lr 1e-3 --stages 2 --posenet_name 'mlp' --checkpoint './checkpoint/pretrain_baseline' --keypoints cpn_ft_h36m_dbb
python3 run_baseline.py --note pretrain --lr 1e-3 --stages 2 --posenet_name 'mlp' --checkpoint './checkpoint/pretrain_baseline' --keypoints detectron_ft_h36m
python3 run_baseline.py --note pretrain --lr 1e-3 --stages 2 --posenet_name 'mlp' --checkpoint './checkpoint/pretrain_baseline' --keypoints hr

# st-gcn
python3 run_baseline.py --note pretrain --dropout -1 --lr 1e-3 --posenet_name 'stgcn' --checkpoint './checkpoint/pretrain_baseline' --keypoints gt
python3 run_baseline.py --note pretrain --dropout -1 --lr 1e-3 --posenet_name 'stgcn' --checkpoint './checkpoint/pretrain_baseline' --keypoints cpn_ft_h36m_dbb
python3 run_baseline.py --note pretrain --dropout -1 --lr 1e-3 --posenet_name 'stgcn' --checkpoint './checkpoint/pretrain_baseline' --keypoints detectron_ft_h36m
python3 run_baseline.py --note pretrain --dropout -1 --lr 1e-3 --posenet_name 'stgcn' --checkpoint './checkpoint/pretrain_baseline' --keypoints hr
# note: for st-gcn, the baseline requires a different dropout setting (-1: the default dropout setting), while the poseaug requires dropout=0.

```
### PoseAug
```sh
# gcn
python3 run_poseaug.py --note poseaug --dropout 0 --posenet_name 'gcn' --lr_p 1e-3 --checkpoint './checkpoint/poseaug' --keypoints gt
python3 run_poseaug.py --note poseaug --dropout 0 --posenet_name 'gcn' --lr_p 1e-3 --checkpoint './checkpoint/poseaug' --keypoints cpn_ft_h36m_dbb
python3 run_poseaug.py --note poseaug --dropout 0 --posenet_name 'gcn' --lr_p 1e-3 --checkpoint './checkpoint/poseaug' --keypoints detectron_ft_h36m
python3 run_poseaug.py --note poseaug --dropout 0 --posenet_name 'gcn' --lr_p 1e-3 --checkpoint './checkpoint/poseaug' --keypoints hr

# video
python3 run_poseaug.py --note poseaug --posenet_name 'videopose' --lr_p 1e-4 --checkpoint './checkpoint/poseaug' --keypoints gt
python3 run_poseaug.py --note poseaug --posenet_name 'videopose' --lr_p 1e-4 --checkpoint './checkpoint/poseaug' --keypoints cpn_ft_h36m_dbb
python3 run_poseaug.py --note poseaug --posenet_name 'videopose' --lr_p 1e-4 --checkpoint './checkpoint/poseaug' --keypoints detectron_ft_h36m
python3 run_poseaug.py --note poseaug --posenet_name 'videopose' --lr_p 1e-4 --checkpoint './checkpoint/poseaug' --keypoints hr

# mlp
python3 run_poseaug.py --note poseaug --posenet_name 'mlp' --lr_p 1e-4 --stages 2 --checkpoint './checkpoint/poseaug' --keypoints gt
python3 run_poseaug.py --note poseaug --posenet_name 'mlp' --lr_p 1e-4 --stages 2 --checkpoint './checkpoint/poseaug' --keypoints cpn_ft_h36m_dbb
python3 run_poseaug.py --note poseaug --posenet_name 'mlp' --lr_p 1e-4 --stages 2 --checkpoint './checkpoint/poseaug' --keypoints detectron_ft_h36m
python3 run_poseaug.py --note poseaug --posenet_name 'mlp' --lr_p 1e-4 --stages 2 --checkpoint './checkpoint/poseaug' --keypoints hr

# st-gcn
python3 run_poseaug.py --note poseaug --dropout 0 --posenet_name 'stgcn' --lr_p 1e-4 --checkpoint './checkpoint/poseaug' --keypoints gt
python3 run_poseaug.py --note poseaug --dropout 0 --posenet_name 'stgcn' --lr_p 1e-4 --checkpoint './checkpoint/poseaug' --keypoints cpn_ft_h36m_dbb
python3 run_poseaug.py --note poseaug --dropout 0 --posenet_name 'stgcn' --lr_p 1e-4 --checkpoint './checkpoint/poseaug' --keypoints detectron_ft_h36m
python3 run_poseaug.py --note poseaug --dropout 0 --posenet_name 'stgcn' --lr_p 1e-4 --checkpoint './checkpoint/poseaug' --keypoints hr

```

### Comment:
* For simplicity, all the hyper-param are the same. If you want to explore better performance for specific setting, try changing the hyper-param. 
* The GAN training may collapse, change the hyper-param (e.g., random_seed) and re-train the models will solve the problem.

### Monitor the PoseAug training process
```sh
cd ./checkpoint/poseaug
tensorboard --logdir=/path/to/eventfile
```

### Analysis the result
We provide a `checkpoint/PoseAug_result_summary.ipynb` which can generate the result summary table for all 16 experiments.


### Evaluate
```sh
python3 run_evaluate.py --posenet_name 'videopose' --keypoints gt --evaluate '/path/to/checkpoint'
```
