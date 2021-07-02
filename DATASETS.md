# Dataset setup
# [Human3.6M](http://vision.imar.ro/human3.6m/)
The code for Human3.6M data preparation is borrowed from [VideoPose3D](https://github.com/facebookresearch/VideoPose3D), [SemGCN](https://github.com/garyzhao/SemGCN), [EvoSkeleton](https://github.com/Nicholasli1995/EvoSkeleton).

## Prepare the ground truth 2D 3D data pair for Human3.6
* Setup from original source (recommended)
    * Please follow the instruction from [VideoPose3D](https://github.com/facebookresearch/VideoPose3D/blob/master/DATASETS.md) to process the data from the official [Human3.6M](http://vision.imar.ro/human3.6m/) website.
    * Then generate the 2D and 3D data by `prepare_data_h36m.py`. (Note that `prepare_data_h36m.py` is borrowed from [SemGCN](https://github.com/garyzhao/SemGCN) with 16 joints configuration, which is slightly different from [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) with 17 joints configuration)
    
* Setup from preprocessed dataset
    * Get preprocessed `h36m.zip`: 
      Please follow the instruction from [SemGCN](https://github.com/garyzhao/SemGCN/blob/master/data/README.md) to get the `h36m.zip`.
    * Convert `h36m.zip` to ground-truth 2D 3D npz file: 
      Process `h36m.zip` by `prepare_data_h36m.py` to get `data_3d_h36m.npz` and `data_2d_h36m_gt.npz`
```sh
cd data
python prepare_data_h36m.py --from-archive h36m.zip
cd ..
```
After this step, you should end up with two files in the `data` directory: `data_3d_h36m.npz` for the 3D poses, and `data_2d_h36m_gt.npz` for the ground-truth 2D poses,
which will look like:
   ```
   ${PoseAug}
   ├── data
      ├── data_3d_h36m.npz
      ├── data_2d_h36m_gt.npz
   ```

## Prepare other detected 2D pose for Human3.6M (optional)
In this step, you need to download the detected 2D pose npz file and delete the Neck/Nose axis (e.g., the shape of array: nx17x2 -> nx16x2; n: number_of_frame) for every subject and action.

* To download the Det and CPN 2D pose, please follow the instruction of [VideoPose3D](https://github.com/facebookresearch/VideoPose3D/blob/master/DATASETS.md) and download the `data_2d_h36m_cpn_ft_h36m_dbb.npz` and `data_2d_h36m_detectron_ft_h36m.npz`. 

```sh
cd data
wget https://dl.fbaipublicfiles.com/video-pose-3d/data_2d_h36m_cpn_ft_h36m_dbb.npz
wget https://dl.fbaipublicfiles.com/video-pose-3d/data_2d_h36m_detectron_ft_h36m.npz
cd ..
``` 

* To download the HHR 2D pose, please follow the instruction of [EvoSkeleton](https://github.com/Nicholasli1995/EvoSkeleton/blob/master/docs/HHR.md) and download the `twoDPose_HRN_test.npy` and `twoDPose_HRN_train.npy`, 
and convert them to the same format as `data_2d_h36m_gt.npz`.

* You can also download our [pre-processed joints files](https://drive.google.com/drive/folders/1jVyz9HdT0Jq3-YPZnOQ6GEcOVDRZAifK?usp=sharing). 

Until here, you will have a data folder:
   ```
   ${PoseAug}
   ├── data
      ├── data_3d_h36m.npz
      ├── data_2d_h36m_gt.npz
      ├── data_2d_h36m_detectron_ft_h36m.npz
      ├── data_2d_h36m_cpn_ft_h36m_dbb.npz
      ├── data_2d_h36m_hr.npz
   ```
Please make sure the 2D data are all 16 joints setting.


# [3DHP](http://gvv.mpi-inf.mpg.de/3dhp-dataset/)
The code for 3DHP data preparation is borrowed from [SPIN](https://github.com/nkolot/SPIN)

* Please follow the instruction from [SPIN](https://github.com/nkolot/SPIN/blob/master/fetch_data.sh) to download the preprocessed compression file `dataset_extras.tar.gz` then unzip it to get mpi_inf_3dhp_valid.npz and put it at `data_extra/dataset_extras/mpi_inf_3dhp_valid.npz` (24 joints).
* Then process the `dataset_extras/mpi_inf_3dhp_valid.npz` with `prepare_data_3dhp.py` or `prepare_data_3dhp.ipynb` file to get the `test_3dhp.npz` (16 joints) and place it at `data_extra/test_set`.

Until here, you will have a data_extra folder:
   ```
   ${PoseAug}
   ├── data_extra
      ├── bone_length_npy
         ├── hm36s15678_bl_templates.npy
      ├── dataset_extras
         ├── mpi_inf_3dhp_valid.npz
         ├── ... (not in use)
      ├── test_set
         ├── test_3dhp.npz
      ├── prepare_data_3dhp.ipynb
      ├── prepare_data_3dhp.py
   ```

All the data are set up.
