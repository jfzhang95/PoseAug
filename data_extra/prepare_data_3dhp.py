import numpy as np

"""
'this file check and convert the pose data'
'-----------------'
0'Right Ankle', 3
1'Right Knee', 2
2'Right Hip', 1
3'Left Hip', 4
4'Left Knee', 5
5'Left Ankle', 6
6'Right Wrist', 15
7'Right Elbow', 14
8'Right Shoulder', 13
9'Left Shoulder', 10
10'Left Elbow', 11
11'Left Wrist', 12
12'Neck', 8 
13'Top of Head', 9
14'Pelvis)', 0
15'Thorax', 7 
16'Spine', mpi3d
17'Jaw', mpi3d
18'Head', mpi3d

mpi3dval: reorder = [14,2,1,0,3,4,5,16,12,18,9,10,11,8,7,6]
"""

# load the download data
mpi3dval_path = './dataset_extras/mpi_inf_3dhp_valid.npz'
mpi_inf_3dhp_valid = np.load(mpi3dval_path)
print(mpi_inf_3dhp_valid.files)

# convert the data to a list to processing.
mpi3d_val_list = []
for i in range(2929):
    tmp_dict = {}
    tmp_dict['filename'] = mpi_inf_3dhp_valid['imgname'][i]
    tmp_dict['kpts2d'] = mpi_inf_3dhp_valid['part'][i]
    tmp_dict['kpts3d'] = mpi_inf_3dhp_valid['S'][i]

    # prepare the image width for 2D keypoint normalization.
    if '/TS5/' in tmp_dict['filename']:
        tmp_dict['width'] = 1920
        tmp_dict['height'] = 1080
    elif '/TS6/' in tmp_dict['filename']:
        tmp_dict['width'] = 1920
        tmp_dict['height'] = 1080
    else:
        tmp_dict['width'] = 2048
        tmp_dict['height'] = 2048

    mpi3d_val_list.append(tmp_dict)


def normalize_screen_coordinates(X,mask, w, h):
    assert X.shape[-1] == 2

    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return (X / w * 2 - [1, h / w] ) * mask

def get_2d_pose_reorderednormed(source):
    reorder = [14,2,1,0,3,4,5,16,12,18,9,10,11,8,7,6]
    tmp_array = source['kpts2d'][reorder][:, :2]
    mask = source['kpts2d'][reorder][:, 2:]
    tmp_array1 = normalize_screen_coordinates(tmp_array, mask, source['width'], source['height'])
    return tmp_array1, mask

def get_3d_pose_reordered(source):
    reorder = [14,2,1,0,3,4,5,16,12,18,9,10,11,8,7,6]
    tmp_array = source['kpts3d'][reorder][:, :3]
    mask = source['kpts3d'][reorder][:, 3:]
    return tmp_array, mask


# convert the pose to 16 joints and put into array
mpi3d_mask_check = []
mpi3d_data_2dpose = []
mpi3d_data_3dpose = []

for source in mpi3d_val_list:
    tmp2d, tmp_2dmask = get_2d_pose_reorderednormed(source)
    tmp3d, tmp_3dmask = get_3d_pose_reordered(source)
    assert np.sum(np.abs(tmp_2dmask - tmp_3dmask)) == 0
    #     mask_check.append(tmp_2dmask)

    if not np.sum(tmp_2dmask) == (len(tmp_2dmask)):
        mpi3d_mask_check.append(tmp_2dmask)

    mpi3d_data_2dpose.append(tmp2d)
    mpi3d_data_3dpose.append(tmp3d)

mpi3d_data_2dpose = np.array(mpi3d_data_2dpose)
mpi3d_data_3dpose = np.array(mpi3d_data_3dpose)

# save the npz for test purpose
print(mpi3d_data_3dpose.shape)
print(mpi3d_data_2dpose.shape)
np.savez('./test_set/test_3dhp.npz',pose3d=mpi3d_data_3dpose,pose2d=mpi3d_data_2dpose)