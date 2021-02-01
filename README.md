## CLOCs: Camera-LiDAR Object Candidates Fusion for 3D Object Detection

简介CLOCs

## Environment（环境依赖）
依赖：

python3.6

pytorch1.1

ubuntu 16.04/18.04

## Performance on KITTI validation set (3712 training, 3769 validation)
### CLOCs_SecCas (SECOND+Cascade-RCNN) VS SECOND:
```
new 40 recall points
Car:      Easy@0.7       Moderate@0.7   Hard@0.7
bev:  AP: 96.51 / 95.61, 92.37 / 89.54, 89.41 / 86.96
3d:   AP: 92.74 / 90.97, 82.90 / 79.94, 77.75 / 77.09
```
```
old 11 recall points
Car:      Easy@0.7       Moderate@0.7   Hard@0.7
bev:  AP: 90.52 / 90.36, 89.29 / 88.10, 87.84 / 86.80
3d:   AP: 89.49 / 88.31, 79.31 / 77.99, 77.36 / 76.52
```
## Install（安装包依赖）

The code is developed based on SECOND-1.5, please follow the [SECOND-1.5](https://github.com/traveller59/second.pytorch/tree/v1.5) to setup the environment, the dependences for SECOND-1.5 are needed.
```bash
pip install shapely fire pybind11 tensorboardX protobuf scikit-image numba pillow
```
Follow the instructions to install `spconv v1.0` ([commit 8da6f96](https://github.com/traveller59/spconv/tree/8da6f967fb9a054d8870c3515b1b44eca2103634)). Although CLOCs fusion does not need spconv, but SECOND codebase expects it to be correctly configured.

## Prepare dataset (KITTI)
Download KITTI dataset and organize the files as follows:

```plain
└── KITTI_DATASET_ROOT
       ├── training    <-- 7481 train data
       |   ├── image_2 <-- for visualization
       |   ├── calib
       |   ├── label_2
       |   ├── velodyne
       |   └── velodyne_reduced <-- empty directory
       └── testing     <-- 7580 test data
       |   ├── image_2 <-- for visualization
       |   ├── calib
       |   ├── velodyne
       |   └── velodyne_reduced <-- empty directory
       └── kitti_dbinfos_train.pkl
       ├── kitti_infos_train.pkl
       ├── kitti_infos_test.pkl
       ├── kitti_infos_val.pkl
       └── kitti_infos_trainval.pkl
```

Next, you could follow the SECOND-1.5 instructions to create kitti infos, reduced point cloud and groundtruth-database infos, or just download these files from [here](https://drive.google.com/drive/folders/1ScFUWPwzK5_VXb-LYQZuZVkiBj-dTMJ9?usp=sharing) and put them in the correct directories as shown above.

## Fusion of SECOND and Cascade-RCNN
### Preparation
CLOCs operates on the combined output of a 3D detector and a 2D detector. For this example, we use SECOND as the 3D detector, Cascade-RCNN as the 2D detector. 

1. For this example, we use detections with sigmoid scores, you could download the Cascade-RCNN detections for the KITTI train and validations set from [here](https://drive.google.com/drive/folders/1ScFUWPwzK5_VXb-LYQZuZVkiBj-dTMJ9?usp=sharing) file name:'cascade_rcnn_sigmoid_data', or you could run the 2D detector by your self and save the results for the fusion. You could also use your own 2D detector to generate these 2D detections and save them in KITTI format for fusion. 

2. Then download the pretrained SECOND models from [here](https://drive.google.com/drive/folders/1ScFUWPwzK5_VXb-LYQZuZVkiBj-dTMJ9?usp=sharing) file name: 'second_model.zip', create an empty directory named ```model_dir``` under your CLOCs root directory and unzip the files to ```model_dir```. Your CLOCs directory should look like this:
```plain
└── CLOCs
       ├── d2_detection_data    <-- 2D detection candidates data
       ├── model_dir       <-- SECOND pretrained weights extracted from 'second_model.zip' 
       ├── second 
       ├── torchplus 
       ├── README.md
```

3. Then modify the config file carefully:
```bash
train_input_reader: {
  ...
  database_sampler {
    database_info_path: "/dir/to/your/kitti_dbinfos_train.pkl"
    ...
  }
  kitti_info_path: "/dir/to/your/kitti_infos_train.pkl"
  kitti_root_path: "/dir/to/your/KITTI_DATASET_ROOT"
}
...
train_config: {
  ...
  detection_2d_path: "/dir/to/2d_detection/data"
}
...
eval_input_reader: {
  ...
  kitti_info_path: "/dir/to/your/kitti_infos_val.pkl"
  kitti_root_path: "/dir/to/your/KITTI_DATASET_ROOT"
}

```
### Train
```bash
python ./pytorch/train.py train --config_path=./configs/car.fhd.config --model_dir=/dir/to/your_model_dir
```
The trained models and related information will be saved in '/dir/to/your_model_dir'
### Evaluation
```bash
python ./pytorch/train.py evaluate --config_path=./configs/car.fhd.config --model_dir=/dir/to/your/trained_model --measure_time=True --batch_size=1
```
For example if you want to test the pretrained model downloaded from [here](https://drive.google.com/drive/folders/1ScFUWPwzK5_VXb-LYQZuZVkiBj-dTMJ9?usp=sharing) file name: 'CLOCs_SecCas_pretrained.zip', unzip it, then you could run:
```bash
python ./pytorch/train.py evaluate --config_path=./configs/car.fhd.config --model_dir=/dir/to/your/CLOCs_SecCas_pretrained --measure_time=True --batch_size=1
```
If you want to export KITTI format label files, add ```pickle_result=False``` at the end of the above commamd.



## Fusion of other 3D and 2D detectors

步骤一：准备2D检测candidates，运行2D检测器，将结果保存为KITTI格式。推荐直接使用预测的值（NMS的分数阈值设为0）

步骤二：准备3D检测candidates，运行3D检测器，将结果保存为SECOND可以读取的格式。

including a matrix with shape of N by 7 that contains the N 3D bounding boxes, and a N-element vector for the 3D confidence scores. 7 parameters correspond to the representation of a 3D bounding box. Be careful with the order and coordinate of the 7 parameters, if the parameters are in LiDAR coordinate, the order should be ```x, y, z, width, length, height, heading```; if the parameters are in camera coordinate, the orderr should be ```x, y, z, lenght, height, width, heading```. The details of the transformation functions can be found in file './second/pytorch/core/box_torch_ops.py'.

步骤三：由于三维目标检测器的detection candidates的个数不同，需要修改CLOCs相关的参数。然后训练融合。比如，以一帧数据为例，second会生成70400个(200×176×2)个detection candidates；对于一维的目标检测器而言，这个值是很大的。对于其他二阶段的目标检测器，可以用二阶段的未进行NMS的detection candidates来进行融合（最终只需要融合几百个）

步骤四：CLOCs的结果是3D detection candidates的修正后的置信度结果。再将这个结果送到后处理中进行融合。

