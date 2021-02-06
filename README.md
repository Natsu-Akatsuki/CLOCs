# CLOCs: Camera-LiDAR Object Candidates Fusion for 3D Object Detection



## 环境依赖）
依赖：

python3.6

pytorch1.1

ubuntu 16.04/18.04

## 

## 测评指标

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

当前代码基于 SECOND-1.5, please follow the [SECOND-1.5](https://github.com/traveller59/second.pytorch/tree/v1.5) to setup the environment, the dependences for SECOND-1.5 are needed.
```bash
pip install shapely fire pybind11 tensorboardX protobuf scikit-image numba pillow
```
Follow the instructions to install `spconv v1.0` ([commit 8da6f96](https://github.com/traveller59/spconv/tree/8da6f967fb9a054d8870c3515b1b44eca2103634)). Although CLOCs fusion does not need spconv, but SECOND codebase expects it to be correctly configured.



## 准备数据集

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

基于 SECOND-1.5 的教程创建 kitti infos, reduced point cloud and groundtruth-database infos, or 直接下载数据 [here](https://drive.google.com/drive/folders/1ScFUWPwzK5_VXb-LYQZuZVkiBj-dTMJ9?usp=sharing) ，并放置到相应目录



## 决策层数据融合
### Preparation
SECOND as the 3D detector, Cascade-RCNN as the 2D detector. 

1. 2D目标检测器使用`sigmoid scores`, 可以从这里直接输出文件 [here](https://drive.google.com/drive/folders/1ScFUWPwzK5_VXb-LYQZuZVkiBj-dTMJ9?usp=sharing) `cascade_rcnn_sigmoid_data`；或使用这个或自己的二维目标检测器并导出相关的KITTI格式的结果。
2. 下载`second`预训练模型到 ```model_dir```  [here](https://drive.google.com/drive/folders/1ScFUWPwzK5_VXb-LYQZuZVkiBj-dTMJ9?usp=sharing) 
3. 相应的文件夹架构如下：
```plain
└── CLOCs
       ├── d2_detection_data    <-- 2D detection candidates data
       ├── model_dir       <-- SECOND pretrained weights extracted from 'second_model.zip' 
       ├── second 
       ├── torchplus 
       ├── README.md
```

3. 修改配置文件(`CLOCs/second/configs/car.fhd.config`)
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


### 训练

```bash
python ./pytorch/train.py train --config_path=./configs/car.fhd.config --model_dir=/dir/to/your_model_dir
```
模型和相关信息将导出到`/dir/to/your_model_dir`



### 测评

```bash
python ./pytorch/train.py evaluate 
--config_path=./configs/car.fhd.config   \
--model_dir=/dir/to/your/trained_model \
--measure_time=True \
--batch_size=1
```
测评预训练模型`CLOCs_SecCas_pretrained.zip`
```bash
python ./pytorch/train.py evaluate \
--config_path=./configs/car.fhd.config \
--model_dir=/home/helios/pcdet/CLOCs/model_dir/CLOCs_SecCas_pretrained \
--measure_time=True \
--batch_size=1 \
--pickle_result=False
```
若想输出kitti格式的结果, 设置参数 ```pickle_result=False``` 



## Fusion of other 3D and 2D detectors

步骤一：准备2D检测candidates，运行2D检测器，将结果保存为KITTI格式。推荐直接使用预测的值（NMS的分数阈值设为0）

步骤二：准备3D检测candidates，运行3D检测器，将结果保存为SECOND可以读取的格式。

including a matrix with shape of N by 7 that contains the N 3D bounding boxes, and a N-element vector for the 3D confidence scores. 7 parameters correspond to the representation of a 3D bounding box. Be careful with the order and coordinate of the 7 parameters, if the parameters are in LiDAR coordinate, the order should be ```x, y, z, width, length, height, heading```; if the parameters are in camera coordinate, the orderr should be ```x, y, z, lenght, height, width, heading```. The details of the transformation functions can be found in file './second/pytorch/core/box_torch_ops.py'.

步骤三：由于三维目标检测器的detection candidates的个数不同，需要修改CLOCs相关的参数。然后训练融合。比如，以一帧数据为例，second会生成70400个(200×176×2)个detection candidates；对于一维的目标检测器而言，这个值是很大的。对于其他二阶段的目标检测器，可以用二阶段的未进行NMS的detection candidates来进行融合（最终只需要融合几百个）

步骤四：CLOCs的结果是3D detection candidates的修正后的置信度结果。再将这个结果送到后处理中进行融合。

