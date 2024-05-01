# Efficient-RGB-T-Early-Fusion-Detection
<div align='left'>
<img src=misc/fig1.png width=63%/>
<img src=misc/fig-strategyComparison.png width=34%/>
</div>

**This is the official repository for our paper "Rethinking Early-Fusion Strategies for Improved Multispectral Object Detection".**

**Our main contributions are summarized as follows:**

- Different from previous works, we summarize three key obstacles limiting the early-fusion strategy, including information interference, domain gap, and representation learning.

- For each obstacle, we propose the corresponding solution: we develop 1) a ShaPE module to address the information interference issue, 2) a weakly supervised learning method to reduce domain gap and improve semantic localization abilities, and 3) a CoreKD to enhance the representation learning of single-branch networks.

- Extensive experiments validate that the early-fusion strategy, equipped with our ShaPE module, weakly supervised learning, and CoreKD technique, shows significant improvement. Additionally, we only retain the ShaPE module during the inference phase. Consequently, our method is efficient and achieves improved performance.

## Environmental Requirements

- Our code is implemented using both [MMDetection](https://github.com/open-mmlab/mmdetection) and [YOLOv5](https://github.com/ultralytics/yolov5). You are encouraged to install these environments using [Anaconda](https://www.anaconda.com/).

- In our MMDetection environment, we use:

		python==3.8.18
		torch==1.12.1+cu111
		torchvision==0.13.1+cu111
		mmcv==2.1.0
		mmdet==3.2.0

- In our YOLOv5 environment, we use:

		python==3.7.16
		torch==1.12.1+cu111
		torchvision==0.13.1+cu111

## Dataset
**M3FD** dataset can be found [here](https://github.com/JinyuanLiu-CV/TarDAL). This dataset doesn't provide a unified data split. In this paper, we provide the **M3FD-zxSplit**. In the `dataset/m3fd-zxSplit` folder, we upload the `train.txt` and `test.txt` files.

**FLIR** dataset can be found at [here](https://github.com/SamVadidar/RGBT?tab=readme-ov-file).

---

**We take the M3FD dataset as an example and describe the dataset generation process.** 

- First, we download the images of the `M3FD_Detection` dataset [here](https://github.com/JinyuanLiu-CV/TarDAL), and unzip `M3FD_Detection.zip` into the `dataset` folder.
		
		├─dataset
		│  │  splitM3FD_zxSceneSplit.py
		│  │  
		│  ├─M3FD
		│  │  └─M3FD_Detection
		│  └─m3fd-zxSplit
		│          test.txt
		│          train.txt
		...

- Then, we obtain the YOLO-format and COCO-format datasets by running the following commands:
	
		cd ./dataset
		# ATTENTION: Please ensure the m3fd dataset has been downloaded and unzipped here!
		# build the YOLO-format dataset
		python ./splitM3FD_zxSceneSplit.py
		# build the COCO-format dataset
		python ./m3fd2coco.py

## Inference

- We upload the necessary files to the `mmdetection` folder, which are required to be merged into the [MMDetection](https://github.com/open-mmlab/mmdetection) repository.

- Model checkpoints can be downloaded from this [cloud link](https://pan.baidu.com/s/1QIrmVyjTZRz3PqXVzGzIeg), extractor code: `ckpt`. Download and unzip the it into the `mmdetection` folder.
		
		...
		│      
		└─mmdetection
    		│  runs.zip
    		│  
			...

---

- ### Detection result using only thermal images

		cd mmdetection
		python tools/test.py ./runs/train/M3FD_thermal_gfl_r50_fpn_1x_bs4/gfl_r50_fpn_1x_m3fd.py ./runs/train/M3FD_thermal_gfl_r50_fpn_1x_bs4/epoch_12.pth --work-dir ./runs/inference

- The result is


		+------------+-------+--------+--------+-------+-------+-------+
		| category   | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
		+------------+-------+--------+--------+-------+-------+-------+
		| People     | 0.35  | 0.646  | 0.342  | 0.15  | 0.553 | 0.723 |
		| Car        | 0.477 | 0.739  | 0.501  | 0.127 | 0.48  | 0.797 |
		| Bus        | 0.36  | 0.542  | 0.383  | 0.0   | 0.122 | 0.544 |
		| Motorcycle | 0.212 | 0.365  | 0.219  | 0.0   | 0.193 | 0.622 |
		| Lamp       | 0.052 | 0.152  | 0.03   | 0.024 | 0.25  | nan   |
		| Truck      | 0.336 | 0.477  | 0.38   | 0.001 | 0.316 | 0.625 |
		+------------+-------+--------+--------+-------+-------+-------+
		05/01 23:07:32 - mmengine - INFO - bbox_mAP_copypaste: 0.298 0.487 0.309 0.050 0.319 0.662

---

- ### Detection result using plain RGB-T early-fusion strategy

		python tools/test.py ./runs/train/M3FD_rgbtEarly_gfl_r50_fpn_1x_bs4/gfl_r50_fpn_1x_m3fd.py ./runs/train/M3FD_rgbtEarly_gfl_r50_fpn_1x_bs4/epoch_12.pth --work-dir ./runs/inference

- The result is

		+------------+-------+--------+--------+-------+-------+-------+
		| category   | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
		+------------+-------+--------+--------+-------+-------+-------+
		| People     | 0.358 | 0.64   | 0.36   | 0.147 | 0.572 | 0.755 |
		| Car        | 0.538 | 0.784  | 0.586  | 0.197 | 0.559 | 0.824 |
		| Bus        | 0.379 | 0.545  | 0.409  | 0.089 | 0.159 | 0.56  |
		| Motorcycle | 0.248 | 0.395  | 0.279  | 0.002 | 0.293 | 0.586 |
		| Lamp       | 0.136 | 0.297  | 0.101  | 0.08  | 0.479 | nan   |
		| Truck      | 0.373 | 0.523  | 0.415  | 0.005 | 0.339 | 0.679 |
		+------------+-------+--------+--------+-------+-------+-------+
		05/01 23:15:39 - mmengine - INFO - bbox_mAP_copypaste: 0.339 0.531 0.358 0.087 0.400 0.681

---

- ### Detection result using ShaPE module
	
		python tools/test.py ./runs/train/M3FD_rgbtEarly_zxModifiedStem_gfl_r50_fpn_1x_bs4/gfl_r50_fpn_1x_m3fd.py ./runs/train/M3FD_rgbtEarly_zxModifiedStem_gfl_r50_fpn_1x_bs4/epoch_12.pth --work-dir ./runs/inference

- The result is

		+------------+-------+--------+--------+-------+-------+-------+
		| category   | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
		+------------+-------+--------+--------+-------+-------+-------+
		| People     | 0.371 | 0.658  | 0.364  | 0.155 | 0.586 | 0.757 |
		| Car        | 0.547 | 0.791  | 0.59   | 0.197 | 0.572 | 0.836 |
		| Bus        | 0.454 | 0.63   | 0.486  | 0.137 | 0.204 | 0.65  |
		| Motorcycle | 0.227 | 0.419  | 0.231  | 0.001 | 0.241 | 0.61  |
		| Lamp       | 0.14  | 0.302  | 0.113  | 0.086 | 0.48  | nan   |
		| Truck      | 0.385 | 0.542  | 0.425  | 0.003 | 0.364 | 0.691 |
		+------------+-------+--------+--------+-------+-------+-------+
		05/01 23:20:58 - mmengine - INFO - bbox_mAP_copypaste: 0.354 0.557 0.368 0.096 0.408 0.709

---

- ### Detection result using ShaPE + Weakly Supervised Learning

		python tools/test.py ./runs/train/M3FD_rgbtEarly_zxModifiedStem_gfl_r50_fpn_1x_bs4_clipTransfer/gfl_r50_fpn_1x_m3fd.py ./runs/train/M3FD_rgbtEarly_zxModifiedStem_gfl_r50_fpn_1x_bs4_clipTransfer/epoch_12.pth --work-dir ./runs/inference

- The result is

		+------------+-------+--------+--------+-------+-------+-------+
		| category   | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
		+------------+-------+--------+--------+-------+-------+-------+
		| People     | 0.366 | 0.651  | 0.362  | 0.157 | 0.575 | 0.766 |
		| Car        | 0.555 | 0.798  | 0.601  | 0.205 | 0.58  | 0.843 |
		| Bus        | 0.468 | 0.66   | 0.511  | 0.077 | 0.239 | 0.651 |
		| Motorcycle | 0.238 | 0.418  | 0.241  | 0.0   | 0.249 | 0.625 |
		| Lamp       | 0.139 | 0.309  | 0.112  | 0.081 | 0.497 | nan   |
		| Truck      | 0.389 | 0.54   | 0.424  | 0.004 | 0.368 | 0.693 |
		+------------+-------+--------+--------+-------+-------+-------+
		05/01 23:27:45 - mmengine - INFO - bbox_mAP_copypaste: 0.359 0.563 0.375 0.087 0.418 0.715

---

- ### Detection result using our EME method
	
		python tools/test.py ./runs/train/M3FD_coreKD_rgbtEarly_zxModifiedStem_gfl_r101tor50_fpn_1x_bs4_clipTransfer/gfl_r50_fpn_1x_m3fd_kdMed2Ear.py ./runs/train/M3FD_coreKD_rgbtEarly_zxModifiedStem_gfl_r101tor50_fpn_1x_bs4_clipTransfer/epoch_10.pth --work-dir ./runs/inference

- The result is

		+------------+-------+--------+--------+-------+-------+-------+
		| category   | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
		+------------+-------+--------+--------+-------+-------+-------+
		| People     | 0.387 | 0.685  | 0.383  | 0.18  | 0.591 | 0.776 |
		| Car        | 0.558 | 0.813  | 0.604  | 0.216 | 0.581 | 0.85  |
		| Bus        | 0.465 | 0.633  | 0.511  | 0.08  | 0.2   | 0.694 |
		| Motorcycle | 0.278 | 0.427  | 0.327  | 0.008 | 0.353 | 0.629 |
		| Lamp       | 0.16  | 0.359  | 0.122  | 0.103 | 0.511 | nan   |
		| Truck      | 0.377 | 0.539  | 0.432  | 0.01  | 0.344 | 0.701 |
		+------------+-------+--------+--------+-------+-------+-------+
		05/01 23:32:26 - mmengine - INFO - bbox_mAP_copypaste: 0.371 0.576 0.396 0.099 0.430 0.730

---

- ### Detection result using ShaPE but with EME Checkpoint

	This setting demonstrates that only the ShaPE module is retained during the inference phase, while weakly supervised learning and coreKD are removed.

		python tools/test.py ./runs/train/M3FD_rgbtEarly_zxModifiedStem_gfl_r50_fpn_1x_bs4/gfl_r50_fpn_1x_m3fd.py ./runs/train/M3FD_coreKD_rgbtEarly_zxModifiedStem_gfl_r101tor50_fpn_1x_bs4_clipTransfer/epoch_10.pth --work-dir ./runs/inference

- The result is 
  
		+------------+-------+--------+--------+-------+-------+-------+
		| category   | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
		+------------+-------+--------+--------+-------+-------+-------+
		| People     | 0.387 | 0.685  | 0.383  | 0.18  | 0.591 | 0.776 |
		| Car        | 0.558 | 0.813  | 0.604  | 0.216 | 0.581 | 0.85  |
		| Bus        | 0.465 | 0.633  | 0.511  | 0.08  | 0.2   | 0.694 |
		| Motorcycle | 0.278 | 0.427  | 0.327  | 0.008 | 0.353 | 0.629 |
		| Lamp       | 0.16  | 0.359  | 0.122  | 0.103 | 0.511 | nan   |
		| Truck      | 0.377 | 0.539  | 0.432  | 0.01  | 0.344 | 0.701 |
		+------------+-------+--------+--------+-------+-------+-------+
		05/01 23:55:59 - mmengine - INFO - bbox_mAP_copypaste: 0.371 0.576 0.396 0.099 0.430 0.730