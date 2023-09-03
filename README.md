![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg)
# EDiffusion: Edge-Federated Person Re-Identification by Stable Diffusion
<div align="center">
<img src=./illustrate.svg width=85% />
</div>
This repository is the PyTorch source code implementation of 
[EDiffusion: Edge-Federated Person Re-Identification by Stable Diffusion]() and is currently being reviewed at JSAC. In the following is an instruction to use the code
to train and evaluate the text-based ReID model on the [RSTPReid](
https://github.com/NjtechCVLab/RSTPReid-Dataset) dataset. 

[//]: # (<img src="https://github.com/honestws/TextEdgeReID/blob/master/illustrate.svg"/><br/>  )

[//]: # (### Requirements)

[//]: # ()
[//]: # (Code was tested in virtual environment with Python 3.8 and 1 * RTX 3090 24G. )

[//]: # (The full installed packages in our virtual enviroment  were presented in the 'requirements.txt' file. )

[//]: # ()
[//]: # (### Data preparation)

[//]: # (Download [Market1501 Dataset]&#40;https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html&#41; [[Google]]&#40;https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view&#41; [[Baidu]]&#40;https://pan.baidu.com/s/1ntIi2Op&#41;)

[//]: # ()
[//]: # (Preparation: Open and edit the script `prepare_market.py` in the editor. Change the fifth line in `prepare_market.py` to your download path. Run the following script in the terminal to put the images with the same id in one folder:)

[//]: # (```bash)

[//]: # (python prepare_market.py)

[//]: # (```)

[//]: # ()
[//]: # (Remark: We will remove cross-camera annotations based on 'def get_camera_person_info' in `builder.py`.)

[//]: # ()
[//]: # (We use 'tree' command to show the prejoct's directory listing)

[//]: # (in a neater format for different subdirectories, files and folders in our experiment as follows:)

[//]: # (```)

[//]: # (.)

[//]: # (├── argpaser.py)

[//]: # (├── builder.py)

[//]: # (├── continual_list.py)

[//]: # (├── dreamer.py)

[//]: # (├── DukeMTMC-ReID)

[//]: # (│   ├── bounding_box_test)

[//]: # (│   ├── bounding_box_train)

[//]: # (│   ├── CITATION.txt)

[//]: # (│   ├── LICENSE_DukeMTMC-reID.txt)

[//]: # (│   ├── LICENSE_DukeMTMC.txt)

[//]: # (│   ├── pytorch)

[//]: # (│   ├── query)

[//]: # (│   └── README.md)

[//]: # (├── evaluator.py)

[//]: # (├── final_images)

[//]: # (│   └── output_04456.png)

[//]: # (├── log)

[//]: # (│   └── events.out.tfevents.1667903357.server)

[//]: # (├── lossfun.py)

[//]: # (├── __MACOSX)

[//]: # (│   └── bounding_box_train)

[//]: # (├── main.py)

[//]: # (├── Market-1501)

[//]: # (│   ├── bounding_box_test)

[//]: # (│   ├── bounding_box_train)

[//]: # (│   ├── gt_bbox)

[//]: # (│   ├── gt_query)

[//]: # (│   ├── pytorch)

[//]: # (│   ├── query)

[//]: # (│   └── readme.txt)

[//]: # (├── MARS)

[//]: # (│   ├── bbox_test)

[//]: # (│   ├── bbox_test.zip)

[//]: # (│   ├── bbox_train)

[//]: # (│   ├── bbox_train.zip)

[//]: # (│   └── pytorch)

[//]: # (├── model.py)

[//]: # (├── MSMT17)

[//]: # (│   ├── bounding_box_test)

[//]: # (│   ├── bounding_box_train)

[//]: # (│   ├── __MACOSX)

[//]: # (│   ├── pytorch)

[//]: # (│   ├── query)

[//]: # (│   └── test)

[//]: # (├── MSMT17.zip)

[//]: # (├── net)

[//]: # (│   ├── requirements.txt)

[//]: # (│   ├── result.pth)

[//]: # (│   └── teacher.pth)

[//]: # (├── OPP-PesonReID.zip)

[//]: # (├── prepare_dukemtmc.py)

[//]: # (├── prepare_market.py)

[//]: # (├── prepare_mars.py)

[//]: # (├── prepare_msmt.py)

[//]: # (├── __pycache__)

[//]: # (│   ├── argpaser.cpython-38.pyc)

[//]: # (│   ├── builder.cpython-38.pyc)

[//]: # (│   ├── dreamer.cpython-38.pyc)

[//]: # (│   ├── evaluator.cpython-38.pyc)

[//]: # (│   ├── lossfun.cpython-38.pyc)

[//]: # (│   ├── model.cpython-38.pyc)

[//]: # (│   ├── trainer.cpython-38.pyc)

[//]: # (│   └── util.cpython-38.pyc)

[//]: # (├── README.md)

[//]: # (├── requirements.txt)

[//]: # (├── teacher.pth)

[//]: # (├── trainer.py)

[//]: # (├── util.py)

[//]: # (└── wget-log)

[//]: # (```)

[//]: # (Futhermore, you also can test our code on [DukeMTMC-reID Dataset]&#40;[GoogleDriver]&#40;https://drive.google.com/open?id=1jjE85dRCMOgRtvJ5RQV9-Afs-2_5dY3O&#41; or &#40;[BaiduYun]&#40;https://pan.baidu.com/s/1jS0XM7Var5nQGcbf9xUztw&#41; password: bhbh&#41;&#41;.)

[//]: # (### Model preparation)

[//]: # (Please find the pretrained teacher Re-ID model in)

[//]: # ([BaiduPan]&#40;https://pan.baidu.com/s/15h4UAkAMghtVCZUcz24OFw&#41; &#40;password: bwsa&#41;.)

[//]: # (After downloading *teacher.pth*, please put it into *./net/* folder.)

[//]: # ()
[//]: # ()
[//]: # (### Run the code)

[//]: # ()
[//]: # (Please enter the main folder, Train the OPP model by)

[//]: # (```bash)

[//]: # (python main.py --dream_person 1 --ms 5000 --T 2.0 --lamb 0.05 --sigma 1.0 --batch_size 32  --data_dir your_project_path/OPP-PersonReID/Market-1501/pytorch/)

[//]: # (```)

[//]: # (`--dream_person` num of person for dreaming.)

[//]: # ()
[//]: # (`--ms` memory size of dreamer.)

[//]: # ()
[//]: # (`--T` temperature for target generation)

[//]: # ()
[//]: # (`--lamb` coefficient for the mix loss function)

[//]: # ()
[//]: # (`--sigma` parameter of Gaussian Kernel)

[//]: # ()
[//]: # (`--batch_size` training batch size.)

[//]: # ()
[//]: # (`--data_dir` the path of the training data.)

[//]: # ()
[//]: # (### Monitoring training progress)

[//]: # (```)

[//]: # (tensorboard.sh --port 6006 --logdir your_project_path/log)
```


