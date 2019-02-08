# Robot Chef - Computer Vision


### Contents

3. [Usage](#usage)

### Installation

1. From `src` do `cd computer_vision/`
2. `mkdir data`
3. `git clone https://github.com/epic-kitchens/annotations.git`
4. `git clone https://github.com/epic-kitchens/download-scripts.git`
5. `cd download-scripts`
6. `sh download_object_detection_images.sh ../data`
7. `cd ../`
8. You will need to unzip the files downloaded. Use this command for each file you want to unzip:
    - `cd` to one of the tar files in the data dir.
    - `mkdir [tar name]`
    - `tar -C ./[tar name] -xvf [tar name].tar` 
9. `git clone https://github.com/matterport/Mask_RCNN.git`
10. `cd Mask_RCNN`
<!-- 11. `wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5` -->
13. `pip3 install -r requirements.txt`
14. `python3 setup.py install`
15. 

18. `python3 ./DriverRCNN.py train --dataset='./data/EPIC_KITCHENS_2018/object_detection_images' --weights=coco`

### Usage
#### Client Execution
Inside the `computer_vision` directory execute:
``

#### Server Execution
These instructions are made with the UNCC server in mind. It also is intended that you use Linux, or Cmder
1. Log into the server via: `ssh [username]@hpc.uncc.edu` using your username that you were approved for.
    1.   You will have to input your password and possibly a 2 step verification. Note, if you are a student 
    using your student id, you will be inputting your current student login password.
2. Typing `ls` should show the current directory that you are allowed to work in.
    1. If you are using the UNCC server, you should see `master_data` and `workspace`
4. Be sure to execute:
    1. `mkdir weights`
    2. `wget https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5`
3. Jobs can be executed via: `qsub face.sh`
    1. If you get the error: `qsub:  script is written in DOS/Windows text format` you can resolve this using 
    `dos2unix face.sh`
4. `qstat` shows all current jobs while `qstat -u [username]` shows the job for the current user
5. If you `cd /[projectname]/log` then use `nano [log file name]` you can view the log output of the server.
6. If you would like to code the project locally, and if you are using linux Ubuntu 16.04, then you can mount the file 
system on the server via: `sshfs [username]@hpc.uncc.edu: [local_folder_to_mount_to] -o nonempty`
7. If you want to add more videos you can do so without a file transfer application by using: `scp -v ./[path to the video you want to send, you can also use * to send all videos or files in the directory] 
[username]@hpc.uncc.edu:/users/[username]/[path to the folder you want to store the videos]`
8. If you need to delete a job, then execute `qdel [JOBID]`
9. If you load a python3 module, you can load [Jupyter](https://techtalktone.wordpress.com/2017/03/28/running-jupyter-notebooks-on-a-remote-server-via-ssh/)


### Note:
* If you are installing tensorflow for python3.7 use pip3 install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.10.1-py3-none-any.whl
* If you want to visualize tensorflow models you can call `tensorboard --logdir=./run_logs/[a specific folder if you want, otherwise leave blank] --host=127.0.0.1`

