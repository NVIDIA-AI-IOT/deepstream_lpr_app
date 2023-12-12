#!/bin/bash

# Download CAR model
mkdir -p ./models/tao_pretrained_models/trafficcamnet
cd ./models/tao_pretrained_models/trafficcamnet
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/trafficcamnet/versions/pruned_v1.0/files/trafficnet_int8.txt
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/trafficcamnet/versions/pruned_v1.0/files/resnet18_trafficcamnet_pruned.etlt
cd -

# Download LPD model
mkdir -p ./models/LP/LPD
cd ./models/LP/LPD
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/lpdnet/pruned_v2.2/files?redirect=true&path=LPDNet_usa_pruned_tao5.onnx' -O LPDNet_usa_pruned_tao5.onnx
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/lpdnet/pruned_v2.2/files?redirect=true&path=usa_cal_8.6.1.bin' -O usa_cal_8.6.1.bin
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/lpdnet/pruned_v2.2/files?redirect=true&path=usa_cal_8.5.3.bin' -O usa_cal_8.5.3.bin
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/lpdnet/versions/pruned_v1.0/files/usa_lpd_label.txt
cd -

# Download LPD yolov4-tiny model
mkdir -p ./models/tao_pretrained_models/yolov4-tiny
cd ./models/tao_pretrained_models/yolov4-tiny
wget 'https://api.ngc.nvidia.com/v2/models/nvidia/tao/lpdnet/versions/pruned_v2.1/files/yolov4_tiny_usa_deployable.etlt'
wget 'https://api.ngc.nvidia.com/v2/models/nvidia/tao/lpdnet/versions/pruned_v2.1/files/yolov4_tiny_usa_cal.bin'
wget 'https://api.ngc.nvidia.com/v2/models/nvidia/tao/lpdnet/versions/pruned_v2.1/files/usa_lpd_label.txt'
cd -

# Download LPR model
mkdir -p ./models/LP/LPR
cd ./models/LP/LPR
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/lprnet/versions/deployable_v1.0/files/us_lprnet_baseline18_deployable.etlt
touch labels_us.txt
cd -
