#!/bin/bash

# Download CAR model
mkdir -p ./models/tlt_pretrained_models/trafficcamnet
cd ./models/tlt_pretrained_models/trafficcamnet
wget https://api.ngc.nvidia.com/v2/models/nvidia/tlt_trafficcamnet/versions/pruned_v1.0/files/trafficnet_int8.txt
wget https://api.ngc.nvidia.com/v2/models/nvidia/tlt_trafficcamnet/versions/pruned_v1.0/files/resnet18_trafficcamnet_pruned.etlt
cd -

# Download LPD model
mkdir -p ./models/LP/LPD
cd ./models/LP/LPD
wget https://api.ngc.nvidia.com/v2/models/nvidia/tlt_lpdnet/versions/pruned_v1.0/files/ccpd_pruned.etlt
wget https://api.ngc.nvidia.com/v2/models/nvidia/tlt_lpdnet/versions/pruned_v1.0/files/ccpd_cal.bin
wget https://api.ngc.nvidia.com/v2/models/nvidia/tlt_lpdnet/versions/pruned_v1.0/files/ccpd_label.txt
cd -

# Download LPR model
mkdir -p ./models/LP/LPR
cd ./models/LP/LPR
wget https://api.ngc.nvidia.com/v2/models/nvidia/tlt_lprnet/versions/deployable_v1.0/files/ch_lprnet_baseline18_deployable.etlt
touch labels_ch.txt
cd -
