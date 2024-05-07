#!/bin/bash

export PATH=$PATH:/usr/src/tensorrt/bin
if [ -f "tao-converter" ]; then
    echo "tao-converter exist"
else
    echo " download tao-converter from https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/resources/tao-converter/version"
    IS_JETSON_PLATFORM=`uname -i | grep aarch64`
    if [ ! ${IS_JETSON_PLATFORM} ]; then
        wget --content-disposition 'https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/tao/tao-converter/v5.1.0_8.6.3.1_x86/files?redirect=true&path=tao-converter' -O tao-converter
    else
        wget --content-disposition 'https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/tao/tao-converter/v5.1.0_jp6.0_aarch64/files?redirect=true&path=tao-converter' -O tao-converter
    fi
fi

chmod 755 tao-converter

echo "need to download_us.sh first"

echo "prepare trafficcamnet"
mkdir -p triton_models/us/trafficcamnet/1/
./tao-converter -k tlt_encode -t int8 -c models/tao_pretrained_models/trafficcamnet/trafficcamnet_int8.txt -e triton_models/us/trafficcamnet/1/resnet18_trafficcamnet_pruned.etlt_b1_gpu0_int8.engine -b 1 -d 3,544,960 models/tao_pretrained_models/trafficcamnet/resnet18_trafficcamnet_pruned.etlt

echo "prepare US_LPD"
mkdir -p triton_models/us/US_LPD/1/
trtexec --onnx=models/LP/LPD/LPDNet_usa_pruned_tao5.onnx --int8 --calib=models/LP/LPD/usa_cal_8.6.1.bin \
 --saveEngine=triton_models/us/US_LPD/1/LPDNet_usa_pruned_tao5.onnx_b16_gpu0_int8.engine --minShapes="input_1:0":1x3x480x640 \
 --optShapes="input_1:0":16x3x480x640 --maxShapes="input_1:0":16x3x480x640
cp models/LP/LPD/usa_lpd_label.txt triton_models/us/US_LPD/

echo "prepare us yolov4-tiny"
mkdir -p triton_models/us/us_lpd_yolov4-tiny/1/
./tao-converter -k nvidia_tlt -t int8 -c models/tao_pretrained_models/yolov4-tiny/yolov4_tiny_usa_cal.bin -e triton_models/us/us_lpd_yolov4-tiny/1/yolov4_tiny_usa_deployable.etlt_b16_gpu0_int8.engine -b 16 -p Input,1x3x480x640,8x3x480x640,16x3x480x640 \
--layerPrecisions cls/mul:fp32,box/mul_6:fp32,box/add:fp32,box/mul_4:fp32,box/add_1:fp32,cls/Reshape_reshape:fp32,box/Reshape_reshape:fp32,encoded_detections:fp32,bg_leaky_conv1024_lrelu:fp32,sm_bbox_processor/concat_concat:fp32,sm_bbox_processor/sub:fp32,sm_bbox_processor/Exp:fp32,yolo_conv1_4_lrelu:fp32,yolo_conv1_3_1_lrelu:fp32,md_leaky_conv512_lrelu:fp32,sm_bbox_processor/Reshape_reshape:fp32,conv_sm_object:fp32,yolo_conv5_1_lrelu:fp32,concatenate_6:fp32,yolo_conv3_1_lrelu:fp32,concatenate_5:fp32,yolo_neck_1_lrelu:fp32 \
--layerOutputTypes cls/mul:fp32,box/mul_6:fp32,box/add:fp32,box/mul_4:fp32,box/add_1:fp32,cls/Reshape_reshape:fp32,box/Reshape_reshape:fp32,encoded_detections:fp32,bg_leaky_conv1024_lrelu:fp32,sm_bbox_processor/concat_concat:fp32,sm_bbox_processor/sub:fp32,sm_bbox_processor/Exp:fp32,yolo_conv1_4_lrelu:fp32,yolo_conv1_3_1_lrelu:fp32,md_leaky_conv512_lrelu:fp32,sm_bbox_processor/Reshape_reshape:fp32,conv_sm_object:fp32,yolo_conv5_1_lrelu:fp32,concatenate_6:fp32,yolo_conv3_1_lrelu:fp32,concatenate_5:fp32,yolo_neck_1_lrelu:fp32 \
--precisionConstraints obey models/tao_pretrained_models/yolov4-tiny/yolov4_tiny_usa_deployable.etlt 
cp models/tao_pretrained_models/yolov4-tiny/usa_lpd_label.txt triton_models/us/us_lpd_yolov4-tiny

echo "prepare us_lprnet"
mkdir -p triton_models/us/us_lprnet/1/
./tao-converter -k nvidia_tlt -t fp16 -e triton_models/us/us_lprnet/1/us_lprnet_baseline18_deployable.etlt_b16_gpu0_fp16.engine -p image_input,1x3x48x96,8x3x48x96,16x3x48x96 models/LP/LPR/us_lprnet_baseline18_deployable.etlt

