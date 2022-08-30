#!/bin/bash

echo "p1:" $1
echo "p2:" $2


function convert_model {
    echo "in convert_model $1"
    echo "tao-converter for ds5.0.1"
    if [ ! -f "tao-converter" ]; then
       echo "tao-converter does exist, please refer to section Prerequisition"
    else
      if [ "$1" = "us" ]; then
          ./tao-converter -k nvidia_tlt -p image_input,1x3x48x96,4x3x48x96,16x3x48x96 \
           models/LP/LPR/us_lprnet_baseline18_deployable.etlt -t fp16 -e models/LP/LPR/us_lprnet_baseline18_deployable.etlt_b16_gpu0_fp16.engine
      else
          ./tao-converter -k nvidia_tlt -p image_input,1x3x48x96,4x3x48x96,16x3x48x96 \
           models/LP/LPR/ch_lprnet_baseline18_deployable.etlt -t fp16 -e models/LP/LPR/ch_lprnet_baseline18_deployable.etlt_b16_gpu0_fp16.engine
      fi 
    fi
}

if [ "$1" = "us" ]; then
  echo "start download_us.sh"
  chmod 775 download_us.sh
  ./download_us.sh
else
  echo "start download_ch.sh"
  chmod 775 download_ch.sh
  ./download_ch.sh
fi

#DS5.0.1 gst-nvinfer cannot generate TRT engine for LPR model, so generate it with tao-converter
if [ "$2" = "1" ]; then
   convert_model $1
fi
