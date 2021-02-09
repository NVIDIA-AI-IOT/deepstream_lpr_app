lpr-test-app should work with libnvdsinfer_custom_impl_lpr.so for the customized post-processing of the TLT LPR model.

libnvdsinfer_custom_impl_lpr.so is built with nvinfer_custom_lpr_parser for car plate recognition.
Copy the libnvdsinfer_custom_impl_lpr.so file to /opt/nvidia/deepstream/deepstream-5.0/lib/ folder after build.

TLT LPR model trained for different country car plate OCR recognition with different language dictionary for car plate.
Prepare the dictionary file for the OCR according to the trained TLT LPR model. The dictionary file name should be "dict.txt". In the dictionary file, one line for one character. 
The format of the dictionary file can refer to the sample dictionary file of "dict_us.txt" and "dict_ch.txt" for US car plate recognition model and Chinese car plate recognition modle seperately.
The dictionary file should be in the same directory of the built binary of lpr-test-app and the other config files.

This sample only support MP4 files with h264 video.

The command line is:

./lpr-test-app <1:US car plate model|2: Chinese car plate model> <1: output as h264 file| 2:fakesink 3:display output> <0:ROI disable|0:ROI enable> <input mp4 file name> <input mp4 file name> ... <output file name>

For example:
Current default sample of U.S.A plate recognition, e.g. the command line for 16 streams recognition is:
./lpr-test-app 1 2 0 <input file 0> <input file 1> <input file 2> <input file 3> <input file 4> <input file 5> <input file 6> <input file 7> <input file 8> <input file 9> <input file 10> <input file 11> <input file 12> <input file 13> <input file 14> <input file 15> <output file name>             

For Chinese plate recognition, please make sure the Chinese language support is in the OS.
For example:
With Ubuntu
1. Install Chinese Language package . 
   sudo apt-get install language-pack-zh-hans
2. Set the Chinese language enviroment
   export LANG=zh_CN.UTF-8
   export LANGUAGE="zh_CN:zh:en_US:en"