## Prepare Triton Server For Native Inferencing
As mentioned in the README, the DeepStream LPR sample application should work as Triton client with Triton Server running natively for cAPIs. So the [Triton Inference Server libraries](https://github.com/triton-inference-server/client) should be installed in the machine. A easier way is to run the LPR sample application in the [DeepStream Triton container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/deepstream). 

Run DeepStream Triton container, take DeepStream 6.1 GA container as the example:
```
    docker run --gpus all -it  --ipc=host --rm -v /tmp/.X11-unix:/tmp/.X11-unix  -v $(pwd)/deepstream_lpr_app:/lpr   -e DISPLAY=$DISPLAY -w /lpr nvcr.io/nvidia/deepstream:6.1-triton
```
Inside the container, prepare model engines for Triton server:
```
    //For US car plate recognition
    ./prepare_triton_us.sh

    //For Chinese car plate recognition
    ./prepare_triton_ch.sh
```

Then the LPR sample application can be build and run inside this container.
