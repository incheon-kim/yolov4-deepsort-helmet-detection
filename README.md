# Yolov4-helmet-detection

trained with [darknet framework](https://github.com/AlexeyAB/darknet)

## tested environment
- `Windows 10 x64 2020 (build 19041.388)`
- `NVIDIA RTX 2070 Super`
- `CUDA 10.1 / CuDNN 7.6.5`
- `Python 3.7.7 x64`

## dependencies
- basically darknet dependencies
    - python
        - opencv-python, numpy, scikit-image
    - CUDA 10.1 / CuDNN 7.6.5
    - darknet library
        - `dark.dll + pthreadVC2.dll (windows)`
            - Pre-Compiled - [Google Drive](https://drive.google.com/file/d/1D3bYPyGgWUZavLsDh5SyU0yyPqW-5xiC)
        - `libdarknet.so (linux)`


## trained weight
- [Google Drive](https://drive.google.com/file/d/1uOWZGx1oR1bRwp_mnvxobaXZcWs1X9ar)
- put weight file in `./configs`


## used dataset
- [roboflow.ai public dataset](https://public.roboflow.ai/object-detection/hard-hat-workers) \+ 100 images

## example
