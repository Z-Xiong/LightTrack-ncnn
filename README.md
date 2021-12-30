# LightTrack: Finding Lightweight Neural Networks for Object Tracking via One-Shot Architecture Search (ncnn)

The official implementation by pytorch:

https://github.com/researchmm/LightTrack

# 1. How to build and run it?

## modify your own CMakeList.txt

## build
```
$ mkdir build && cd build
$ cmake .. && make -j 
$ make install
```

## run
```
$ cd install/lighttrack_demo
$ ./LightTrack [videopath(file or camera)]
```

# 2. Q&A

## 1. How to export onnx model from officical code?

https://github.com/Z-Xiong/LightTrack-ncnn/issues/2#issuecomment-1000163993


**OR you can export pnnx model from officical code which is is more convenient than onnx model.**


### (1) Download the pnnx program
https://github.com/pnnx/pnnx
### (2) modify torch2onnx.py to torch2pnnx.py as follows:
```
    # # convert init-net
    # x = torch.randn(1, 3, 127, 127)
    # mod = torch.jit.trace(siam_net, x)
    # mod.save("ligthtrack_init.pt")
    # os.system("./pnnx ligthtrack_init.pt inputshape=[1,3,127,127]")

    # # # convert backone-net
    # x = torch.randn(1, 3, 288, 288)
    # mod = torch.jit.trace(siam_net, x)
    # mod.save("ligthtrack_backbone.pt")
    # os.system("./pnnx ligthtrack_backbone.pt inputshape=[1,3,288,288]")

    # convert track
    zf = torch.randn(1, 96, 8, 8)
    xf = torch.randn(1, 96, 18, 18)
    mod = torch.jit.trace(siam_net, (zf, xf))
    mod.save("ligthtrack_neckhead.pt")
    os.system("./pnnx ligthtrack_neckhead.pt inputshape=[1,96,8,8],[1,96,18,18]")
```
### (3) run torch2pnnx.py
```
$ cd LightTrack/tracking
$ python torch2pnnx.py --arch LightTrackM_Subnet --resume ../snapshot/LightTrackM/LightTrackM.pth --stride 16 --path_name back_04502514044521042540+cls_211000022+reg_100000111_ops_32
```

