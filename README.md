# LightTrack: Finding Lightweight Neural Networks for Object Tracking via One-Shot Architecture Search (ncnn)

The official implementation by pytorch:

https://github.com/researchmm/LightTrack

# 0. Download vulkan sdk
https://sdk.lunarg.com/sdk/download/1.3.204.1/linux/vulkansdk-linux-x86_64-1.3.204.1.tar.gz

# 1. How to build and run it?

## modify your own CMakeList.txt
modify vulkansdk path as yours

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

https://github.com/Z-Xiong/LightTrack-ncnn/issues/2#issuecomment-1000163993# *PS: 这里还是拆分为三个模型的torch2onnx.py，转为为两个模型请参考torch2pnnx.py*


**OR you can export pnnx model from officical code which is is more convenient than onnx model.**

# *PS: 这里做了更新，把原来的三个模型中的backbone和neck_head模型合并为了一个update模型*
### (1) Download the pnnx program
https://github.com/pnnx/pnnx
### (2) modify LightTrack/lib/models/super_model_DP.py as follows:
```
class Super_model_DP_retrain(Super_model):
    def __init__(self, search_size=256, template_size=128, stride=16):
        super(Super_model_DP_retrain, self).__init__(search_size=search_size, template_size=template_size,
                                                     stride=stride)

    def template(self, z):
        # print("z shape is:", z.shape)
        self.zf = self.features(z)

    def track(self, x):
        # print("x shape is:", x.shape)
        # supernet backbone
        xf = self.features(x)

        # print("xf shape is:", xf.shape)
        # print("zf shape is:", self.zf.shape)
        # BN before Pointwise Corr
        zf, xf = self.neck(self.zf, xf)
        # Point-wise Correlation
        feat_dict = self.feature_fusor(zf, xf)
        # supernet head
        oup = self.head(feat_dict)
        return oup['cls'], oup['reg']

    # when convert model, annotate this function
    def forward(self, template, search, label, reg_target, reg_weight):
        '''backbone_index: which layer's feature to use'''
        zf = self.features(template)
        xf = self.features(search)
        # Batch Normalization before Corr
        zf, xf = self.neck(zf, xf)
        # Point-wise Correlation
        feat_dict = self.feature_fusor(zf, xf)
        # supernet head
        oup = self.head(feat_dict)
        # compute loss
        reg_loss = self.add_iouloss(oup['reg'], reg_target, reg_weight)
        cls_loss = self._weighted_BCE(oup['cls'], label)
        return cls_loss, reg_loss

    # # convert init-net
    # def forward(self, z):   # init
    #     self.zf = self.features(z)
    #     return self.zf

    # # convert update
    # def forward(self, zf, x):   # update
    #     xf = self.features(x)
    #     # BN before Pointwise Corr
    #     zf, xf = self.neck(zf, xf)
    #     # Point-wise Correlation
    #     feat_dict = self.feature_fusor(zf, xf)
    #     # supernet head
    #     oup = self.head(feat_dict)
    #     return oup['cls'], oup['reg']
```
### (3) modify torch2onnx.py to torch2pnnx.py as follows:
```
    # # convert init-net
    # x = torch.randn(1, 3, 127, 127)
    # mod = torch.jit.trace(siam_net, x)
    # mod.save("ligthtrack_init.pt")
    # os.system("/home/pxierra/nas/6-Software/pnnx-20211223-ubuntu/pnnx ligthtrack_init.pt inputshape=[1,3,127,127]")

    # convert update
    zf = torch.randn(1, 96, 8, 8)
    x = torch.randn(1, 3, 288, 288)
    mod = torch.jit.trace(siam_net, (zf, x))
    mod.save("ligthtrack_update.pt")
    os.system("/home/pxierra/nas/6-Software/pnnx-20211223-ubuntu/pnnx ligthtrack_update.pt inputshape=[1,96,8,8],[1,3,288,288]")
```
### (4) run torch2pnnx.py
```
$ cd LightTrack/tracking
$ python torch2pnnx.py --arch LightTrackM_Subnet --resume ../snapshot/LightTrackM/LightTrackM.pth --stride 16 --path_name back_04502514044521042540+cls_211000022+reg_100000111_ops_32
```

