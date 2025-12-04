import imageio.v3 as iio, numpy as np
for name in ["1.tif", "2.tif"]:
    arr = iio.imread(name)           # (5, H, W)
    ch5 = arr[4]                     # 第 5 个通道
    proj_all = arr.max(axis=0)       # 所有通道的最大投影
    iio.imwrite(name.replace(".tif", "_ch5.tif"), ch5)