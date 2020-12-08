import decord


path = '../../../../raid/Kinetics_700/train/zumba/'
name = 'Zx7Ixqyq3-o_000264_000274.mp4'

vr = decord.VideoReader(path+name)
print('native output:', type(vr[0]), vr[0].shape)
# native output: <class 'decord.ndarray.NDArray'>, (240, 426, 3)
# you only need to set the output type once
#decord.bridge.set_bridge('mxnet')
# <class 'mxnet.ndarray.ndarray.NDArray'> (240, 426, 3)
# or pytorch and tensorflow(>=2.2.0)
#print(vr.shape)
decord.bridge.set_bridge('torch')
print(len(vr),vr[0].shape)
#print(type(vr[0], vr[0].shape))

#decord.bridge.set_bridge('tensorflow')
# or back to decord native format
#decord.bridge.set_bridge('native')
