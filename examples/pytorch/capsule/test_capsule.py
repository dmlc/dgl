from capsule_model import DGLCapsuleLayer
import torch as th

device='cuda'

model = DGLCapsuleLayer(in_units=8, in_channels=1152, num_units=10, use_routing=True, unit_size=16)

x=th.randn((128,8,1152))

model(x)