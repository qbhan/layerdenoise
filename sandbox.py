import torch
from layer_network import LayerNet
from affn import AffinityNet
from utils import *
from torch_utils import WeightedFilter

n_in = 24
spp = 8

layer = LayerNet(n_in, tonemap_log, True).cuda()
affn = AffinityNet(n_in, True).cuda()


data = {
    'radiance': torch.randn((8, spp, 3, 128, 128)).cuda(),
    'features': torch.randn((8, spp, n_in, 128, 128)).cuda(),
}
a = affn(data)
b = layer(data)
print(a.shape, b.shape)

import time
iter = 20
t_affn = 0.0
for _ in range(iter):
    data = {
    'radiance': torch.randn((8, spp, 3, 128, 128)).cuda(),
    'features': torch.randn((8, spp, n_in, 128, 128)).cuda(),
    }
    torch.cuda.synchronize()
    start = time.time()
    a = affn(data)
    torch.cuda.synchronize()
    t_affn += time.time() - start
print('speed affn: {:.3f} sec'.format(t_affn / iter))
t_layer = 0.0
for _ in range(iter):
    data = {
    'radiance': torch.randn((8, spp, 3, 128, 128)).cuda(),
    'features': torch.randn((8, spp, n_in, 128, 128)).cuda(),
    }
    torch.cuda.synchronize()
    start = time.time()
    a = layer(data)
    torch.cuda.synchronize()
    t_layer += time.time() - start
print('speed layer: {:.3f} sec'.format(t_layer / iter))
    
 
# f = WeightedFilter(channels=3, kernel_size=13, bias=False, splat=True, level=2).cuda()
# out = f(data['radiance'], torch.randn(8, 13*13, 128, 128).cuda().contiguous())
# print(out.shape)

# import torch.nn.functional as F

# a = torch.arange(128*128).view(1, 1, 128, 128)
# print(a.shape)
# b = a.unfold(2, 3, 1).unfold(3, 3, 1)
# print(b.shape)
# print(b[0, 0, 0, 1])
# c = a.unfold(2, 3, 2).unfold(3, 3, 2)
# print(c.shape)
# print(c[0, 0, 0, 1])