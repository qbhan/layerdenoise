import torch
from layer_network import LayerNet
from affn import AffinityNet
from utils import *
from torch_utils import WeightedFilter

torch.autograd.set_detect_anomaly(True)
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

# import time
# iter = 20
# t_affn = 0.0
# for _ in range(iter):
#     data = {
#     'radiance': torch.randn((8, spp, 3, 128, 128)).cuda(),
#     'features': torch.randn((8, spp, n_in, 128, 128)).cuda(),
#     }
#     torch.cuda.synchronize()
#     start = time.time()
#     a = affn(data)
#     torch.cuda.synchronize()
#     t_affn += time.time() - start
# print('speed affn: {:.3f} sec'.format(t_affn / iter))
# t_layer = 0.0
# for _ in range(iter):
#     data = {
#     'radiance': torch.randn((8, spp, 3, 128, 128)).cuda(),  
#     'features': torch.randn((8, spp, n_in, 128, 128)).cuda(),
#     }
#     torch.cuda.synchronize()
#     start = time.time()
#     a = layer(data)
#     torch.cuda.synchronize()
#     t_layer += time.time() - start
# print('speed layer: {:.3f} sec'.format(t_layer / iter))
    
 
c = torch.randn((8, 3, 128, 128)).cuda()
loss = torch.nn.functional.mse_loss(a, c)
loss.backward()