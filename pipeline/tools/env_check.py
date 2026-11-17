import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, os.path.expanduser('~/projects/detector-program/OpenPCDet'))
import torch  # noqa: E402
import spconv  # noqa: E402
import pcdet  # noqa: E402

print('torch', torch.__version__, 'cuda visible', torch.cuda.is_available(), 'spconv', getattr(spconv, '__version__', '?'), 'pcdet', pcdet.__file__)
import spconv.pytorch as sp  # noqa: E402
x = sp.SparseConvTensor(torch.randn(10, 4), torch.stack([torch.zeros(10, dtype=torch.int32), torch.randint(0, 8, (10,), dtype=torch.int32),
                                                          torch.randint(0, 8, (10,), dtype=torch.int32), torch.randint(0, 8, (10,), dtype=torch.int32)], 1), [8, 8, 8], 1)
conv = sp.SubMConv3d(4, 8, 3, indice_key='a')
print('spconv cpu submconv ok', conv(x).features.shape)
