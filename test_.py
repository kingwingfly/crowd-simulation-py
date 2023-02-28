import numpy as np
from torch import tensor
from torch.nn.functional import normalize

dic = {1: tensor([1, 2, 3], dtype=float)}

print(normalize(dic[1], p=1, dim=0))
