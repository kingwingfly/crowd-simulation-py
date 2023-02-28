import numpy as np
from torch import tensor
from torch.nn.functional import normalize

print(normalize(tensor([-1, -2, -3], dtype=float), p=1, dim=0))
