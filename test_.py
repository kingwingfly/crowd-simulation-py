from torch import tensor
from torch.nn.functional import normalize

t1 = tensor([1, 2, 3, 0.5], dtype=float)
print(normalize(t1, p=1, dim=0))

lst = [1]
lst[0] -= 1
print(lst)
