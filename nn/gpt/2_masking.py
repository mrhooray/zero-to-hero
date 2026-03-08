import torch
import torch.nn.functional as F

torch.manual_seed(42)
B, T, C = 4, 8, 2
x = torch.rand(B, T, C)
print(x.shape)
print(x)

# mean operation for example
# dot product for attention

# v1
xbow1 = torch.zeros_like(x)
for b in range(B):
    for t in range(T):
        xacc = x[b, : t + 1]
        xbow1[b, t] = xacc.mean(dim=0)
print(xbow1)

# v2
tril = torch.ones(T, T).tril()
w = tril / tril.sum(dim=1, keepdim=True)
xbow2 = w @ x
print(torch.allclose(xbow1, xbow2))

# v3
tril = torch.ones(T, T).tril()
w = torch.zeros(T, T).masked_fill(tril == 0, -torch.inf)
w = F.softmax(w, dim=1)
xbow3 = w @ x
print(torch.allclose(xbow1, xbow3))
