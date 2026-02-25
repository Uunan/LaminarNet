import torch
import torch.nn.functional as F
import torch.nn as nn
from laminarnet.model import CrossStratumRouting

B, N, D = 1, 10, 1
stride = 2

csr = CrossStratumRouting(D, stride)
csr.eval()

h_fine_seq = torch.arange(1, 11).view(B, N, 1).float()
h_coarse_seq = torch.zeros(B, N // stride, 1).float()

with torch.no_grad():
    fwd_f, fwd_c = csr(h_fine_seq, h_coarse_seq)

# Simulating step
out_f = []
out_c = []
buf_f = torch.zeros(B, stride - 1, D)
last_c = torch.zeros(B, 1, D) # To hold the upsampled coarse token

for t in range(N):
    x_t = h_fine_seq[:, t:t+1, :]
    
    # Check if we compute a new coarse token at time t
    if t % stride == 0:
        # At t=0, window is [0, x0] (buf is 0, x_t is x0)
        # At t=2, window is [x1, x2] (buf is x1, x_t is x2)
        window = torch.cat([buf_f, x_t], dim=1) # shape (B, stride, D)
        
        # downsample window
        f_to_c = window.mean(dim=1, keepdim=True)
        
        # get matching coarse input token at t // stride
        h_c_in = h_coarse_seq[:, t // stride: t // stride + 1, :]
        
        # update coarse
        g_f2c = csr.gate_f2c(f_to_c)
        last_c = h_c_in + g_f2c * f_to_c
        out_c.append(last_c)
    
    # Regardless of updating or not, we emit `last_c` for fine update
    c_to_f = last_c  # In step, Upsample is just repeat, so for 1 token it's itself
    
    g_c2f = csr.gate_c2f(c_to_f)
    new_x_t = x_t + g_c2f * c_to_f
    out_f.append(new_x_t)
    
    # Shift buffer: buf_f becomes x_t (for stride=2, buf is length 1)
    # for general stride, rotate out oldest
    buf_f = torch.cat([buf_f[:, 1:], x_t], dim=1)

step_f = torch.cat(out_f, dim=1)
step_c = torch.cat(out_c, dim=1)

print("FWD F:")
print(fwd_f[0, :, 0])
print("STEP F:")
print(step_f[0, :, 0])

print("Diff F:", (fwd_f - step_f).abs().max().item())
print("Diff C:", (fwd_c - step_c).abs().max().item())

