import torch
from torch import nn
from torch.nn import functional as F
from function import LocalAttention, TorchLocalAttention
import time


def test_efficiency_forward(h, w, c, kh, kw):
    x = torch.rand(1, c, h, w).cuda()

    m = LocalAttention(c, c, kh, kw).cuda()
    m_torch = TorchLocalAttention(c, c, kh, kw).cuda()

    with torch.no_grad():
        torch.cuda.reset_max_memory_allocated()
        z = m_torch(x)
        memory_torch = torch.cuda.max_memory_allocated() / 1000000
        del z

    with torch.no_grad():
        torch.cuda.reset_max_memory_allocated()
        z = m(x)
        memory = torch.cuda.max_memory_allocated() / 1000000
        del z

    with torch.no_grad():
        torch.cuda.synchronize()
        t_torch = time.time()
        for i in range(3):
            z = m_torch(x)
        torch.cuda.synchronize()
        t_torch = (time.time() - t_torch) / 3
        del z

        torch.cuda.synchronize()
        t = time.time()
        for i in range(3):
            z = m(x)
        torch.cuda.synchronize()
        t = (time.time() - t) / 3
        del z
    print("{:.2f},{:.2f}||{:.5f},{:.5f}".format(memory_torch, memory, t_torch, t))

    
def test_efficiency_backward(h, w, c, kh, kw):
    x = torch.rand(1, c, h, w).cuda()
    x.requires_grad_()

    m_torch = TorchLocalAttention(c, c, kh, kw).cuda()
    torch.cuda.reset_max_memory_allocated()
    z = m_torch(x)
    grad = torch.rand(z.size()).cuda()
    z.backward(grad)
    memory_torch = torch.cuda.max_memory_allocated() / 1000000
    x.grad.data.zero_()
    del z

    m = LocalAttention(c, c, kh, kw).cuda()
    torch.cuda.reset_max_memory_allocated()
    z = m(x)
    grad = torch.rand(z.size()).cuda()
    z.backward(grad)
    memory = torch.cuda.max_memory_allocated() / 1000000
    x.grad.data.zero_()
    del z

    torch.cuda.synchronize()
    t_torch = time.time()
    for i in range(3):
        z = m_torch(x)
        z.backward(grad)
        x.grad.data.zero_()
    torch.cuda.synchronize()
    t_torch = (time.time() - t_torch) / 3
    del z

    torch.cuda.synchronize()
    t = time.time()
    for i in range(3):
        z = m(x)
        z.backward(grad)
        x.grad.data.zero_()
    torch.cuda.synchronize()
    t = (time.time() - t) / 3
    del z
    print("{:.2f},{:.2f}||{:.5f},{:.5f}".format(memory_torch, memory, t_torch, t))


if __name__ == '__main__':
    for im in [128, 64, 32]:
        for c in [64, 32 ,16]:
            for block in [21, 11, 5]:
                print("input:{} channel:{} block:{}".format(im, c, block))
                test_efficiency_forward(im, im, c, block, block)
                test_efficiency_backward(im, im, c, block, block)
   
