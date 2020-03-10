import torch
from torch.nn import functional as F
from function import f_weighting, TorchLocalAttention
import time


def check(a, b):
    return (a-b).abs().max()


def f_torch(x_theta, x_phi, kh, kw):
    n, c, h, w = x_theta.size()  # (N, inter_channels, H, W)
    pad = (kh//2, kw//2)
    x_theta = F.unfold(x_theta, kernel_size=(kh, kw), stride=1, padding=pad)
    x_theta = x_theta.permute(0, 2, 1).contiguous()
    x_theta = x_theta.view(n*h*w, c, kh*kw)

    x_phi = x_phi.view(n*h*w, kh*kw, 1)

    out = torch.matmul(x_theta, x_phi)
    out = out.squeeze(-1)
    out = out.view(n, h, w, c)
    out = out.permute(0, 3, 1, 2).contiguous()

    return out


def test_correct(h, w, c, kh, kw):
    x1 = torch.rand(4, c, h, w).cuda()
    y1 = torch.rand(4, h, w, kh*kw).cuda()
    x2 = x1.clone()
    y2 = y1.clone()

    x1.requires_grad_()
    y1.requires_grad_()
    x2.requires_grad_()
    y2.requires_grad_()

    z1 = TorchLocalAttention.f_weighting(x1, y1, kh, kw)
    z2 = f_weighting(x2, y2, kh, kw)

    grad = torch.rand(z1.size()).cuda()

    z1.backward(grad)
    z2.backward(grad)

    err1 = check(z1.data, z2.data)
    err2 = check(x1.grad.data, x2.grad.data)
    err3 = check(y1.grad.data, y2.grad.data)
    print("maximum difference: {:.5f}\t{:.5f}\t{:.5f}".format(err1.item(), err2.item(), err3.item()))

    
def test_efficiency_forward(h, w, c, kh, kw):
    x = torch.rand(1, c, h, w).cuda()
    y = torch.rand(1, h, w, kh*kw).cuda()

    with torch.no_grad():
        torch.cuda.reset_max_memory_allocated()
        z = f_weighting(x, y, kh, kw)
        memory = torch.cuda.max_memory_allocated() / 1000000
        del z

    with torch.no_grad():
        torch.cuda.reset_max_memory_allocated()
        z = TorchLocalAttention.f_weighting(x, y, kh, kw)
        memory_torch = torch.cuda.max_memory_allocated() / 1000000
        del z

    with torch.no_grad():
        torch.cuda.synchronize()
        t = time.time()
        for i in range(3):
            z = f_weighting(x, y, kh, kw)
        torch.cuda.synchronize()
        t = (time.time() - t) / 3
        del z

        torch.cuda.synchronize()
        t_torch = time.time()
        for i in range(3):
            z = TorchLocalAttention.f_weighting(x, y, kh, kw)
        torch.cuda.synchronize()
        t_torch = (time.time() - t_torch) / 3
        del z
    print("{:.2f},{:.2f}||{:.5f},{:.5f}".format(memory_torch, memory, t_torch, t))


def test_efficiency_backward(h, w, c, kh, kw):
    x = torch.rand(1, c, h, w).cuda()
    y = torch.rand(1, h, w, kh*kw).cuda()
    x.requires_grad_()
    y.requires_grad_()

    torch.cuda.reset_max_memory_allocated()
    z = f_weighting(x, y, kh, kw)
    grad = torch.rand(z.size()).cuda()
    z.backward(grad)
    memory = torch.cuda.max_memory_allocated() / 1000000
    x.grad.data.zero_()
    y.grad.data.zero_()
    del z

    torch.cuda.reset_max_memory_allocated()
    z = TorchLocalAttention.f_weighting(x, y, kh, kw)
    grad = torch.rand(z.size()).cuda()
    z.backward(grad)
    memory_torch = torch.cuda.max_memory_allocated() / 1000000
    x.grad.data.zero_()
    y.grad.data.zero_()
    del z

    torch.cuda.synchronize()
    t = time.time()
    for i in range(3):
        z = f_weighting(x, y, kh, kw)
        z.backward(grad)
        x.grad.data.zero_()
        y.grad.data.zero_()
    torch.cuda.synchronize()
    t = (time.time() - t) / 3
    del z

    torch.cuda.synchronize()
    t_torch = time.time()
    for i in range(3):
        z = f_torch(x, y, kh, kw)
        z.backward(grad)
        x.grad.data.zero_()
        y.grad.data.zero_()
    torch.cuda.synchronize()
    t_torch = (time.time() - t_torch) / 3
    del z
    print("{:.2f},{:.2f}||{:.5f},{:.5f}".format(memory_torch, memory, t_torch, t))


if __name__ == '__main__':
    for im in [128, 64, 32]:
        for c in [64, 32 ,16]:
            for block in [21, 11, 5]:
                print("input:{} channel:{} block:{}".format(im, c, block))
                test_correct(im, im, c, block, block)
                test_efficiency_forward(im, im, c, block, block)
                test_efficiency_backward(im, im, c, block, block)
