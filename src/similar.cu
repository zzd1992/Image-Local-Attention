#include "kernels.cuh"

torch::Tensor similar_cuda_forward(const torch::Tensor &x_ori,
                                   const torch::Tensor &x_loc, const int kH,
                                   const int kW) {
  TypeCheck(x_ori);
  TypeCheck(x_loc);
  const int batch = x_ori.size(0);
  const int channels = x_ori.size(1);
  const int height = x_ori.size(2);
  const int width = x_ori.size(3);

  const int rH = kH / 2;
  const int rW = kW / 2;
  const int patch = kH * kW;
  const int per_channel = height * width;
  auto output = torch::empty({batch, height, width, patch}, x_ori.options());

  f_cc2k<float, double>(at::cuda::getCurrentCUDAStream(),
                        x_ori.data_ptr<float>(), x_loc.data_ptr<float>(), kH,
                        kW, rH, rW, patch, channels, height, width, per_channel,
                        batch, output.data_ptr<float>());

  return output;
}

//////////////////////////////////////////////////////////////

torch::Tensor similar_cuda_backward(const torch::Tensor &x,
                                    const torch::Tensor &grad_out, const int kH,
                                    const int kW, const bool is_ori) {
  TypeCheck(x);
  const int batch = x.size(0);
  const int channels = x.size(1);
  const int height = x.size(2);
  const int width = x.size(3);

  const int rH = kH / 2;
  const int rW = kW / 2;
  const int patch = kH * kW;
  const int per_channel = height * width;
  const int per_input = per_channel * channels;

  auto grad_inp = torch::empty({batch, channels, height, width}, x.options());

  if (is_ori) {
    f_ck2c_ori<float, double>(
        at::cuda::getCurrentCUDAStream(), x.data_ptr<float>(),
        grad_out.data_ptr<float>(), kH, kW, rH, rW, patch, channels, height,
        width, per_channel, per_input, batch, grad_inp.data_ptr<float>());
  } else {
    f_ck2c_loc<float, double>(
        at::cuda::getCurrentCUDAStream(), x.data_ptr<float>(),
        grad_out.data_ptr<float>(), kH, kW, rH, rW, patch, channels, height,
        width, per_channel, per_input, batch, grad_inp.data_ptr<float>());
  }

  return grad_inp;
}
