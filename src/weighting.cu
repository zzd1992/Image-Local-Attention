#include "kernels.cuh"

torch::Tensor weighting_cuda_forward(const torch::Tensor &x_ori,
                                     const torch::Tensor &x_weight,
                                     const int kH, const int kW) {
  TypeCheck(x_ori);
  TypeCheck(x_weight);
  const int batch = x_ori.size(0);
  const int channels = x_ori.size(1);
  const int height = x_ori.size(2);
  const int width = x_ori.size(3);

  const int rH = kH / 2;
  const int rW = kW / 2;
  const int patch = kH * kW;
  const int per_channel = height * width;
  const int per_input = per_channel * channels;
  auto output = torch::empty({batch, channels, height, width}, x_ori.options());

  f_ck2c_ori<float, double>(
      at::cuda::getCurrentCUDAStream(), x_ori.data_ptr<float>(),
      x_weight.data_ptr<float>(), kH, kW, rH, rW, patch, channels, height,
      width, per_channel, per_input, batch, output.data_ptr<float>());
  return output;
}

//////////////////////////////////////////////////////////////

torch::Tensor weighting_cuda_backward_ori(const torch::Tensor &x_weight,
                                          const torch::Tensor &grad_out,
                                          const int kH, const int kW) {
  TypeCheck(x_weight);
  const int batch = x_weight.size(0);
  const int channels = grad_out.size(1);
  const int height = x_weight.size(1);
  const int width = x_weight.size(2);

  const int rH = kH / 2;
  const int rW = kW / 2;
  const int patch = kH * kW;
  const int per_channel = height * width;
  const int per_input = per_channel * channels;
  /* const int per_output = per_channel * patch; */
  auto grad_ori =
      torch::empty({batch, channels, height, width}, x_weight.options());

  f_ck2c_loc<float, double>(
      at::cuda::getCurrentCUDAStream(), grad_out.data_ptr<float>(),
      x_weight.data_ptr<float>(), kH, kW, rH, rW, patch, channels, height,
      width, per_channel, per_input, batch, grad_ori.data_ptr<float>());

  return grad_ori;
}

//////////////////////////////////////////////////////////////

torch::Tensor weighting_cuda_backward_weight(const torch::Tensor &x_ori,
                                             const torch::Tensor &grad_out,
                                             const int kH, const int kW) {
  TypeCheck(x_ori);
  const int batch = x_ori.size(0);
  const int channels = x_ori.size(1);
  const int height = x_ori.size(2);
  const int width = x_ori.size(3);

  const int rH = kH / 2;
  const int rW = kW / 2;
  const int patch = kH * kW;
  const int per_channel = height * width;
  auto grad_weight =
      torch::empty({batch, height, width, patch}, x_ori.options());

  f_cc2k<float, double>(at::cuda::getCurrentCUDAStream(),
                        grad_out.data_ptr<float>(), x_ori.data_ptr<float>(), kH,
                        kW, rH, rW, patch, channels, height, width, per_channel,
                        batch, grad_weight.data_ptr<float>());

  return grad_weight;
}
