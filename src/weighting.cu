#include "weighting.cuh"

torch::Tensor weighting_cuda_forward(
        const torch::Tensor &x_ori,
        const torch::Tensor &x_weight,
        const int kH, const int kW
) {
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

    int start_inp = 0;
    for (int i=0; i<batch; ++i) {
        auto x_weight_row = x_weight.select(0, i);
        weighting_forward_warper<float, double> (
                at::cuda::getCurrentCUDAStream(),
                x_ori.data_ptr<float>() + start_inp,
                x_weight_row.packed_accessor32<float, 3>(),
                kH, kW, rH, rW,
                patch, height, width,
                per_channel, per_input,
                output.data_ptr<float>() + start_inp
        );
        start_inp += per_input;
    }
    cudaDeviceSynchronize();

    return output;
}

//////////////////////////////////////////////////////////////

torch::Tensor weighting_cuda_backward_ori(
        const torch::Tensor &x_weight,
        const torch::Tensor &grad_out,
        const int kH, const int kW
) {
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
    auto grad_ori = torch::empty({batch, channels, height, width}, x_weight.options());

    int start_inp = 0;
    for (int i=0; i<batch; ++i) {
        auto x_weight_row = x_weight.select(0, i);
        auto grad_out_row = grad_out.select(0, i);
        weighting_backward_ori_warper<float, double> (
                at::cuda::getCurrentCUDAStream(),
                x_weight_row.packed_accessor32<float, 3>(),
                grad_out_row.packed_accessor32<float, 3>(),
                kH, kW, rH, rW,
                patch, height, width,
                per_channel, per_input,
                grad_ori.data_ptr<float>() + start_inp
        );
        start_inp += per_input;
    }
    cudaDeviceSynchronize();

    return grad_ori;
}

//////////////////////////////////////////////////////////////

torch::Tensor weighting_cuda_backward_weight(
        const torch::Tensor &x_ori,
        const torch::Tensor &grad_out,
        const int kH, const int kW
) {
    TypeCheck(x_ori);
    const int batch = x_ori.size(0);
    const int channels = x_ori.size(1);
    const int height = x_ori.size(2);
    const int width = x_ori.size(3);

    const int rH = kH / 2;
    const int rW = kW / 2;
    const int patch = kH * kW;
    const int per_channel = height * width;
    const int per_input = per_channel * channels;
    const int per_output = per_channel * patch;
    auto grad_weight = torch::empty({batch, height, width, patch}, x_ori.options());

    int start_inp = 0, start_out = 0;
    for (int i=0; i<batch; ++i) {
        auto grad_out_row = grad_out.select(0, i).contiguous();
        weighting_backward_weight_warper<float, double> (
                at::cuda::getCurrentCUDAStream(),
                x_ori.data_ptr<float>() + start_inp,
                grad_out_row.data_ptr<float>(),
                kH, kW, rH, rW,
                patch, channels, height, width,
                per_channel,
                grad_weight.data_ptr<float>() + start_out
        );
        start_inp += per_input;
        start_out += per_output;
    }
    cudaDeviceSynchronize();

    return grad_weight;
}