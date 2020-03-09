#include "similar.cuh"

torch::Tensor similar_cuda_forward(
        const torch::Tensor &x_ori,
        const torch::Tensor &x_loc,
        const int kH, const int kW
) {
    TypeCheck(x_ori);
    TypeCheck(x_loc);
    const int batch = x_ori.size(0);
    const int channels = x_ori.size(1);
    const int height = x_ori.size(2);
    const int width = x_ori.size(3);

    const int rH = kH / 2;
    const int rW = kW / 2;
    const int patch = kH * kW;
    const int per_input = height * width * channels;
    const int per_output = height * width * patch;
    auto output = torch::empty({batch, height, width, patch}, x_ori.options());

    int start_inp = 0, start_out = 0;
    for (int i=0; i<batch; ++i) {
        similar_forward_warper<float, double> (
                at::cuda::getCurrentCUDAStream(),
                x_ori.data_ptr<float>() + start_inp,
                x_loc.data_ptr<float>() + start_inp,
                kH, kW, rH, rW,
                patch, channels, height, width,
                output.data_ptr<float>() + start_out
        );
        start_inp += per_input;
        start_out += per_output;
    }
    cudaDeviceSynchronize();

    return output;
}

//////////////////////////////////////////////////////////////

torch::Tensor similar_cuda_backward(
        const torch::Tensor &x,
        const torch::Tensor &grad_out,
        const int kH, const int kW,
        const bool is_ori
) {
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

    int start_inp = 0;
    for (int i=0; i<batch; ++i) {
        auto grad_out_row = grad_out.select(0, i);
        similar_backward_warper<float, double> (
                at::cuda::getCurrentCUDAStream(),
                x.data_ptr<float>() + start_inp,
                grad_out_row.packed_accessor32<float, 3>(),
                is_ori,
                kH, kW, rH, rW,
                patch,
                height, width,
                per_channel, per_input,
                grad_inp.data_ptr<float>() + start_inp
        );
        start_inp += per_input;
    }
    cudaDeviceSynchronize();

    return grad_inp;
}