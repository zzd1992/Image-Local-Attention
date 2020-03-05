#include "utils.cuh"
#include <math.h>

template <typename dt, typename dtc>
__global__ void weighting_forward_kernel(
        const dt* x_ori,
        const PTA32(dt, 3) x_weight,
        const int kH,
        const int kW,
        const int rH,
        const int rW,
        const int patch,
        const int height,
        const int width,
        const int per_channel,
        const int per_inp,
        dt* y
) {
    // x_ori: {c, h, w}
    // x_weight: {h, w, k^2}
    // y: {c, h, w}
    KERNEL_LOOP1d(index, per_inp) {
        const int index_ = index % per_channel;
        int w_ori = index_ % width;
        int h_ori = index_ / width;

        const auto p_x_weight = x_weight[h_ori][w_ori];
        const dt* p_x_ori = x_ori + index - index_;

        w_ori -= rW;
        h_ori -= rH;
        dtc val = dtc(0);
        for (int indexK=0; indexK<patch; ++indexK) {
            const int w = w_ori + indexK % kW;
            const int h = h_ori + indexK / kW;
            if (h > -1 && h < height && w > -1 && w < width) {
                val += static_cast<dtc> (p_x_ori[width * h + w] * p_x_weight[indexK]);
            }
        }
        y[index] = static_cast<dt> (val);
    }
}

template <typename dt, typename dtc>
void weighting_forward_warper(
        cudaStream_t stream,
        const dt* x_ori,
        const PTA32(dt, 3) x_weight,
        const int kH,
        const int kW ,
        const int rH,
        const int rW,
        const int patch,
        const int height,
        const int width,
        const int per_channel,
        const int per_inp,
        dt* y) {
    const int blockX = GET_BLOCKS(per_inp);

    weighting_forward_kernel<dt, dtc> <<< blockX, min(per_inp, CUDA_NUM_THREADS), 0, stream >>> (
            x_ori, x_weight,
            kH, kW, rH, rW,
            patch,
            height, width,
            per_channel, per_inp,
            y);
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

template <typename dt, typename dtc>
__global__ void weighting_backward_ori_kernel(
        const PTA32(dt, 3) x_weight,
        const PTA32(dt, 3) grad_out,
        const int kH,
        const int kW,
        const int rH,
        const int rW,
        const int patch,
        const int height,
        const int width,
        const int per_channel,
        const int per_inp,
        dt* grad_inp
) {
    // x_weight: {h, w, k^2}
    // grad_out: {c, h, w} memory is not continuous
    // grad_inp: {c, h, w}
    KERNEL_LOOP1d(index, per_inp) {
        const int c = index / per_channel;
        const int index_ = index % per_channel;
        const int w_ori = index_ % width + rW;
        const int h_ori = index_ / width + rH;

        const auto p_grad_out = grad_out[c];

        dtc val = dtc(0);
        for (int indexK=0; indexK<patch; ++indexK) {
            const int w = w_ori - indexK % kW;
            const int h = h_ori - indexK / kW;
            if (h > -1 && h < height && w > -1 && w < width) {
                val += static_cast<dtc> (x_weight[h][w][indexK] * p_grad_out[h][w]);
            }
        }
        grad_inp[index] = static_cast<dt> (val);
    }
}

template <typename dt, typename dtc>
void weighting_backward_ori_warper(
        cudaStream_t stream,
        const PTA32(dt, 3) x_weight,
        const PTA32(dt, 3) grad_out,
        const int kH,
        const int kW ,
        const int rH,
        const int rW,
        const int patch,
        const int height,
        const int width,
        const int per_channel,
        const int per_inp,
        dt* grad_inp) {
    const int blockX = GET_BLOCKS(per_inp);
    weighting_backward_ori_kernel<dt, dtc> <<< blockX, min(per_inp, CUDA_NUM_THREADS), 0, stream >>> (
            x_weight, grad_out,
            kH, kW, rH, rW,
            patch,
            height, width,
            per_channel, per_inp,
            grad_inp);
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

template <typename dt, typename dtc>
__global__ void weighting_backward_weight_kernel(
        const dt* x_ori,
        const dt* grad_out,
        const int kH,
        const int kW,
        const int rH,
        const int rW,
        const int patch,
        const int channels,
        const int height,
        const int width,
        const int per_channel,
        dt* grad_inp
) {
    // x_ori: {c, h, w}
    // grad_out: {c, h, w}
    // grad_inp: {h, w, k^2}
    for (int indexO=blockIdx.x; indexO<per_channel; indexO+=gridDim.x) {
        const int w_ori = indexO % width;
        const int h_ori = indexO / width;

        KERNEL_LOOP(indexK, patch) {
            const int w = w_ori - rW + indexK % kW;
            const int h = h_ori - rH + indexK / kW;
            const int indexL = width * h + w;

            dtc val = dtc(0);

            if (h > -1 && h < height && w > -1 && w < width) {
                const dt* p_ori = x_ori + indexL;
                const dt* p_grad_out = grad_out + indexO;
                for (int c=0; c<channels; ++c) {
                    val += static_cast<dtc> ((*p_ori) * (*p_grad_out));
                    p_ori += per_channel;
                    p_grad_out += per_channel;
                }
            }
            grad_inp[indexO * patch + indexK] = static_cast<dt> (val);
        }
    }
}

template <typename dt, typename dtc>
void weighting_backward_weight_warper(
        cudaStream_t stream,
        const dt* x_weight,
        const dt* grad_out,
        const int kH,
        const int kW ,
        const int rH,
        const int rW,
        const int patch,
        const int channels,
        const int height,
        const int width,
        const int per_channel,
        dt* grad_inp) {
    weighting_backward_weight_kernel<dt, dtc> <<< min(per_channel, 262144), min(patch, CUDA_NUM_THREADS), 0, stream >>> (
            x_weight, grad_out,
            kH, kW, rH, rW,
            patch, channels,
            height, width, per_channel,
            grad_inp);
}