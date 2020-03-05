#include "utils.cuh"
#include <math.h>

template <typename dt, typename dtc>
__global__ void similar_forward_kernel(
        const dt* x_ori,
        const dt* x_loc,
        const int kH,
        const int kW,
        const int rH,
        const int rW,
        const int patch,
        const int channels,
        const int height,
        const int width,
        const int per_channel,
        dt* y
) {
    // x_ori, x_loc: {c, h, w}
    // y: {h, w, k^2}
    for (int indexO=blockIdx.x; indexO<per_channel; indexO+=gridDim.x) {
        const int w_ori = indexO % width;
        const int h_ori = indexO / width;

        KERNEL_LOOP(indexK, patch) {
            const int w = w_ori - rW + indexK % kW;
            const int h = h_ori - rH + indexK / kW;
            const int indexL = width * h + w;

            dtc val = dtc(0);

            if (h > -1 && h < height && w > -1 && w < width) {
                const dt* p_ori = x_ori + indexO;
                const dt* p_loc = x_loc + indexL;
                for (int c=0; c<channels; ++c) {
                    val += static_cast<dtc> ((*p_ori) * (*p_loc));
                    p_ori += per_channel;
                    p_loc += per_channel;
                }
            }
            y[indexO * patch + indexK] = static_cast<dt> (val);
        }
    }
}

template <typename dt, typename dtc>
void similar_forward_warper(
        cudaStream_t stream,
        const dt* x_ori,
        const dt* x_loc,
        const int kH,
        const int kW ,
        const int rH,
        const int rW,
        const int patch,
        const int channels,
        const int height,
        const int width,
        dt* y) {
    const int per_channel = height * width;
    similar_forward_kernel<dt, dtc> <<< min(per_channel, 262144), min(patch, CUDA_NUM_THREADS), 0, stream >>> (
            x_ori, x_loc,
            kH, kW, rH, rW,
            patch, channels,
            height, width, per_channel,
            y);
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

template <typename dt, typename dtc>
__global__ void similar_backward_ori_kernel(
        const dt* x,
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
    // x: {c, h, w}
    // grad_out: {h, w, k^2} memory is not continuous
    KERNEL_LOOP1d(index, per_inp) {
        const int index_ = index % per_channel;
        int w_ori = index_ % width;
        int h_ori = index_ / width;

        const auto p_grad_out = grad_out[h_ori][w_ori];
        const dt* p_x = x + index - index_;

        w_ori -= rW;
        h_ori -= rH;
        dtc val = dtc(0);
        for (int indexK=0; indexK<patch; ++indexK) {
            const int w = w_ori + indexK % kW;
            const int h = h_ori + indexK / kW;
            if (h > -1 && h < height && w > -1 && w < width) {
                val += static_cast<dtc> (p_x[width * h + w] * p_grad_out[indexK]);
            }
        }
        grad_inp[index] = static_cast<dt> (val);
    }
}

template <typename dt, typename dtc>
__global__ void similar_backward_loc_kernel(
        const dt* x,
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
    // x: {c, h, w}
    // grad_out: {h, w, k^2} memory is not continuous
    KERNEL_LOOP1d(index, per_inp) {
        const int index_ = index % per_channel;
        const int w_ori = index_ % width + rW;
        const int h_ori = index_ / width + rH;

        const dt* p_x = x + index - index_;

        dtc val = dtc(0);
        for (int indexK=0; indexK<patch; ++indexK) {
            const int w = w_ori - indexK % kW;
            const int h = h_ori - indexK / kW;
            if (h > -1 && h < height && w > -1 && w < width) {
                val += static_cast<dtc> (p_x[width * h + w] * grad_out[h][w][indexK]);
            }
        }
        grad_inp[index] = static_cast<dt> (val);
    }
}

template <typename dt, typename dtc>
void similar_backward_warper(
        cudaStream_t stream,
        const dt* x,
        const PTA32(dt, 3) grad_out,
        const bool is_ori,
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
    if (is_ori) {
        similar_backward_ori_kernel<dt, dtc> <<< blockX, min(per_inp, CUDA_NUM_THREADS), 0, stream >>> (
                x, grad_out,
                kH, kW, rH, rW,
                patch,
                height, width,
                per_channel, per_inp,
                grad_inp);
    } else {
        similar_backward_loc_kernel<dt, dtc> <<< blockX, min(per_inp, CUDA_NUM_THREADS), 0, stream >>> (
                x, grad_out,
                kH, kW, rH, rW,
                patch,
                height, width,
                per_channel, per_inp,
                grad_inp);
    }
}
