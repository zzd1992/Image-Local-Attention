#include "utils.cuh"
#include <math.h>

template <typename dt, typename dtc>
__global__ void cc2k(const dt *x_ori, const dt *x_loc, const int kH,
                     const int kW, const int rH, const int rW, const int patch,
                     const int channels, const int height, const int width,
                     const int per_channel, dt *y) {
  // x_ori, x_loc: {b, c, h, w}
  // y: {b, h, w, k^2}
  const int batch_offset = blockIdx.y * channels * per_channel;
  const int batch_offset_out = blockIdx.y * per_channel * patch;
  // Block: One thread per pixel in the patch (kH * kW)
  // Each block computes the result for one pixel
  for (int indexO = blockIdx.x; indexO < per_channel; indexO += gridDim.x) {
    // offsets of the output pixel (has no channels)
    const int w_ori = indexO % width - rW;
    const int h_ori = indexO / width - rH;

    // Each thread computes the result for its pixel in the block
    // where the block spans the kernel size (patch).
    KERNEL_LOOP(indexK, patch) {
      // offsets of input pixels in the kernel
      const int w = w_ori + indexK % kW;
      const int h = h_ori + indexK / kW;
      dtc val{0};

      // coordinates can be out of bounds for some threads
      if (h > -1 && h < height && w > -1 && w < width) {
        const dt *p_ori =
            x_ori + batch_offset + indexO; // target pixel channel=0
        const dt *p_loc =
            x_loc + batch_offset + h * width + w; // patch pixel channel=0
        // Accumulate over channels
        for (int c = 0; c < channels; ++c) {
          val += static_cast<dtc>(__ldg(p_ori) * __ldg(p_loc));
          p_ori += per_channel;
          p_loc += per_channel;
        }
      }
      y[batch_offset_out + indexO * patch + indexK] = static_cast<dt>(val);
    }
  }
}

template <typename dt, typename dtc>
__global__ void ck2c_ori(const dt *x_loc, const dt *x_weight, const int kH,
                         const int kW, const int rH, const int rW,
                         const int patch, const int channels, const int height,
                         const int width, const int per_channel,
                         const int per_inp, dt *y) {
  // x_loc: {b, c, h, w}
  // x_weight: {b, h, w, k^2}
  // y: {b, c, h, w}
  const int batch_offset = blockIdx.y * channels * per_channel;
  const int batch_offset_w = blockIdx.y * per_channel * patch;
  // Each thread computes the result for one output in the block
  KERNEL_LOOP1d(index, per_inp) {
    const int index_ = index % per_channel; // spatial index (no channels)
    const int w_ori = index_ % width - rW;  // left side of kernel x
    const int h_ori = index_ / width - rH;  // top side of kernel y
    const dt *p_weight = x_weight + batch_offset_w +
                         index_ * patch; // start of kernel-patch in x_weight
    const dt *p_loc =
        x_loc + batch_offset + index - index_; // pixel address for channel=0
    dtc val{0};

    // accumulate over kernel size (k^2)
    for (int indexK = 0; indexK < patch; ++indexK) {
      const int w = w_ori + indexK % kW;  // index in kernel x
      const int h = h_ori + indexK / kW;  // index in kernel y
      // w_ori and h_ori can out of bounds
      if (h > -1 && h < height && w > -1 && w < width) {
        val += static_cast<dtc>(__ldg(p_loc + width * h + w) *
                                __ldg(p_weight + indexK));
      }
    }
    y[index + batch_offset] = static_cast<dt>(val);
  }
}

template <typename dt, typename dtc>
__global__ void ck2c_loc(const dt *x_ori, const dt *x_weight, const int kH,
                         const int kW, const int rH, const int rW,
                         const int patch, const int channels, const int height, const int width,
                         const int per_channel, const int per_inp, dt *y) {
  // x_ori: {b, c, h, w}
  // x_weight: {b, h, w, k^2}
  // y: {b, c, h, w}
  const int batch_offset =
      blockIdx.y * channels * width * height;
  const int batch_offset_w = blockIdx.y * height * width * patch;
  // Each thread computes the result for one output in the block
  KERNEL_LOOP1d(index, per_inp) {
    const int index_ = index % per_channel;  // spatial index (no channels)
    const int w_ori = index_ % width + rW;  // right side of kernel x
    const int h_ori = index_ / width + rH;  // bottom of kernel y
    const dt *p_ori = x_ori + batch_offset + index - index_;  // pixel address for channel=0
    dtc val{0};

    // accumulate over kernel size (k^2)
    for (int indexK = 0; indexK < patch; ++indexK) {
      const int w = w_ori - indexK % kW;  // index in kernel x
      const int h = h_ori - indexK / kW;  // index in kernel y
      const int indexW = width * h + w;  // linear index in kernel

      // w_ori and h_ori can be out of bounds
      if (h > -1 && h < height && w > -1 && w < width) {
        val += static_cast<dtc>(
            __ldg(p_ori + indexW) *
            __ldg(x_weight + batch_offset_w + indexW * patch + indexK));
      }
    }
    y[index + batch_offset] = static_cast<dt>(val);
  }
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

template <typename dt, typename dtc>
void f_cc2k(cudaStream_t stream, const dt *x_ori, const dt *x_loc, const int kH,
            const int kW, const int rH, const int rW, const int patch,
            const int channels, const int height, const int width,
            const int per_channel, const int batch, dt *y) {
  // total threads needed: patch * h * w 
  // grid: h * w
  // block_dim: min(max(32, patch), 1024)
  const int threads_per_block = WARP_SIZE * GET_BLOCKS(patch, WARP_SIZE);
  dim3 block_dim(min(threads_per_block, CUDA_NUM_THREADS));
  dim3 grid_dim(per_channel, batch);
  cc2k<dt, dtc><<<grid_dim, block_dim, 0, stream>>>(
      x_ori, x_loc, kH, kW, rH, rW, patch, channels, height, width, per_channel,
      y);
}

template <typename dt, typename dtc>
void f_ck2c_ori(cudaStream_t stream, const dt *x_loc, const dt *x_weight,
                const int kH, const int kW, const int rH, const int rW,
                const int patch, const int channels, const int height,
                const int width, const int per_channel, const int per_inp,
                const int batch, dt *y) {
  // total threads needed: c * h * w 
  // grid: h * w
  // block_dim: min(max(32, c), 1024)
  const int threads_per_block = WARP_SIZE * GET_BLOCKS(channels, WARP_SIZE);
  dim3 block_dim(min(threads_per_block, CUDA_NUM_THREADS));
  dim3 grid_dim(GET_BLOCKS(per_inp, block_dim.x), batch);
  ck2c_ori<dt, dtc><<<grid_dim, block_dim, 0, stream>>>(
      x_loc, x_weight, kH, kW, rH, rW, patch, channels, height, width,
      per_channel, per_inp, y);
}

template <typename dt, typename dtc>
void f_ck2c_loc(cudaStream_t stream, const dt *x_ori, const dt *x_weight,
                const int kH, const int kW, const int rH, const int rW,
                const int patch, const int channels, const int height,
                const int width, const int per_channel, const int per_inp,
                const int batch, dt *y) {
  // total threads needed: c * h * w 
  // grid: h * w
  // block_dim: min(max(32, c), 1024)
  const int threads_per_block = WARP_SIZE * GET_BLOCKS(channels, WARP_SIZE);
  dim3 block_dim(min(threads_per_block, CUDA_NUM_THREADS));
  dim3 grid_dim(GET_BLOCKS(per_inp, block_dim.x), batch);
  ck2c_loc<dt, dtc><<<grid_dim, block_dim, 0, stream>>>(
      x_ori, x_weight, kH, kW, rH, rW, patch, channels, height, width, per_channel,
      per_inp, y);
}
