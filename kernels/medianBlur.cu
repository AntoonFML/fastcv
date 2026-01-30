#define NOMINMAX

#include <iostream>
#include <vector>
#include <algorithm>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/tuple.h>

constexpr int block_size = 16;
using pixel_type = unsigned char;


template <typename T>
__device__ void insertion_sort(T* data, int n){
    for(int i = 1; i < n; i++){
        T key = data[i];
        int j = i - 1;
        while(j >= 0 && data[j] > key){
            data[j+1] = data[j];
            j = j - 1;
        }
        data[j+1] = key;
    }
}

__device__ __forceinline__ pixel_type get_pixel_clamped(const pixel_type* img, int x, int y, int width, int height){
    x = max(0, min(x, width - 1));
    y = max(0, min(y, height - 1));
    return img[y * width + x]; 
}

template <int radius>
__global__ void median_blur_single_channel(const pixel_type* __restrict__ input, 
                                           pixel_type* __restrict__ output, 
                                           int width, int height){
    constexpr int tile_dim = block_size + 2 * radius;
    constexpr int window_size = (2 * radius + 1) * (2 * radius + 1);

    __shared__ pixel_type shared_mem[tile_dim][tile_dim];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int global_x = blockIdx.x * blockDim.x + tx;
    const int global_y = blockIdx.y * blockDim.y + ty;

    const int tile_start_x = blockIdx.x * blockDim.x - radius;
    const int tile_start_y = blockIdx.y * blockDim.y - radius;

    for (int y = ty; y < tile_dim; y += blockDim.y) {
        for (int x = tx; x < tile_dim; x += blockDim.x) {
            int load_x = tile_start_x + x;
            int load_y = tile_start_y + y;
            shared_mem[y][x] = get_pixel_clamped(input, load_x, load_y, width, height);
        }
    }

    __syncthreads();

    if (global_x < width && global_y < height) {
        pixel_type window[window_size];
        const int shared_mem_x = tx + radius;
        const int shared_mem_y = ty + radius;
        
        int k = 0;
        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                window[k++] = shared_mem[shared_mem_y + dy][shared_mem_x + dx];
            }
        }
        insertion_sort(window, window_size);
        output[global_y * width + global_x] = window[window_size / 2];
    }
}

torch::Tensor medianBlur(torch::Tensor input, int k_size) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kUInt8, "Input must be uint8");
    input = input.contiguous();
    
    int h = input.size(0);
    int w = input.size(1);
    int num_pixels = h * w;

    auto output = torch::empty_like(input);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    uchar3* d_in_rgb = (uchar3*)input.data_ptr<pixel_type>();
    uchar3* d_out_rgb = (uchar3*)output.data_ptr<pixel_type>();

    thrust::device_vector<pixel_type> plane_r(num_pixels);
    thrust::device_vector<pixel_type> plane_g(num_pixels);
    thrust::device_vector<pixel_type> plane_b(num_pixels);

    thrust::transform(
        thrust::cuda::par.on(stream),
        thrust::device_pointer_cast(d_in_rgb),
        thrust::device_pointer_cast(d_in_rgb + num_pixels),
        thrust::make_zip_iterator(thrust::make_tuple(plane_r.begin(), plane_g.begin(), plane_b.begin())),
        [] __device__ (const uchar3& p) {
            return thrust::make_tuple(p.x, p.y, p.z);
        }
    );


    dim3 block(block_size, block_size);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    pixel_type* ptr_r = thrust::raw_pointer_cast(plane_r.data());
    pixel_type* ptr_g = thrust::raw_pointer_cast(plane_g.data());
    pixel_type* ptr_b = thrust::raw_pointer_cast(plane_b.data());

    if (k_size == 3) {
        median_blur_single_channel<1><<<grid, block, 0, stream>>>(ptr_r, ptr_r, w, h);
        median_blur_single_channel<1><<<grid, block, 0, stream>>>(ptr_g, ptr_g, w, h);
        median_blur_single_channel<1><<<grid, block, 0, stream>>>(ptr_b, ptr_b, w, h);
    } else if (k_size == 5) {
        median_blur_single_channel<2><<<grid, block, 0, stream>>>(ptr_r, ptr_r, w, h);
        median_blur_single_channel<2><<<grid, block, 0, stream>>>(ptr_g, ptr_g, w, h);
        median_blur_single_channel<2><<<grid, block, 0, stream>>>(ptr_b, ptr_b, w, h);
    } else if (k_size == 7) {
        median_blur_single_channel<3><<<grid, block, 0, stream>>>(ptr_r, ptr_r, w, h);
        median_blur_single_channel<3><<<grid, block, 0, stream>>>(ptr_g, ptr_g, w, h);
        median_blur_single_channel<3><<<grid, block, 0, stream>>>(ptr_b, ptr_b, w, h);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    thrust::transform(
        thrust::cuda::par.on(stream),
        thrust::make_zip_iterator(thrust::make_tuple(plane_r.begin(), plane_g.begin(), plane_b.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(plane_r.end(), plane_g.end(), plane_b.end())),
        thrust::device_pointer_cast(d_out_rgb),
        [] __device__ (const thrust::tuple<pixel_type, pixel_type, pixel_type>& t) { // <--- LAMBDA
            return make_uchar3(thrust::get<0>(t), thrust::get<1>(t), thrust::get<2>(t));
        }
    );

    return output;
}