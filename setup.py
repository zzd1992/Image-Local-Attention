from torch import cuda
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import setup

sources = ['src/similar.cu',
           'src/weighting.cu',
           'localAttention.cpp']

if __name__ == '__main__':
    assert cuda.is_available(), 'Please install CUDA for GPU support.'
    extra_compile_args = {"nvcc": [
        "-DCUDA_HAS_FP16=1",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__"],
        "cxx": []
    }
    setup(
        name='localAttention',
        ext_modules=[CUDAExtension('localAttention', sources,
                                   extra_compile_args = extra_compile_args),
                     ],
        cmdclass={'build_ext': BuildExtension},

    )