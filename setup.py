import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = os.path.dirname(os.path.abspath(__file__))
kernels_dir = os.path.join(this_dir, "kernels")

setup(
    name="fastcv",
    ext_modules=[
        CUDAExtension(
            name="fastcv",
            sources=[
                "kernels/grayscale.cu",
                "kernels/box_blur.cu",
                "kernels/sobel.cu",
                "kernels/dilation.cu",
                "kernels/erosion.cu",
                "kernels/module.cpp",
                "kernels/medianBlur.cu"
            ],
            include_dirs = [
                kernels_dir
            ],
            extra_compile_args={"cxx": ["-O2"],
                                'nvcc': ["--expt-extended-lambda"]},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
