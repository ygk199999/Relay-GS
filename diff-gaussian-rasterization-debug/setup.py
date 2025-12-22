from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    # setup의 name은 패키지 이름이므로 그대로 둡니다.
    name="diff_gaussian_rasterization",
    ext_modules=[
        CUDAExtension(
            # ★★★ 이 부분을 수정하여 단일 모듈을 생성하도록 합니다. ★★★
            name="diff_gaussian_rasterization",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "ext.cpp"],
            # rasterize_points.cu의 내용은 ext.cpp로 옮겼으므로 소스 리스트에서 제거합니다.
            extra_compile_args={"cxx":['-g'], "nvcc": ["-g", "-std=c++17", "--extended-lambda", "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
