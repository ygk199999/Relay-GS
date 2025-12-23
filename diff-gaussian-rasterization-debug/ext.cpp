/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include "rasterize_points.h"

// PYBIND11_MODULE 매크로를 사용하여 Python 모듈을 정의합니다.
// TORCH_EXTENSION_NAME은 컴파일 시 setup.py에 의해 정의됩니다.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // ====================================================================
  //                     수정된 함수 바인딩
  // ====================================================================

  // 1. 바닐라(기본) 렌더링 함수를 Python에 'rasterize_gaussians_vanilla' 라는 이름으로 노출시킵니다.
  m.def("rasterize_gaussians_vanilla", &RasterizeGaussiansVanillaCUDA, "Vanilla forward rasterization");
  
  // 2. Python 정렬을 위한 전처리 단계 함수를 'preprocess_gaussians' 라는 이름으로 노출시킵니다.
  m.def("preprocess_gaussians", &PreprocessGaussiansCUDA, "Preprocess step for Python sorting");

  // 3. Python에서 정렬된 데이터로 렌더링하는 함수를 'render_sorted_gaussians' 라는 이름으로 노출시킵니다.
  m.def("render_sorted_gaussians", &RenderSortedGaussiansCUDA, "Render step with sorted data from Python");
  
  // ====================================================================
  //                     기존 함수 바인딩 (수정 없음)
  // ====================================================================
  
  // 역방향 전파 함수를 'rasterize_gaussians_backward' 라는 이름으로 노출시킵니다.
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA, "Backward pass for rasterization");

  // 뷰 프러스텀 컬링 함수를 'mark_visible' 이라는 이름으로 노출시킵니다.
  m.def("mark_visible", &markVisible, "Mark visible Gaussians");
}