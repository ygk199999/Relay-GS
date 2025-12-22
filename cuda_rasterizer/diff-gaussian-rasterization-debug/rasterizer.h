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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>
#include <tuple> 
#include <utility> // <<< std::pair를 사용하기 위해 헤더 추가
#include <torch/extension.h> 

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		// 함수 1: 바닐라(기본) 전체 렌더링 파이프라인
		// <<< 수정됨: 반환 타입을 int에서 std::pair<int, int>로 변경
		// std::pair<int, int> -> {총 작업량, 시야 내 가우시안 수}
		static std::pair<int, int> forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int D_t, int M,
			const float* background,
			const int width, int height,
			const float* means3D,
			float* out_means3D,
			const float* shs,
			const float* colors_precomp,
			const float* flows_precomp,
			const float* opacities,
			const float* ts,
			const float* scales,
			const float* scales_t,
			const float scale_modifier,
			const float* rotations,
			const float* rotations_r,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float timestamp,
			const float time_duration,
			const bool rot_4d, const int gaussian_dim, const bool force_sh_3d,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			float* out_color,
			float* out_flow,
			float* out_depth,
			float* out_T,
			int* radii = nullptr,
			bool debug = false);

		// 함수 2: Python 정렬을 위한 전처리 및 비닝 단계
		// <<< 수정됨: 반환 튜플에 int를 하나 더 추가
		// std::tuple<int, int, ...> -> {총 작업량, 시야 내 가우시안 수, ...}
		static std::tuple<int, int, torch::Tensor, torch::Tensor, torch::Tensor> preprocess_and_bin(
			std::function<char* (size_t)> geometryBuffer,
			const int P, int D, int D_t, int M,
			const int width, int height,
			const float* means3D,
			float* out_means3D, // Can be output
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* ts,
			const float* scales,
			const float* scales_t,
			const float scale_modifier,
			const float* rotations,
			const float* rotations_r,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float timestamp,
			const float time_duration,
			const bool rot_4d, const int gaussian_dim, const bool force_sh_3d,
			const float tan_fovx, const float tan_fovy,
			const bool prefiltered,
			int*& radii, // radii를 출력으로 받기 위해 참조(&)로 변경
			bool debug = false);
		
		// 함수 3: Python에서 정렬된 데이터로 렌더링하는 단계
		static void render_sorted(
			char* geom_buffer_ptr, // <<<--- 이 부분이 수정되었습니다.
			std::function<char* (size_t)> imageBuffer,
			const int P,
			const int width, int height,
			const float* background,
			const torch::Tensor& point_list,
			const torch::Tensor& ranges,
			const float* colors_precomp,
			const float* flows_precomp,
			float* out_color,
			float* out_flow,
			float* out_depth,
			float* out_T,
			bool debug = false);

		// 역방향 전파 함수
		static void backward(
			const int P, int D, int D_t, int M, int R,
			const float* background,
			const int width, int height,
			const float* out_means3D,
			const float* shs,
			const float* colors_precomp,
			const float* flows_2d,
			const float* opacities,
			const float* ts,
			const float* scales,
			const float* scales_t,
			const float scale_modifier,
			const float* rotations,
			const float* rotations_r,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* campos,
			const float timestamp,
			const float time_duration,
			const bool rot_4d, const int gaussian_dim, const bool force_sh_3d,
			const float tan_fovx, float tan_fovy,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			const float* dL_dpix,
			const float* dL_depths,
			const float* dL_masks,
			const float* dL_dpix_flow,
			float* dL_dmean2D,
			float* dL_dconic,
			float* dL_dopacity,
			float* dL_dcolor,
			float* dL_dmean3D,
			float* dL_dcov3D,
			float* dL_dsh,
			float* dL_dflows,
			float* dL_dts,
			float* dL_dscale,
			float* dL_dscale_t,
			float* dL_drot,
			float* dL_drot_r,
			bool debug);
	};
};

#endif
