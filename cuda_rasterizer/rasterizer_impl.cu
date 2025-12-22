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

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <utility> // for std::pair

#include <thrust/device_ptr.h>
#include <thrust/count.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"
#include "rasterizer.h"
// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	float3 orig_point={orig_points[3*idx],orig_points[3*idx+1],orig_points[3*idx+2]};
	present[idx] = in_frustum(orig_point, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
std::pair<int, int>  CudaRasterizer::Rasterizer::forward(
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
	int* radii,
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	CHECK_CUDA(FORWARD::preprocess(
		P, D, D_t, M,
		means3D,
		out_means3D,
		ts,
		(glm::vec3*)scales,
		scales_t,
		scale_modifier,
		(glm::vec4*)rotations,
		(glm::vec4*)rotations_r,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		timestamp,
		time_duration,
		rot_4d, gaussian_dim, force_sh_3d,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered
	), debug);

	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug);
	
	int num_rendered_overlaps;
	CHECK_CUDA(cudaMemcpy(&num_rendered_overlaps, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);
    
	thrust::device_ptr<int> dev_radii_ptr(radii);
    int num_gaussians_in_view = thrust::count_if(thrust::device, dev_radii_ptr, dev_radii_ptr + P, [] __device__ (int x) { return x > 0; });
	
	if(num_rendered_overlaps == 0)
    {
        CHECK_CUDA(cudaMemset(out_T, 0, width * height * sizeof(float)), debug);
        return {0, num_gaussians_in_view};
    }

	size_t binning_chunk_size = required<BinningState>(num_rendered_overlaps);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered_overlaps);

	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid);
	CHECK_CUDA(, debug);

	if (debug && num_rendered_overlaps > 0)
	{
		std::vector<uint64_t> host_keys_unsorted(num_rendered_overlaps);
		std::vector<uint32_t> host_values_unsorted(num_rendered_overlaps);
		cudaMemcpy(host_keys_unsorted.data(), binningState.point_list_keys_unsorted, num_rendered_overlaps * sizeof(uint64_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(host_values_unsorted.data(), binningState.point_list_unsorted, num_rendered_overlaps * sizeof(uint32_t), cudaMemcpyDeviceToHost);
		
		std::ofstream outfile_unsorted("unsorted_tile_data.txt");
		if (outfile_unsorted.is_open())
		{
			for (int i = 0; i < num_rendered_overlaps; ++i)
			{
				uint64_t key = host_keys_unsorted[i];
				uint32_t tile_id = key >> 32;
				uint32_t depth_bits = key & 0xFFFFFFFF;
				float depth_value = *((float*)&depth_bits);
				uint32_t gaussian_id = host_values_unsorted[i];
				outfile_unsorted << i << "," << tile_id << "," << gaussian_id << "," << depth_value << "\n";
			}
			outfile_unsorted.close();
		}
	}

	int bit = 32;

	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered_overlaps, 0, 32 + bit), debug);

	if (debug && num_rendered_overlaps > 0)
	{
		std::vector<uint64_t> host_keys(num_rendered_overlaps);
		std::vector<uint32_t> host_values(num_rendered_overlaps);
		cudaMemcpy(host_keys.data(), binningState.point_list_keys, num_rendered_overlaps * sizeof(uint64_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(host_values.data(), binningState.point_list, num_rendered_overlaps * sizeof(uint32_t), cudaMemcpyDeviceToHost);
		
		std::ofstream outfile_sorted("sorted_tile_data.txt");
		if (outfile_sorted.is_open())
		{
			for (int i = 0; i < num_rendered_overlaps; ++i)
			{
				uint64_t key = host_keys[i];
				uint32_t tile_id = key >> 32;
				uint32_t depth_bits = key & 0xFFFFFFFF;
				float depth_value = *((float*)&depth_bits);
				uint32_t gaussian_id = host_values[i];
				outfile_sorted << i << "," << tile_id << "," << gaussian_id << "," << depth_value << "\n";
			}
			outfile_sorted.close();
		}
	}

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	if (num_rendered_overlaps > 0)
		identifyTileRanges << <(num_rendered_overlaps + 255) / 256, 256 >> > (
			num_rendered_overlaps,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	const float* flow_ptr = flows_precomp;
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		flow_ptr,
		geomState.depths,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color,
		out_flow,
		out_depth), debug);

	if (debug)
	{
		std::vector<uint32_t> host_contrib(width * height);
		cudaMemcpy(host_contrib.data(), imgState.n_contrib, width * height * sizeof(uint32_t), cudaMemcpyDeviceToHost);

		std::ofstream outfile_contrib("pixel_contrib_counts.txt");
		if (outfile_contrib.is_open())
		{
			for (int y = 0; y < height; ++y)
			{
				for (int x = 0; x < width; ++x)
				{
					uint32_t count = host_contrib[y * width + x];
					outfile_contrib << x << "," << y << "," << count << "\n";
				}
			}
			outfile_contrib.close();
		}
	}

	CHECK_CUDA(cudaMemcpy(out_T, imgState.accum_alpha, width * height * sizeof(float), cudaMemcpyDeviceToDevice), debug);
	
	return {num_rendered_overlaps, num_gaussians_in_view};
}

std::tuple<int, int, torch::Tensor, torch::Tensor, torch::Tensor> CudaRasterizer::Rasterizer::preprocess_and_bin(
	std::function<char* (size_t)> geometryBuffer,
	const int P, int D, int D_t, int M,
	const int width, int height,
	const float* means3D,
	float* out_means3D,
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
	int*& radii,
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	CHECK_CUDA(FORWARD::preprocess(
		P, D, D_t, M,
		means3D,
		out_means3D,
		ts,
		(glm::vec3*)scales,
		scales_t,
		scale_modifier,
		(glm::vec4*)rotations,
		(glm::vec4*)rotations_r,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		timestamp,
		time_duration,
		rot_4d, gaussian_dim, force_sh_3d,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered
	), debug);

	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug);

	int num_rendered_overlaps;
	CHECK_CUDA(cudaMemcpy(&num_rendered_overlaps, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);
    
    thrust::device_ptr<int> dev_radii_ptr(radii);
    int num_gaussians_in_view = thrust::count_if(thrust::device, dev_radii_ptr, dev_radii_ptr + P, [] __device__ (int x) { return x > 0; });

    if(num_rendered_overlaps == 0)
    {
        auto options_long = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
        auto options_int = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
		auto options_float = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
        return std::make_tuple(0, num_gaussians_in_view, torch::empty({0}, options_long), torch::empty({0}, options_int), torch::empty({0}, options_float));
    }

    auto options_long = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto options_int = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    torch::Tensor keys_unsorted = torch::empty({num_rendered_overlaps}, options_long);
    torch::Tensor values_unsorted = torch::empty({num_rendered_overlaps}, options_int);

	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P, geomState.means2D, geomState.depths, geomState.point_offsets,
		(uint64_t*)keys_unsorted.data_ptr(), (uint32_t*)values_unsorted.data_ptr(),
		radii, tile_grid);
	CHECK_CUDA(, debug);

	auto options_float = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    auto depths_tensor = torch::from_blob(geomState.depths, {P}, options_float).clone();

	return std::make_tuple(num_rendered_overlaps, num_gaussians_in_view, keys_unsorted, values_unsorted, depths_tensor);
}

void CudaRasterizer::Rasterizer::render_sorted(
	char* geom_buffer_ptr,
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
	bool debug)
{
	char* chunkptr = geom_buffer_ptr;
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	const float* flow_ptr = flows_precomp;
	
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		(const uint2*)ranges.data_ptr(),
		(const uint32_t*)point_list.data_ptr(),
		width, height,
		geomState.means2D,
		feature_ptr,
		flow_ptr,
		geomState.depths,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color,
		out_flow,
		out_depth), debug);

	CHECK_CUDA(cudaMemcpy(out_T, imgState.accum_alpha, width * height * sizeof(float), cudaMemcpyDeviceToDevice), debug);
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
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
	char* img_buffer,
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
	bool debug)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	const float* depth_ptr = geomState.depths;
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		depth_ptr,
		flows_2d,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		dL_depths,
		dL_masks,
		dL_dpix_flow,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor, dL_dflows), debug);

	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, D, D_t, M,
		(float3*)out_means3D,
		radii,
		shs,
		ts,
		opacities,
		geomState.clamped,
		geomState.tiles_touched,
		(glm::vec3*)scales,
		scales_t,
		(glm::vec4*)rotations,
		(glm::vec4*)rotations_r,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		timestamp,
		time_duration,
		rot_4d, gaussian_dim, force_sh_3d,
		(float3*)dL_dmean2D,
		dL_dconic,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh, dL_dts,
		(glm::vec3*)dL_dscale,
		dL_dscale_t,
		(glm::vec4*)dL_drot,
		(glm::vec4*)dL_drot_r,
		dL_dopacity), debug);
}