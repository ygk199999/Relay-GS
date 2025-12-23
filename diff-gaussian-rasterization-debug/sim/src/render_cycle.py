#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
from scene import Scene
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import time
import pandas as pd

# ====================================================================================
# ======================== 새로운 CSV 포맷 저장을 위한 함수들 ========================
# ====================================================================================

def save_as_grouped_csv(df, output_path):
    """
    DataFrame을 'Tile_ID,Gaussian_Count,Gaussian_IDs' 형식의 CSV로 저장합니다. (Depth 정보 없음)
    """
    try:
        grouped = df.groupby('Tile_ID')['Gaussian_ID'].agg(list)
        with open(output_path, 'w') as f:
            f.write("Tile_ID,Gaussian_Count,Gaussian_IDs\n")
            for tile_id, gaussian_ids in grouped.items():
                count = len(gaussian_ids)
                ids_str = ' '.join(map(str, gaussian_ids))
                f.write(f'{tile_id},{count},"{ids_str}"\n')
    except Exception as e:
        print(f"\n[ERROR] Failed to save in grouped CSV format: {e}")

def save_as_grouped_csv_with_depth(df, output_path):
    """
    DataFrame을 'Tile_ID,Gaussian_Count,Gaussian_IDs,Depths' 형식의 CSV로 저장합니다.
    Python 정렬 시뮬레이션을 위해 Depth 정보를 보존합니다.
    """
    try:
        # Tile_ID로 그룹화하고, 각 타일의 Gaussian_ID와 Depth를 리스트로 집계합니다.
        grouped = df.groupby('Tile_ID').agg({
            'Gaussian_ID': list,
            'Depth': list
        })
        
        with open(output_path, 'w') as f:
            # 새로운 헤더
            f.write("Tile_ID,Gaussian_Count,Gaussian_IDs,Depths\n")
            for tile_id, data in grouped.iterrows():
                gaussian_ids = data['Gaussian_ID']
                depths = data['Depth']
                count = len(gaussian_ids)
                
                # ID와 Depth를 공백으로 구분된 문자열로 변환
                ids_str = ' '.join(map(str, gaussian_ids))
                depths_str = ' '.join(map(str, depths))
                
                # 파일에 쓰기
                f.write(f'{tile_id},{count},"{ids_str}","{depths_str}"\n')
    except Exception as e:
        print(f"\n[ERROR] Failed to save in grouped CSV format with depth: {e}")

def save_max_contrib_per_tile(df, output_path, image_width, image_height, tile_dim=8):
    """
    각 타일별 최대 기여자 수를 'Tile_ID,Max_Contributor_Count' 형식의 CSV로 저장합니다.
    """
    try:
        num_tiles_x = (image_width + tile_dim - 1) // tile_dim
        df['Tile_ID'] = (df['PixelY'] // tile_dim) * num_tiles_x + (df['PixelX'] // tile_dim)
        max_contrib_per_tile = df.groupby('Tile_ID')['ContributorCount'].max()
        max_contrib_per_tile.to_csv(output_path, header=['Max_Contributor_Count'], index_label='Tile_ID')
    except Exception as e:
        print(f"\n[ERROR] Failed to save max contributor count per tile: {e}")


# ====================================================================================
# ===================== 수정된 render_set 함수 (새로운 명칭 적용) =====================
# ====================================================================================

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    
    culling_log_file = None
    if pipeline.debug:
        # 🌟🌟🌟 이름 변경 🌟🌟🌟
        mode_name = "log_with_depth" if pipeline.log_with_depth else "cuda_sorting"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_base_path = os.path.join("..", f"{name}_{mode_name}_{timestamp}")
        print(f"\n[INFO] Debug logs for this run will be saved in: {log_base_path}")

        # 🌟🌟🌟 이름 변경 🌟🌟🌟
        # 로그 폴더 이름 접미사를 동적으로 설정
        log_suffix = "_with_depth" if pipeline.log_with_depth else "_grouped"
        unsorted_log_path = os.path.join(log_base_path, f"unsorted{log_suffix}")
        sorted_log_path = os.path.join(log_base_path, f"sorted{log_suffix}")
        contrib_log_path = os.path.join(log_base_path, "pixel_contrib_summary")
        
        # 필요한 모든 디렉토리 생성
        for path in [log_base_path, unsorted_log_path, sorted_log_path, contrib_log_path]:
            makedirs(path, exist_ok=True)
            
        # 🌟🌟🌟 이름 변경 🌟🌟🌟
        # 사용자에게 현재 모드와 로그 형식을 명확히 안내
        if pipeline.log_with_depth:
            print("\n[INFO] Log with Depth mode: Logs will include depth information for simulation.")
            print("\n[WARNING] C++ backend logs (unsorted, sorted, contrib) will not be generated in this mode.")
        else:
            print("\n[INFO] CUDA sorting mode (Standard): Logs will not include depth information.")

        # C++ 백엔드가 생성할 임시 파일 이름 정의
        source_file_unsorted = "unsorted_tile_data.txt"
        source_file_sorted = "sorted_tile_data.txt"
        source_file_contrib = "pixel_contrib_counts.txt"
        
        # 이전 실행에서 남은 임시 파일이 있다면 삭제
        for f in [source_file_unsorted, source_file_sorted, source_file_contrib]:
            if os.path.exists(f):
                os.remove(f)
        
        # Culling 로그 파일 경로 설정 및 파일 열기
        culling_log_path = os.path.join(log_base_path, "culling_log.csv")
        culling_log_file = open(culling_log_path, 'w')
        culling_log_file.write("View_Index,Gaussians_In_View,Total_Gaussians\n")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    total_gaussians = len(gaussians.get_xyz)
    progress_bar = tqdm(views, desc="Rendering progress")
    for idx, view in enumerate(progress_bar):
        outputs = render(view[1].cuda(), gaussians, pipeline, background)
        rendering = outputs["render"]
        
        num_in_view = outputs.get("num_gaussians_in_view", 0)
        progress_bar.set_description(f"Render progress (View {idx+1}/{len(views)} | Visible: {num_in_view}/{total_gaussians})")

        if pipeline.debug:
            if culling_log_file:
                culling_log_file.write(f"{idx},{num_in_view},{total_gaussians}\n")
                culling_log_file.flush()

            # unsorted 로그 처리
            if os.path.exists(source_file_unsorted):
                df_unsorted = pd.read_csv(source_file_unsorted, header=None, names=['Instance_Index', 'Tile_ID', 'Gaussian_ID', 'Depth'])
                dest_path_csv = os.path.join(unsorted_log_path, f"unsorted_{idx:05d}.csv")
                
                # 🌟🌟🌟 이름 변경 🌟🌟🌟
                if pipeline.log_with_depth:
                    save_as_grouped_csv_with_depth(df_unsorted, dest_path_csv)
                else:
                    save_as_grouped_csv(df_unsorted, dest_path_csv)
                os.remove(source_file_unsorted)

            # sorted 로그 처리
            if os.path.exists(source_file_sorted):
                df_sorted = pd.read_csv(source_file_sorted, header=None, names=['Instance_Index', 'Tile_ID', 'Gaussian_ID', 'Depth'])
                dest_path_csv = os.path.join(sorted_log_path, f"sorted_{idx:05d}.csv")

                # 🌟🌟🌟 이름 변경 🌟🌟🌟
                if pipeline.log_with_depth:
                    save_as_grouped_csv_with_depth(df_sorted, dest_path_csv)
                else:
                    save_as_grouped_csv(df_sorted, dest_path_csv)
                os.remove(source_file_sorted)

            # contributor 로그 처리
            if os.path.exists(source_file_contrib):
                df_contrib = pd.read_csv(source_file_contrib, header=None, names=['PixelX', 'PixelY', 'ContributorCount'])
                dest_path_csv = os.path.join(contrib_log_path, f"contrib_summary_{idx:05d}.csv")
                save_max_contrib_per_tile(df_contrib, dest_path_csv, view[1].image_width, view[1].image_height)
                os.remove(source_file_contrib)
            
            # 🌟🌟🌟 이름 변경 🌟🌟🌟
            # C++ 로그 파일 존재 여부 확인 (Python 정렬 모드 제외)
            if idx == 0 and not pipeline.log_with_depth:
                is_file_found = 'df_unsorted' in locals() or 'df_sorted' in locals() or 'df_contrib' in locals()
                if not is_file_found:
                    print(f"\n[DEBUG] C++ output files not found. Ensure the custom C++ backend is compiled and running correctly.")

        gt = view[0][0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
    
    if culling_log_file:
        culling_log_file.close()

# ====================================================================================
# ============================ 이하 코드는 변경 없음 =================================
# ====================================================================================
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, gaussian_dim=4, rot_4d=True)
        scene = Scene(dataset, gaussians, shuffle=False)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)