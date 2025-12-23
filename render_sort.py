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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, reset_sort_state
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import time
import pandas as pd
import numpy as np

def save_python_debug_data_as_npz(outputs, dest_path):
    """ (신규) Python 파이프라인에서 반환된 텐서를 직접 .npz 파일로 저장합니다. """
    try:
        keys = outputs.get("unsorted_keys")
        values = outputs.get("unsorted_values")
        depths_tensor = outputs.get("depths")

        if keys is None or values is None or depths_tensor is None or keys.numel() == 0:
            print(f"[DEBUG] No data to save for this frame.")
            return

        tile_ids = (keys >> 32).cpu().numpy()
        gauss_ids = values.cpu().numpy()
        
        # 모든 가우시안의 깊이를 한 번에 가져와서 인덱싱
        all_depths = depths_tensor.cpu().numpy()
        gauss_depths = all_depths[gauss_ids]

        # 고유한 타일 ID 찾기
        unique_tiles = np.unique(tile_ids)
        
        # 각 타일 ID에 해당하는 데이터를 딕셔너리에 저장
        data_to_save = {}
        for tile_id in unique_tiles:
            mask = (tile_ids == tile_id)
            # [Gaussian_ID, Depth] 형태의 2열 배열로 저장
            data_to_save[str(tile_id)] = np.stack((gauss_ids[mask], gauss_depths[mask]), axis=-1).astype(np.float32)
        
        np.savez_compressed(dest_path, **data_to_save)
        print(f"\n[DEBUG] Saved unsorted data from Python pipeline to {dest_path}")

    except Exception as e:
        print(f"\n[ERROR] Failed to save Python debug data as .npz: {e}")

def save_cpp_debug_data_as_npz(source_file, dest_path):
    """ (신규) C++ 파이프라인의 임시 .txt 파일을 .npz로 변환하여 저장합니다. """
    try:
        csv_header = ['Instance_Index', 'Tile_ID', 'Gaussian_ID', 'Depth']
        df = pd.read_csv(source_file, header=None, names=csv_header)
        
        data_to_save = {}
        for tile_id, group_df in df.groupby('Tile_ID'):
            array = group_df[['Gaussian_ID', 'Depth']].to_numpy(dtype=np.float32)
            data_to_save[str(tile_id)] = array
        np.savez_compressed(dest_path, **data_to_save)
        print(f"\n[DEBUG] Converted C++ debug data and saved to {dest_path}")

    except Exception as e:
        print(f"\n[ERROR] Failed to convert C++ debug data to .npz: {e}")


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    # --- 렌더링 결과 저장 폴더 이름 동적 생성 ---
    if pipeline.python_sorting:
        interval = pipeline.resort_interval
        render_folder_name = f"renders_incremental_resort_{interval}" if interval > 0 else "renders_incremental_pure"
    else:
        render_folder_name = "renders_vanilla"
        
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), render_folder_name)
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    
    # 디버깅 경로 설정 (모두 .npz를 저장하도록 통일)
    debug_output_path = None
    if pipeline.debug:
        if pipeline.python_sorting:
            debug_output_path = os.path.join(model_path, "renders_npz_python", name)
        else:
            debug_output_path = os.path.join(model_path, "renders_npz_cpp", name)
        makedirs(debug_output_path, exist_ok=True)

    # C++ 디버그용 임시 파일 (필요한 경우에만 사용)
    source_file_unsorted = "unsorted_tile_data.txt"
    if os.path.exists(source_file_unsorted): os.remove(source_file_unsorted)

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        view_cam = view[1].cuda()
        
        force_resort_flag = False
        if pipeline.python_sorting and pipeline.resort_interval > 0 and idx % pipeline.resort_interval == 0:
            force_resort_flag = True
        print("-" * 50)
        print(f"[DEBUG_FLAG_CHECK] pipeline.python_sorting: {pipeline.python_sorting}")
        print(f"[DEBUG_FLAG_CHECK] force_resort_flag: {force_resort_flag}")
        print("-" * 50)
        outputs = render(view_cam, gaussians, pipeline, background, force_resort=force_resort_flag)
        
        # --- 디버그 데이터 저장 로직 ---
        if pipeline.debug:
            # Python 정렬 파이프라인의 디버깅 출력
            if pipeline.python_sorting:
                dest_path = os.path.join(debug_output_path, f"unsorted_data_{idx:05d}.npz")
                save_python_debug_data_as_npz(outputs, dest_path)
            
            # C++ 정렬 파이프라인의 디버깅 출력
            else:
                wait_time = 0
                while not os.path.exists(source_file_unsorted) and wait_time < 3.0:
                    time.sleep(0.01)
                    wait_time += 0.01
                
                if os.path.exists(source_file_unsorted):
                    dest_path = os.path.join(debug_output_path, f"unsorted_data_{idx:05d}.npz")
                    print(f"[DEBUG CHECK] About to save NPZ to: {dest_path}")
                    save_cpp_debug_data_as_npz(source_file_unsorted, dest_path)
                    os.remove(source_file_unsorted)
                elif idx == 0:
                    print(f"\n[DEBUG] C++ output file not found. Ensure rasterizer's debug flag is on.")

        # 렌더링 및 GT 이미지 저장
        torchvision.utils.save_image(outputs["render"], os.path.join(render_path, f'{idx:05d}.png'))
        torchvision.utils.save_image(view[0], os.path.join(gts_path, f'{idx:05d}.png'))

# (render_sets 및 main 함수는 이전과 동일하게 유지)
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, gaussian_dim=4, rot_4d=True)
        
        if dataset.loaded_pth:
            print(f"Initializing scene for checkpoint loading: {dataset.loaded_pth}")
            scene = Scene(dataset, gaussians, shuffle=False)
        else:
            print(f"Initializing scene to load iteration: {iteration}")
            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
            
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        if not skip_train:
             reset_sort_state()
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
        
        if not skip_test:
             reset_sort_state()
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
    
    pipeline_options = pipeline.extract(args)
    model_options = model.extract(args)

    print("Rendering " + args.model_path)
    safe_state(args.quiet)
    render_sets(model_options, args.iteration, pipeline_options, args.skip_train, args.skip_test)
