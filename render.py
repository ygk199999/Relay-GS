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
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import shutil
import time
import pandas as pd
import numpy as np # NumPy 라이브러리 추가

def save_grouped_list_as_npz(df, dest_npz_path):
    """
    DataFrame을 입력받아, 그룹화된 데이터를 NumPy .npz 파일로 저장합니다.
    """
    try:
        # Tile_ID로 그룹화하고 각 그룹을 NumPy 배열로 변환
        grouped = df.groupby('Tile_ID')[['Gaussian_ID', 'Depth']].apply(
            lambda x: x.to_numpy(dtype=np.float32)
        )
        
        # .npz 파일로 저장하기 위해 딕셔너리 생성 (키는 문자열이어야 함)
        data_to_save = {str(tile_id): array for tile_id, array in grouped.items()}
        
        # 압축하여 저장
        np.savez_compressed(dest_npz_path, **data_to_save)

    except Exception as e:
        print(f"\n[ERROR] Failed to create .npz file from DataFrame: {e}")

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    
    debug_output_path = None
    if pipeline.debug:
        # 바이너리 데이터를 저장할 새로운 디렉토리 경로 설정
        debug_output_path = os.path.join(model_path, "renders_npz", name)
        makedirs(debug_output_path, exist_ok=True)
        
        # C++ 백엔드가 생성하는 임시 파일 이름
        source_file_unsorted = "unsorted_tile_data.txt"
        source_file_sorted = "sorted_tile_data.txt"
        source_file_contrib = "pixel_contrib_counts.txt"
        
        # 렌더링 시작 전 이전 임시 파일 정리
        for f in [source_file_unsorted, source_file_sorted, source_file_contrib]:
            if os.path.exists(f):
                os.remove(f)

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        outputs = render(view[1].cuda(), gaussians, pipeline, background)
        rendering = outputs["render"]
        
        if pipeline.debug:
            wait_time = 0
            # C++ 백엔드가 임시 파일을 모두 생성할 때까지 대기
            while not (os.path.exists(source_file_sorted) and os.path.exists(source_file_contrib) and os.path.exists(source_file_unsorted)) and wait_time < 3.0:
                time.sleep(0.01)
                wait_time += 0.01

            csv_header = ['Instance_Index', 'Tile_ID', 'Gaussian_ID', 'Depth']

            # 정렬되지 않은(unsorted) 데이터를 .npz로 저장
            if os.path.exists(source_file_unsorted):
                df_unsorted = pd.read_csv(source_file_unsorted, header=None, names=csv_header)
                # .npz 파일로 저장
                dest_path = os.path.join(debug_output_path, f"grouped_unsorted_list_{idx:05d}.npz")
                save_grouped_list_as_npz(df_unsorted, dest_path)
                os.remove(source_file_unsorted)

            # CSV 관련 임시 파일이 존재하면 삭제만 수행
            if os.path.exists(source_file_sorted):
                os.remove(source_file_sorted)
            if os.path.exists(source_file_contrib):
                os.remove(source_file_contrib)

            if not os.path.exists(source_file_unsorted) and not os.path.exists(source_file_sorted) and idx == 0:
                print(f"\n[DEBUG] C++ output files not found.")
        
        gt = view[0][0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

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
    if not args.source_path: # If source_path is not provided, assume it's the same as model_path
        args.source_path = args.model_path
    print("Rendering " + args.model_path)

    # Rendering Loop
    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)