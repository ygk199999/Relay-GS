import pandas as pd
import numpy as np
import argparse
import json
import os

def analyze_performance(csv_path, config_path="src/config_merged.json"):
    """
    performance_details_over_frames.csv 파일을 읽어
    각 시나리오의 상세 레이턴시 내역, 평균 성능, 
    [Baseline-Seq] 및 [Baseline-Pipe] 대비 Speedup을 모두 계산하고 출력합니다.
    (첫 번째 프레임은 평균 계산에서 제외합니다.)
    """
    # --- 1. 데이터 및 설정 파일 로드 ---
    try:
        df_all_frames = pd.read_csv(csv_path)
        print(f"[*] Successfully loaded {len(df_all_frames)} total frames from '{os.path.basename(csv_path)}'")
    except FileNotFoundError:
        print(f"Error: CSV file not found at '{csv_path}'")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file '{csv_path}' is empty.")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # --- 2. 첫 번째 프레임(index 0)을 평균 계산에서 제외 ---
    if len(df_all_frames) > 1:
        df = df_all_frames.iloc[1:].copy()
        first_frame_idx = df_all_frames.iloc[0].get('frame_idx', 0)
        print(f"[*] Excluding first frame (frame_idx {first_frame_idx}) from average calculation.")
        print(f"[*] Analyzing results based on {len(df)} frames (frame {df.iloc[0].get('frame_idx', 1)} to {df.iloc[-1].get('frame_idx', -1)}).")
    elif len(df_all_frames) == 1:
        print("[Warning] Only one frame found in CSV. Analysis will be based on this single frame.")
        df = df_all_frames.copy()
    else:
        print("[Error] No data frames found to analyze.")
        return

    # --- 3. 설정 로드 (Clock 등) ---
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        clk_hz = config['simulation_global_settings'].get('clock_frequency_ghz', 0.3) * 1.0e9
    except FileNotFoundError:
        print(f"Warning: Config file '{config_path}' not found. Assuming 0.3 GHz clock.")
        clk_hz = 0.3 * 1.0e9
    except Exception as e:
        print(f"Error loading config '{config_path}': {e}. Assuming 0.3 GHz clock.")
        clk_hz = 0.3 * 1.0e9

    # --- 4. 컬럼 확인 ---
    required_cols = [
        'ccu_stage_latency', 'gsu_vru_baseline_latency_sequential', 
        'gsu_vru_baseline_latency_pipelined', 'gsu_baseline_latency_total', 
        'vru_latency_on_baseline', 'total_latency_sequential', 
        'total_latency_pipelined', 'vru_latency_on_incremental', 
        'gsu_incremental_latency_total', 'gsu_incremental_latency_sep', 
        'gsu_incremental_latency_sgs', 'gsu_incremental_latency_merge', 
        'dram_read_gsu_incremental_bits', 'dram_read_gsu_baseline_bits'
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"\nError: CSV file is missing required columns: {missing_cols}")
        return

    # --- 5. 시나리오별 평균 상세 레이턴시 계산 ---
    
    # 공통
    avg_ccu_lat = df['ccu_stage_latency'].mean()
    
    # Baseline
    avg_base_gsu_lat = df['gsu_baseline_latency_total'].mean()
    avg_base_vru_lat = df['vru_latency_on_baseline'].mean()
    avg_base_gsu_vru_pipe_lat = df['gsu_vru_baseline_latency_pipelined'].mean()
    
    # Incremental
    avg_incr_gsu_total_lat = df['gsu_incremental_latency_total'].mean()
    avg_incr_gsu_sep_lat = df['gsu_incremental_latency_sep'].mean()
    avg_incr_gsu_sgs_lat = df['gsu_incremental_latency_sgs'].mean()
    avg_incr_gsu_merge_lat = df['gsu_incremental_latency_merge'].mean()
    avg_incr_vru_lat = df['vru_latency_on_incremental'].mean()
    avg_incr_vru_half_lat = avg_incr_vru_lat / 2.0

    analysis_results = {}
    baseline_seq_fps = 0.0
    baseline_pipe_fps = 0.0

    # --- 시나리오 1: Baseline (Sequential) ---
    scenario_name = "Baseline (Sequential)"
    total_lat = avg_ccu_lat + avg_base_gsu_lat + avg_base_vru_lat
    fps = clk_hz / total_lat if total_lat > 0 else 0
    baseline_seq_fps = fps # 기준점 1
    analysis_results[scenario_name] = {
        "CCU": avg_ccu_lat, "GSU (Full Sort)": avg_base_gsu_lat, "VRU": avg_base_vru_lat,
        "Total": total_lat, "FPS": fps,
        "Speedup_vs_Seq": 1.0 # 기준
    }

    # --- 시나리오 2: Baseline (Pipelined) ---
    scenario_name = "Baseline (Pipelined)"
    total_lat = avg_ccu_lat + avg_base_gsu_vru_pipe_lat
    fps = clk_hz / total_lat if total_lat > 0 else 0
    baseline_pipe_fps = fps # 기준점 2
    analysis_results[scenario_name] = {
        "CCU": avg_ccu_lat, "GSU-VRU Pipe (Total)": avg_base_gsu_vru_pipe_lat,
        "Total": total_lat, "FPS": fps,
        "Speedup_vs_Seq": fps / baseline_seq_fps if baseline_seq_fps > 0 else 0
    }

    # --- 시나리오 3: Incremental (Sequential) ---
    scenario_name = "Incremental (Sequential)"
    total_lat = avg_ccu_lat + avg_incr_gsu_total_lat + avg_incr_vru_lat
    fps = clk_hz / total_lat if total_lat > 0 else 0
    analysis_results[scenario_name] = {
        "CCU": avg_ccu_lat, "GSU (Incr. Total)": avg_incr_gsu_total_lat, "VRU": avg_incr_vru_lat,
        "Total": total_lat, "FPS": fps,
        "Speedup_vs_Seq": fps / baseline_seq_fps if baseline_seq_fps > 0 else 0
    }

    # --- 시나리오 4: Incremental (Pipelined) ---
    scenario_name = "Incremental (Pipelined)"
    gsu_init_lat = avg_incr_gsu_sep_lat + avg_incr_gsu_sgs_lat
    pipe_stage_lat = np.maximum(avg_incr_gsu_merge_lat, avg_incr_vru_lat)
    total_lat = avg_ccu_lat + gsu_init_lat + pipe_stage_lat
    fps = clk_hz / total_lat if total_lat > 0 else 0
    analysis_results[scenario_name] = {
        "CCU": avg_ccu_lat, "GSU (Init)": gsu_init_lat, "GSU-VRU Pipe (max)": pipe_stage_lat,
        "(Memo) GSU-Merge": avg_incr_gsu_merge_lat, "(Memo) VRU": avg_incr_vru_lat,
        "Total": total_lat, "FPS": fps,
        "Speedup_vs_Seq": fps / baseline_seq_fps if baseline_seq_fps > 0 else 0
    }

    # --- 시나리오 5: Incremental (Seq, VRU/2) ---
    scenario_name = "Incremental (Seq, VRU/2)"
    total_lat = avg_ccu_lat + avg_incr_gsu_total_lat + avg_incr_vru_half_lat
    fps = clk_hz / total_lat if total_lat > 0 else 0
    analysis_results[scenario_name] = {
        "CCU": avg_ccu_lat, "GSU (Incr. Total)": avg_incr_gsu_total_lat, "VRU (Half)": avg_incr_vru_half_lat,
        "Total": total_lat, "FPS": fps,
        "Speedup_vs_Seq": fps / baseline_seq_fps if baseline_seq_fps > 0 else 0
    }

    # --- 시나리오 6: Incremental (Pipe, VRU/2) ---
    scenario_name = "Incremental (Pipe, VRU/2)"
    gsu_init_lat = avg_incr_gsu_sep_lat + avg_incr_gsu_sgs_lat
    pipe_stage_lat = np.maximum(avg_incr_gsu_merge_lat, avg_incr_vru_half_lat)
    total_lat = avg_ccu_lat + gsu_init_lat + pipe_stage_lat
    fps = clk_hz / total_lat if total_lat > 0 else 0
    analysis_results[scenario_name] = {
        "CCU": avg_ccu_lat, "GS (Init)": gsu_init_lat, "GSU-VRU Pipe (max)": pipe_stage_lat,
        "(Memo) GSU-Merge": avg_incr_gsu_merge_lat, "(Memo) VRU (Half)": avg_incr_vru_half_lat,
        "Total": total_lat, "FPS": fps,
        "Speedup_vs_Seq": fps / baseline_seq_fps if baseline_seq_fps > 0 else 0
    }

    # --- [수정] Speedup_vs_Pipe 값을 모든 시나리오에 추가 ---
    if baseline_pipe_fps > 0:
        for scenario, data in analysis_results.items():
            data["Speedup_vs_Pipe"] = data["FPS"] / baseline_pipe_fps
    else:
        for scenario, data in analysis_results.items():
            data["Speedup_vs_Pipe"] = 0.0 # 0으로 나누기 방지

    # --- 6. 최종 리포트 출력 ---
    print("\n" + "="*85)
    print(" Overall Performance & Speedup Analysis (Detailed Breakdown)")
    print(f" (Clock: {clk_hz/1e9:.2f} GHz)")
    print("="*85)

    for scenario, data in analysis_results.items():
        print(f"\n--- Scenario: {scenario} ---")
        
        for component, latency in data.items():
            # [수정] Speedup 키 이름 변경
            if component not in ["FPS", "Speedup_vs_Seq", "Speedup_vs_Pipe", "Total"] and not component.startswith("(Memo)"):
                print(f"  - Avg. {component:<18}: {latency:12,.0f} cycles")
            if component.startswith("(Memo)") and data.get(component, 0) > 0:
                 print(f"    {component:<16}: {latency:12,.0f} cycles")

        print("  --------------------------------------")
        print(f"  - Avg. Total Latency  : {data['Total']:12,.0f} cycles")
        print(f"  - Resulting Avg. FPS  : {data['FPS']:12.2f} FPS")
        # [수정] 두 종류의 Speedup 출력
        print(f"  - Speedup vs. Base-Seq: {data.get('Speedup_vs_Seq', 0.0):12.2f}x")
        print(f"  - Speedup vs. Base-Pipe: {data.get('Speedup_vs_Pipe', 0.0):12.2f}x")

    # --- 7. DRAM 분석 ---
    avg_dram_incremental_bytes = df['dram_read_gsu_incremental_bits'].mean() / 8
    avg_dram_baseline_bytes = df['dram_read_gsu_baseline_bits'].mean() / 8
    
    if avg_dram_baseline_bytes > 0:
        dram_savings = (1 - avg_dram_incremental_bytes / avg_dram_baseline_bytes) * 100
    else:
        dram_savings = 0.0

    print("\n" + "="*85)
    print("[Average GSU DRAM Read per Frame]")
    print(f"- Incremental Method         : {avg_dram_incremental_bytes:15,.0f} Bytes")
    print(f"- Baseline Method            : {avg_dram_baseline_bytes:15,.0f} Bytes")
    print(f"- DRAM Read Savings          : {dram_savings:15.2f}%")
    print("="*85)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Analyzes the detailed performance CSV generated by the GSCore performance estimator.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("csv_path", type=str, help="Path to the 'performance_details_over_frames.csv' file.")
    parser.add_argument("-c", "--config", type=str, default="src/config_merged.json", 
                        help="Path to the config.json file to get clock frequency.\nDefault: src/config_merged.json")
    args = parser.parse_args()
    
    if args.config == "src/config.json" and not os.path.exists("src/config.json"):
        print("Warning: 'src/config.json' not found, falling back to 'src/config_merged.json'")
        args.config = "src/config_merged.json"
        
    analyze_performance(args.csv_path, args.config)