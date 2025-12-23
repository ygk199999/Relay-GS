import json
import csv
import math
import numpy as np
import os
import argparse
import concurrent.futures
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

try:
    # 수정된 sorting_algorithms 모듈을 임포트
    from sorting_algorithms import CostTracker, incremental_sort_merge, full_sort, get_id
except ImportError:
    print("Error: 'sorting_algorithms.py' not found."); exit()

# --- (read_csv_to_list, read_and_parse_tile_csv, save_... 등 헬퍼 함수는 동일) ---
def read_csv_to_list(path):
    data = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cleaned_row = {}
                for k, v in row.items():
                    key = k.strip(); value = v.strip()
                    try: cleaned_row[key] = int(value)
                    except (ValueError, TypeError):
                        try: cleaned_row[key] = float(value)
                        except (ValueError, TypeError): cleaned_row[key] = value
                data.append(cleaned_row)
    except FileNotFoundError: return None
    return data

def read_and_parse_tile_csv(path):
    tile_stats, per_tile_data = {}, defaultdict(list)
    if not path or not os.path.exists(path): return None, None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    tile_id, count = int(row['Tile_ID']), int(row['Gaussian_Count'])
                    tile_stats[tile_id] = {"total": count, "rendered": count}
                    if count > 0:
                        ids_str, depths_str = row.get('Gaussian_IDs', '').strip(), row.get('Depths', '').strip()
                        if not ids_str or not depths_str: continue
                        ids = [int(i) for i in ids_str.split(' ')]; depths = [float(d) for d in depths_str.split(' ')]
                        if len(ids) == len(depths): per_tile_data[tile_id] = list(zip(ids, depths))
                except (ValueError, KeyError, TypeError): continue
    except FileNotFoundError: return None, None
    return tile_stats, per_tile_data

def save_report_to_json(frame_idx, performance_stats, op_counts, config, output_dir="reports"):
    pass # (주석 처리됨)

def save_list_of_dicts_to_csv(data, file_path):
    if not data: return
    try:
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        print(f"[+] Performance breakdown for graphing saved to: {file_path}")
    except (IOError, IndexError) as e: print(f"Error writing to CSV {file_path}: {e}")

# --- Core Logic ---
def _calculate_vru_latency_for_tile(rendered_gaussians, avg_contrib, config):
    if rendered_gaussians == 0: return 0
    vru_config = config['hardware_architecture']['vru_config']; vru_costs = config['cycle_costs']['vru']
    pixels_per_tile = vru_config.get('num_vr_cores_x', 1) * vru_config.get('pixels_per_vr_core_x', 1) * vru_config.get('num_vr_cores_y', 1) * vru_config.get('pixels_per_vr_core_y', 1)
    num_vr_cores = vru_config.get('num_vr_cores_x', 1) * vru_config.get('num_vr_cores_y', 1)
    vru_serial_work = avg_contrib * pixels_per_tile * vru_costs.get('alpha_computation_per_pixel', 0)
    return vru_serial_work / num_vr_cores if num_vr_cores > 0 else vru_serial_work

def _calculate_performance_for_counts(op_counts, config, gsu_vru_latency_override=None):
    hw, gs, costs = config['hardware_architecture'], config['simulation_global_settings'], config['cycle_costs']
    clk_hz = gs.get('clock_frequency_ghz', 1.0) * 1.0e9
    num_ccu_pipes = hw['ccu_config'].get('num_pipelines', 4)
    mem_config = hw['memory_config']
    bytes_per_cycle = (mem_config.get('dram_bandwidth_gb_per_sec', 32) * 1e9) / clk_hz if clk_hz > 0 else 0
    
    input_gaussians_count = op_counts.get('ccu_input_gaussians', 0)
    processed_gaussians_count = op_counts.get('ccu_visible_gaussians', 0)
    
    # CCU Compute
    cull_w = op_counts.get('ccu_frustum_culling_check', 0) * costs['ccu'].get('frustum_culling_check', 0)
    cov_w = op_counts.get('ccu_covariance_computation', 0) * costs['ccu'].get('ccu_covariance_computation', 0)
    sh_w = op_counts.get('ccu_sh_evaluation', 0) * costs['ccu'].get('ccu_sh_evaluation', 0)
    tile_isect_w = op_counts.get('ccu_tile_intersection_ops', 0) * costs['ccu'].get('aabb_intersect_check_per_tile', 1)
    subtile_gen_w = op_counts.get('ccu_subtile_generation_ops', 0) * costs['ccu'].get('subtile_bitmap_calc_per_gaussian', 1)
    ccu_compute_breakdown = {"Frustum Culling": cull_w, "Covariance Comp.": cov_w, "SH Evaluation": sh_w, "Tile Intersection": tile_isect_w, "Subtile Gen.": subtile_gen_w}
    ccu_compute_latency = max(v / num_ccu_pipes for v in ccu_compute_breakdown.values()) if num_ccu_pipes > 0 else 0
    
    # CCU Memory Load (In)
    bytes_per_full_gaussian = mem_config.get('bytes_per_raw_gaussian', 59)
    bytes_per_culling_info = mem_config.get('bytes_per_culling_info')
    if bytes_per_culling_info and bytes_per_culling_info > 0:
        bytes_per_detail_info = bytes_per_full_gaussian - bytes_per_culling_info
        culling_info_bytes = input_gaussians_count * bytes_per_culling_info
        detail_info_bytes = processed_gaussians_count * bytes_per_detail_info
        raw_bytes = culling_info_bytes + detail_info_bytes
    else:
        raw_bytes = input_gaussians_count * bytes_per_full_gaussian
    mem_load_latency = raw_bytes / bytes_per_cycle if bytes_per_cycle > 0 else 0
    
    # CCU Memory Write (Out)
    bytes_per_sort_entry = mem_config.get('bytes_per_sort_entry', 6) # [우리 기준 적용]
    gsu_metadata_bytes = op_counts.get('gsu_total_gaussians_sorted', 0) * bytes_per_sort_entry
    vru_feature_bytes = processed_gaussians_count * mem_config.get('bytes_per_processed_gaussian', 48) # [우리 기준 적용]
    ccu_write_latency = (gsu_metadata_bytes + vru_feature_bytes) / bytes_per_cycle if bytes_per_cycle > 0 else 0
    
    ccu_latencies = {"Compute": ccu_compute_latency, "Memory Load (In)": mem_load_latency, "Memory Write (Out)": ccu_write_latency}
    ccu_stage_latency = max(ccu_latencies.values())
    ccu_bottleneck_source = max(ccu_latencies, key=ccu_latencies.get)
    
    gsu_vru_stage_latency = gsu_vru_latency_override if gsu_vru_latency_override is not None else 0
    gsu_vru_bottleneck_source = op_counts.get('gsu_vru_bottleneck_source', "N/A")

    total_system_latency = ccu_stage_latency + gsu_vru_stage_latency
    fps = clk_hz / total_system_latency if total_system_latency > 0 else 0
    
    return {"fps": fps, "time_per_frame_ms": 1000/fps if fps > 0 else 0, "total_latency": total_system_latency, "ccu_stage_latency": ccu_stage_latency, "ccu_bottleneck_source": ccu_bottleneck_source, "gsu_vru_stage_latency": gsu_vru_stage_latency, "gsu_vru_bottleneck_source": gsu_vru_bottleneck_source}

# ==============================================================================
# === [수정됨] 파이프라인 계산 로직 추가 ===
# ==============================================================================
def _calculate_pipelined_latency(cost_bd, vru_latency_for_tile):
    """
    정렬 상세 내역(cost_bd)과 VRU 시간을 받아
    '파이프라인 겹침(overlap)'을 적용한 최종 GSU-VRU 시간을 계산합니다.
    """
    
    method_name = cost_bd.get('method_name', 'Full Sort (GSCore)')
    
    if method_name == "Full Sort (GSCore)":
        # === (A) GSCore (Baseline) 파이프라인 ===
        # 총 시간 = (Stage 1) + (겹치는 Stage 2)
        
        # 1. 초기 지연 시간: Stage 1 (Approx. Sort)이 끝날 때까지 VRU 대기
        init_latency = cost_bd['approx_sort_cost']
        
        # 2. 파이프라인 시간: MAX(Precise Sort 1청크, VRU 1청크) * 청크 수
        num_chunks = cost_bd['num_chunks']
        if num_chunks == 0:
            return init_latency, "N/A" # 래스터할 것이 없음
            
        sort_per_chunk = cost_bd['precise_sort_cost_per_chunk']
        raster_per_chunk = vru_latency_for_tile / num_chunks
        
        # MAX(정렬, 래스터)가 병목이 되어 파이프라인 시간을 결정
        pipeline_bottleneck = max(sort_per_chunk, raster_per_chunk)
        pipeline_time = pipeline_bottleneck * num_chunks
        
        final_latency = init_latency + pipeline_time
        bottleneck_source = "Sorting (Precise)" if sort_per_chunk > raster_per_chunk else "Rasterization"
        
        return final_latency, bottleneck_source

    else:
        # === (B) 우리 방식 (Hash) 파이프라인 ===
        # 총 시간 = (초기 준비) + (겹치는 Merge/Raster)
        
        # 1. 초기 지연 시간: Step 1(분리) + Step 2(새 항목 정렬)
        init_latency = cost_bd['separation'].final_cycles + cost_bd['sort_new'].final_cycles
        
        # 2. 파이프라인 시간: MAX(Merge, Rasterization)
        # 우리 방식은 '가우시안 1개' 단위로 겹치므로, 두 작업의 총 시간을 비교
        merge_time = cost_bd['merge'].final_cycles
        raster_time = vru_latency_for_tile
        
        pipeline_bottleneck = max(merge_time, raster_time)
        pipeline_time = pipeline_bottleneck # 둘 중 더 오래 걸리는 작업이 전체 시간을 결정
        
        final_latency = init_latency + pipeline_time
        bottleneck_source = "Sorting (Merge)" if merge_time > raster_time else "Rasterization"

        return final_latency, bottleneck_source

# ==============================================================================
# === [수정됨] 리포트 함수 수정 ===
# ==============================================================================
def generate_full_report_text(frame_idx, stats, sort_stats):
    lines = ["\n" + "="*85, f"Analysis Report for Frame: {frame_idx}", 
             f"\n[ Performance Summary (Pipelined) ]",
             f"- Estimated FPS: {stats.get('fps_pipelined', 0):.2f}",
             f"- Time per Frame: {stats.get('time_per_frame_ms_pipelined', 0):.2f} ms", 
             f"- Total System Latency: {int(stats.get('total_latency_pipelined', 0)):,} cycles",
             
             f"\n[ Performance Summary (Sequential) ]",
             f"- Estimated FPS: {stats.get('fps_sequential', 0):.2f}",
             f"- Time per Frame: {stats.get('time_per_frame_ms_sequential', 0):.2f} ms",
             f"- Total System Latency: {int(stats.get('total_latency_sequential', 0)):,} cycles",

             "\n[ System-Level Pipeline Breakdown (Pipelined) ]", 
             f"1. CCU Stage Latency:          {int(stats.get('ccu_stage_latency', 0)):>12,} cycles (Bottleneck: {stats.get('ccu_bottleneck_source', 'N/A')})",
             f"2. GSU-VRU Stage Latency:     {int(stats.get('gsu_vru_stage_latency_pipelined', 0)):>12,} cycles (Bottleneck: {stats.get('gsu_vru_bottleneck_source_pipelined', 'N/A')})",
             
             "\n[ GSU-VRU Method Comparison (Latency of the Bottleneck Pipeline) ]"]
    
    proposed_latency_pipe = sort_stats.get('proposed_latency_pipelined', 0)
    baseline_latency_pipe = sort_stats.get('baseline_latency_pipelined', 0)
    
    lines.append(f"- Sort Method                 : {sort_stats.get('method_name', 'N/A')}")
    lines.append(f"- Hash Method                 : {sort_stats.get('hash_method', 'N/A')}")
    lines.append(f"- Proposed Latency (Pipelined)  : {int(proposed_latency_pipe):>15,} cycles")
    if 'breakdown_sort_total' in sort_stats and sort_stats['breakdown_sort_total']:
        bd = sort_stats['breakdown_sort_total']
        lines.append(f"  - Sorting Sub-Total (Seq)   : {int(bd.final_cycles):>15,} cycles")
        if sort_stats.get('method_name') == 'Incremental Merge (Hash)':
            lines.append(f"    - Separation            : {int(sort_stats.get('breakdown_sep', 0)):>15,} cycles")
            lines.append(f"    - Sort New              : {int(sort_stats.get('breakdown_sort_new', 0)):>15,} cycles")
            lines.append(f"    - Merge                 : {int(sort_stats.get('breakdown_merge', 0)):>15,} cycles")
        lines.append(f"  - Raster Sub-Total          : {int(sort_stats.get('breakdown_vru', 0)):>15,} cycles")

    lines.append(f"- Baseline Latency (Pipelined)  : {int(baseline_latency_pipe):>15,} cycles")
    if baseline_latency_pipe > 0: lines.append(f"- Savings vs Baseline (Pipe)  : {(1 - proposed_latency_pipe / baseline_latency_pipe) * 100:>15.2f}%")
    lines.append("=" * 85)
    return "\n".join(lines)

# ==============================================================================
# === [수정됨] process_frame 함수 수정 ===
# ==============================================================================
def process_frame(frame_idx, frame_data, config, log_dirs, start_frame, hash_sim_mode):
    current_tile_log_path = os.path.join(log_dirs['current_tile'], f"unsorted_{frame_idx:05d}.csv")
    prev_tile_log_path = os.path.join(log_dirs['prev_tile'], f"sorted_{frame_idx-1:05d}.csv")
    tile_stats, current_tile_data = read_and_parse_tile_csv(current_tile_log_path)
    if not current_tile_data: return None

    is_first_frame = (frame_idx == start_frame)
    prev_tile_data = None if is_first_frame else read_and_parse_tile_csv(prev_tile_log_path)[1]
    
    contrib_log_path = os.path.join(log_dirs['contrib'], f"contrib_summary_{frame_idx:05d}.csv")
    contrib_data = read_csv_to_list(contrib_log_path)
    avg_max_contrib = np.mean([row['Max_Contributor_Count'] for row in contrib_data]) if contrib_data else 0

    num_pipes = config['hardware_architecture']['gsu_config'].get('num_pipelines', 1)
    all_tile_ids = sorted(list(set(current_tile_data.keys()) | (set(prev_tile_data.keys()) if prev_tile_data else set())))
    
    # [수정됨] 파이프라인/순차 비용을 모두 저장하도록 pipe_workloads 확장
    pipe_workloads = [{'total_sequential': 0, 'total_pipelined': 0,
                       'sorting_total': CostTracker(), 'sorting_separation': CostTracker(), 
                       'sorting_sort_new': CostTracker(), 'sorting_merge': CostTracker(), 
                       'vru': 0} for _ in range(num_pipes)]
    
    is_full_sort_mode = is_first_frame or prev_tile_data is None
    total_still_gaussians, total_new_gaussians = 0, 0

    # --- 1. Proposed Method (우리 방식) 시뮬레이션 ---
    for i, tile_id in enumerate(all_tile_ids):
        pipe_idx = i % num_pipes
        current_list = current_tile_data.get(tile_id, [])
        prev_list = [] if is_full_sort_mode else prev_tile_data.get(tile_id, [])
        
        if not is_full_sort_mode:
            prev_ids = {get_id(item) for item in prev_list}; current_ids = {get_id(item) for item in current_list}
            total_new_gaussians += len(current_ids - prev_ids); total_still_gaussians += len(current_ids & prev_ids)

        # 상세 내역이 포함된 cost_bd를 받음
        _, cost_bd = incremental_sort_merge(prev_list, current_list, config)
        if cost_bd['method_name'] == "Full Sort (GSCore)": is_full_sort_mode = True
        
        rendered_gaussians = tile_stats.get(tile_id, {}).get('rendered', 0)
        vru_latency_for_tile = _calculate_vru_latency_for_tile(rendered_gaussians, avg_max_contrib, config)
        
        # [수정됨] 순차 비용과 파이프라인 비용을 별도 계산
        total_sorting_cost = cost_bd['total']
        latency_sequential = total_sorting_cost.final_cycles + vru_latency_for_tile
        latency_pipelined, _ = _calculate_pipelined_latency(cost_bd, vru_latency_for_tile)
        
        pipe_workloads[pipe_idx]['total_sequential'] += latency_sequential
        pipe_workloads[pipe_idx]['total_pipelined'] += latency_pipelined
        
        # (상세 내역 저장은 동일)
        pipe_workloads[pipe_idx]['sorting_total'].compute_cycles += total_sorting_cost.compute_cycles
        pipe_workloads[pipe_idx]['sorting_total'].memory_cycles += total_sorting_cost.memory_cycles
        pipe_workloads[pipe_idx]['sorting_total'].final_cycles += total_sorting_cost.final_cycles
        
        for phase in ['separation', 'sort_new', 'merge']:
             cost_item = cost_bd.get(phase, CostTracker())
             pipe_workloads[pipe_idx][f'sorting_{phase}'].compute_cycles += cost_item.compute_cycles
             pipe_workloads[pipe_idx][f'sorting_{phase}'].memory_cycles += cost_item.memory_cycles
             pipe_workloads[pipe_idx][f'sorting_{phase}'].final_cycles += cost_item.final_cycles
        pipe_workloads[pipe_idx]['vru'] += vru_latency_for_tile

    # [수정됨] 병목 파이프라인의 최종 시간 (Pipelined 기준)
    pipeline_total_latencies = [w['total_pipelined'] for w in pipe_workloads]
    if not pipeline_total_latencies:
        frame_gsu_vru_latency_pipelined, bottleneck_pipe_idx = 0, 0
    else:
        bottleneck_pipe_idx = np.argmax(pipeline_total_latencies)
        frame_gsu_vru_latency_pipelined = pipeline_total_latencies[bottleneck_pipe_idx]
    
    # [수정됨] 병목 파이프라인의 순차(Sequential) 시간도 기록
    frame_gsu_vru_latency_sequential = [w['total_sequential'] for w in pipe_workloads][bottleneck_pipe_idx]
    
    bottleneck_workload = pipe_workloads[bottleneck_pipe_idx]
    bottleneck_sorting_total = bottleneck_workload['sorting_total']
    bottleneck_vru_latency = bottleneck_workload['vru']
    
    sep_cycles = bottleneck_workload['sorting_separation'].final_cycles
    sort_new_cycles = bottleneck_workload['sorting_sort_new'].final_cycles
    merge_cycles = bottleneck_workload['sorting_merge'].final_cycles

    # --- 2. Baseline Method (GSCore) 시뮬레이션 ---
    baseline_workloads_sequential = [0] * num_pipes
    baseline_workloads_pipelined = [0] * num_pipes
    
    for i, tile_id in enumerate(all_tile_ids):
        pipe_idx = i % num_pipes
        # [수정됨] GSCore 방식(prev_list=[])으로 상세 내역(cost_bd)을 받음
        _, cost_bd = incremental_sort_merge([], current_tile_data.get(tile_id, []), config)
        
        rendered_gaussians = tile_stats.get(tile_id, {}).get('rendered', 0)
        vru_latency = _calculate_vru_latency_for_tile(rendered_gaussians, avg_max_contrib, config)
        
        # [수정됨] GSCore 방식의 순차 비용과 파이프라인 비용을 별도 계산
        latency_sequential = cost_bd['total'].final_cycles + vru_latency
        latency_pipelined, _ = _calculate_pipelined_latency(cost_bd, vru_latency)
        
        baseline_workloads_sequential[pipe_idx] += latency_sequential
        baseline_workloads_pipelined[pipe_idx] += latency_pipelined

    frame_baseline_latency_sequential = max(baseline_workloads_sequential) if baseline_workloads_sequential else 0
    frame_baseline_latency_pipelined = max(baseline_workloads_pipelined) if baseline_workloads_pipelined else 0
    
    # --- 3. 리포트 생성 ---
    op_counts = { 'ccu_input_gaussians': frame_data.get('Total_Gaussians', 0), 'ccu_visible_gaussians': frame_data.get('Gaussians_In_View', 0), 'gsu_total_gaussians_sorted': sum(len(v) for v in current_tile_data.values()), 
                 'ccu_frustum_culling_check': frame_data.get('Total_Gaussians', 0), 'ccu_covariance_computation': frame_data.get('Gaussians_In_View', 0), 'ccu_sh_evaluation': frame_data.get('Gaussians_In_View', 0), 
                 'ccu_tile_intersection_ops': int(frame_data.get('Gaussians_In_View', 0) * 2.5), 'ccu_subtile_generation_ops': int(frame_data.get('Gaussians_In_View', 0) * 2.5 * 4.0)}
    
    # [수정됨] 파이프라인 기준 병목 계산
    sort_method_name = "Full Sort (GSCore)" if is_full_sort_mode else "Incremental Merge (Hash)"
    
    # (Pipelined) 병목 소스 계산
    _, gsu_vru_bottleneck_source_pipelined = _calculate_pipelined_latency(
        cost_bd={'method_name': sort_method_name, **cost_bd}, # (임시 cost_bd 사용, 어차피 method_name만 필요)
        vru_latency_for_tile=bottleneck_vru_latency
    )
    op_counts['gsu_vru_bottleneck_source_pipelined'] = gsu_vru_bottleneck_source_pipelined
    
    # [수정됨] _calculate_performance_for_counts는 이제 두 개의 Latency를 받음
    frame_stats_seq = _calculate_performance_for_counts(op_counts, config, gsu_vru_latency_override=frame_gsu_vru_latency_sequential)
    frame_stats_pipe = _calculate_performance_for_counts(op_counts, config, gsu_vru_latency_override=frame_gsu_vru_latency_pipelined)
    
    # 두 통계를 합침
    frame_stats = {
        'fps_sequential': frame_stats_seq['fps'], 'time_per_frame_ms_sequential': frame_stats_seq['time_per_frame_ms'], 'total_latency_sequential': frame_stats_seq['total_latency'],
        'fps_pipelined': frame_stats_pipe['fps'], 'time_per_frame_ms_pipelined': frame_stats_pipe['time_per_frame_ms'], 'total_latency_pipelined': frame_stats_pipe['total_latency'],
        'ccu_stage_latency': frame_stats_pipe['ccu_stage_latency'], 'ccu_bottleneck_source': frame_stats_pipe['ccu_bottleneck_source'],
        'gsu_vru_stage_latency_pipelined': frame_stats_pipe['gsu_vru_stage_latency'], 'gsu_vru_bottleneck_source_pipelined': gsu_vru_bottleneck_source_pipelined
    }
    
    sort_stats = {"proposed_latency_pipelined": frame_gsu_vru_latency_pipelined, 
                  "baseline_latency_pipelined": frame_baseline_latency_pipelined, 
                  "method_name": sort_method_name, "hash_method": hash_sim_mode if sort_method_name != "Full Sort (GSCore)" else "N/A", 
                  "breakdown_sort_total": bottleneck_sorting_total, # 'total' (Sequential)
                  "breakdown_vru": bottleneck_vru_latency, 
                  "breakdown_sep": sep_cycles, "breakdown_sort_new": sort_new_cycles, "breakdown_merge": merge_cycles}
    
    report_text = generate_full_report_text(frame_idx, frame_stats, sort_stats)
    
    # [수정됨] CSV에 두 Latency 모두 기록
    csv_data_row = {'frame_idx': frame_idx, 
                    'fps_pipelined': round(frame_stats['fps_pipelined'], 2), 
                    'total_latency_pipelined': int(frame_stats['total_latency_pipelined']),
                    'gsu_vru_latency_pipelined': int(frame_stats['gsu_vru_stage_latency_pipelined']),
                    'baseline_gsu_vru_latency_pipelined': int(frame_baseline_latency_pipelined),
                    'fps_sequential': round(frame_stats['fps_sequential'], 2),
                    'total_latency_sequential': int(frame_stats['total_latency_sequential']),
                    'gsu_vru_latency_sequential': int(frame_gsu_vru_latency_sequential),
                    'baseline_gsu_vru_latency_sequential': int(frame_baseline_latency_sequential),
                    'ccu_stage_latency': int(frame_stats['ccu_stage_latency']),
                    'gsu_vru_sorting_latency': int(bottleneck_sorting_total.final_cycles), 
                    'gsu_vru_raster_latency': int(bottleneck_vru_latency),
                    'total_gaussians': op_counts['gsu_total_gaussians_sorted'], 'still_gaussians': total_still_gaussians, 'new_gaussians': total_new_gaussians, 
                    'sort_method': sort_method_name, 'hash_method': hash_sim_mode if sort_method_name != "Full Sort (GSCore)" else "N/A", 
                    'gsu_sorting_sep_cycles': int(sep_cycles), 'gsu_sorting_sort_new_cycles': int(sort_new_cycles), 'gsu_sorting_merge_cycles': int(merge_cycles)}
    
    return (frame_idx, frame_stats, frame_gsu_vru_latency_pipelined, frame_baseline_latency_pipelined, report_text, csv_data_row)

# ==============================================================================
# === [수정됨] main 함수 수정 ===
# ==============================================================================
def main(log_scene_path, start_frame=None, end_frame=None):
    if not os.path.isdir(log_scene_path): print(f"Error: Log directory does not exist: {log_scene_path}"); return
    print(f"[*] Analyzing logs from: {os.path.abspath(log_scene_path)}")
    
    scene_name = os.path.basename(os.path.normpath(log_scene_path))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    scene_report_dir = os.path.join("report", f"{scene_name}_reports_{timestamp}")
    os.makedirs(scene_report_dir, exist_ok=True)
    
    log_dirs = {'contrib': os.path.join(log_scene_path, "pixel_contrib_summary"), 'current_tile': os.path.join(log_scene_path, "unsorted_with_depth"), 'prev_tile': os.path.join(log_scene_path, "sorted_with_depth")}
    
    try:
        # Config files are in the 'src' directory
        config_path = "src/config.json"
        config_analyze_path = "src/config_analyze.json"
        with open(config_path, 'r') as f: config = json.load(f)
        with open(config_analyze_path, 'r') as f: config_analyze = json.load(f)
        for k, v in config_analyze.items(): config[k] = {**config.get(k, {}), **v} if isinstance(v, dict) else v
    except FileNotFoundError as e: print(f"Error: Config file not found: {e.filename}"); return
        
    sim_settings = config.get('simulation_global_settings', {})
    hash_sim_mode = "Detailed" if sim_settings.get('detailed_hash_sim', False) else "Fast"

    culling_data = read_csv_to_list(os.path.join(log_scene_path, "culling_log.csv"))
    if not culling_data: print(f"Error: Main culling log not found"); return

    start_frame = start_frame if start_frame is not None else culling_data[0].get('View_Index', 0)
    end_frame = end_frame if end_frame is not None else culling_data[-1].get('View_Index', 0)
    title = f"Analyzing Frames {start_frame} to {end_frame}"
    print(f"\n{'='*85}\n GSCore Performance Estimator ({title})\n{'='*85}")

    summary_log_filename = os.path.join(scene_report_dir, "analysis_summary.txt")
    csv_output_path = os.path.join(scene_report_dir, "performance_details_over_frames.csv")
    with open(summary_log_filename, 'w') as f: 
        f.write(f"GSCore Analysis - {datetime.now()}\n")
        f.write(f"Range: {title}\n")
        f.write(f"Source: {os.path.abspath(log_scene_path)}\n")
        f.write(f"Hash Simulation Mode: {hash_sim_mode}\n")

    # [수정됨] hash_sim_mode를 process_frame의 인자로 전달
    tasks = [(fd['View_Index'], fd, config, log_dirs, start_frame, hash_sim_mode) for fd in culling_data if start_frame <= fd['View_Index'] <= end_frame]
    
    all_results, all_csv_data = [], []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_frame, *task): task for task in tasks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc="Processing Frames"):
            try:
                if (result := future.result()): all_results.append(result)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f'\nFrame {futures[future][0]} generated an exception: {e}')

    all_results.sort(key=lambda x: x[0])
    total_proposed_pipe, total_baseline_pipe = 0, 0
    with open(summary_log_filename, 'a') as f:
        for idx, stats, proposed_pipe, baseline_pipe, report, csv_row in all_results:
            total_proposed_pipe += proposed_pipe; total_baseline_pipe += baseline_pipe
            all_csv_data.append(csv_row)
            # [수정됨] Pipelined 기준으로 콘솔 출력
            print(f"  - Frame {idx} analyzed: FPS (Pipelined): {stats['fps_pipelined']:.2f} | Proposed GSU-VRU (Pipe): {int(proposed_pipe):,} | Baseline GSU-VRU (Pipe): {int(baseline_pipe):,}")
            f.write(report)
            
    save_list_of_dicts_to_csv(all_csv_data, csv_output_path)
    
    # [수정됨] Pipelined 기준으로 콘솔 요약 출력
    print("\n[ Console Performance Summary (Pipelined) ]")
    if not all_results: print("  - No frames were successfully analyzed.")
    else: [print(f"  - Frame {idx:<4}: FPS: {stats['fps_pipelined']:<8.2f} Total Latency: {int(stats['total_latency_pipelined']):>12,} cycles") for idx, stats, _, _, _, _ in all_results]
    if len(all_results) > 1: print(f"\n[ Average FPS Over Analyzed Sequence (Pipelined): {np.mean([r[1]['fps_pipelined'] for r in all_results]):.2f} ]")
    
    print("\n[ Total GSU-VRU Latency Comparison (Pipelined, Over Full Sequence) ]")
    print(f"- Total Proposed Method Latency: {int(total_proposed_pipe):,} cycles")
    print(f"- Total Baseline Method Latency: : {int(total_baseline_pipe):,} cycles")
    if total_baseline_pipe > 0: print(f"- Overall Latency Savings:       {(1 - total_proposed_pipe / total_baseline_pipe) * 100:.2f}%")

    print(f"\n[+] Report saved to: {summary_log_filename}")
    print("\n" + "="*85)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GSCore Performance Estimator with parallel processing.")
    parser.add_argument("-p", "--path", type=str, required=True, help="Path to the scene log directory.")
    parser.add_argument("-s", "--start", type=int, help="Starting frame number.")
    parser.add_argument("-e", "--end", type=int, help="Ending frame number.")
    args = parser.parse_args()
    main(log_scene_path=args.path, start_frame=args.start, end_frame=args.end)