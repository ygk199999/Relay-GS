import json
import csv
import math
import numpy as np
import os
import argparse
from datetime import datetime

# --- Helper Function 1: CSV Reader for Generic Data ---
def read_csv_to_list(path):
    """Reads a CSV file into a list of dictionaries."""
    data = []
    try:
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cleaned_row = {}
                for k, v in row.items():
                    try: cleaned_row[k.strip()] = int(v)
                    except (ValueError, TypeError):
                        try: cleaned_row[k.strip()] = float(v)
                        except (ValueError, TypeError): cleaned_row[k.strip()] = v
                data.append(cleaned_row)
    except FileNotFoundError: return None
    return data

# --- Helper Function 2: CSV Reader for Tile Distribution ---
def read_tile_distribution_csv(path):
    """Reads the tile distribution CSV and creates a tile_stats dictionary."""
    tile_stats = {}
    try:
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    tile_id = int(row['Tile_ID'].strip())
                    count = int(row['Gaussian_Count'].strip())
                    tile_stats[tile_id] = {"total": count, "rendered": count}
                except (ValueError, KeyError, TypeError): continue
    except FileNotFoundError: return None
    return tile_stats if tile_stats else None

# Helper Function 3: Breakdown 데이터 CSV 저장 함수
def save_breakdown_to_csv(data, output_path):
    """Saves a list of dictionaries (breakdown data) to a CSV file."""
    if not data:
        print("  - Warning: No breakdown data to save to CSV.")
        return
    try:
        headers = data[0].keys()
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data)
    except (IOError, IndexError) as e:
        print(f"  - Warning: Could not write breakdown CSV to {output_path}. Error: {e}")


# --- Core Logic: 모든 세부 지표 계산 ---
def _calculate_performance_for_counts(op_counts, tile_stats, config):
    """Calculates all detailed performance metrics and returns them in a single dictionary."""
    hw, gs, costs = config['hardware_architecture'], config['simulation_global_settings'], config['cycle_costs']
    clk_hz = gs.get('clock_frequency_ghz', 1.0) * 1.0e9
    ccu_config = hw['ccu_config']; num_ccu_pipes = ccu_config.get('num_parallel_ccu_pipelines', ccu_config.get('num_pipelines', 1))
    gsu_config = hw['gsu_config']; vru_config = hw['vru_config']
    num_gsu_vru_pipes = gsu_config.get('num_pipelines', hw.get('num_parallel_pipelines', 1))
    mem_config = hw['memory_config']
    bytes_per_cycle = (mem_config['dram_bandwidth_gb_per_sec'] * 1e9) / clk_hz if clk_hz > 0 else 0
    num_vr_cores = vru_config.get('num_vr_cores_x', 1) * vru_config.get('num_vr_cores_y', 1)
    
    input_gaussians_count = op_counts.get('ccu_input_gaussians', 0)
    processed_gaussians_count = op_counts.get('ccu_visible_gaussians', 0)
    total_gaussians_for_sorting = op_counts.get('gsu_total_gaussians_sorted', 0)

    cull_w = op_counts.get('ccu_frustum_culling_check', 0) * costs['ccu']['frustum_culling_check']
    cov_w = op_counts.get('ccu_covariance_computation', 0) * costs['ccu']['ccu_covariance_computation']
    sh_w = op_counts.get('ccu_sh_evaluation', 0) * costs['ccu']['ccu_sh_evaluation']
    tile_isect_w = op_counts.get('ccu_tile_intersection_ops', 0) * costs['ccu'].get('aabb_intersect_check_per_tile', 1)
    subtile_gen_w = op_counts.get('ccu_subtile_generation_ops', 0) * costs['ccu'].get('subtile_bitmap_calc_per_gaussian', 1)
    
    ccu_compute_breakdown = {
        "Frustum Culling": cull_w / num_ccu_pipes if num_ccu_pipes > 0 else 0,
        "Covariance Comp.": cov_w / num_ccu_pipes if num_ccu_pipes > 0 else 0,
        "SH Evaluation": sh_w / num_ccu_pipes if num_ccu_pipes > 0 else 0,
        "Tile Intersection": tile_isect_w / num_ccu_pipes if num_ccu_pipes > 0 else 0,
        "Subtile Gen.": subtile_gen_w / num_ccu_pipes if num_ccu_pipes > 0 else 0
    }
    ccu_compute_latency = max(ccu_compute_breakdown.values()) if ccu_compute_breakdown else 0
    
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
    
    bytes_per_sort_entry = mem_config.get('bytes_per_sort_entry', 6)
    gsu_metadata_bytes = total_gaussians_for_sorting * bytes_per_sort_entry
    vru_feature_bytes = processed_gaussians_count * mem_config.get('bytes_per_processed_gaussian', 72)
    ccu_total_output_bytes = gsu_metadata_bytes + vru_feature_bytes
    ccu_write_latency = ccu_total_output_bytes / bytes_per_cycle if bytes_per_cycle > 0 else 0
    
    ccu_latencies = {"Compute": ccu_compute_latency, "Memory Load (In)": mem_load_latency, "Memory Write (Out)": ccu_write_latency}
    ccu_stage_latency = max(ccu_latencies.values())
    ccu_bottleneck_source = max(ccu_latencies, key=ccu_latencies.get)

    gsu_costs, vru_costs = costs['gsu'], costs['vru']
    gfeat_buffer_size = vru_config['gfeat_buffer_size_gaussians']
    
    tile_width = vru_config.get('num_vr_cores_x', 1) * vru_config.get('pixels_per_vr_core_x', 1)
    tile_height = vru_config.get('num_vr_cores_y', 1) * vru_config.get('pixels_per_vr_core_y', 1)
    pixels_per_tile = tile_width * tile_height
    avg_contrib_for_frame = op_counts.get('avg_max_contrib', 0)

    total_gsu_vru_workload_per_pipe = [0] * num_gsu_vru_pipes
    tile_ids = sorted(list(tile_stats.keys()))
    total_gsu_approx_workload, total_gsu_precise_workload, total_vru_parallel_workload = 0, 0, 0
    
    for i, tile_id in enumerate(tile_ids):
        pipe_idx = i % num_gsu_vru_pipes
        stats = tile_stats[tile_id]
        total_gaussians_in_tile, rendered_gaussians_in_tile = stats.get('total', 0), stats.get('rendered', 0)
        if rendered_gaussians_in_tile == 0: continue
        
        vru_alpha_ops_for_tile = avg_contrib_for_frame * pixels_per_tile
        vru_serial_work_for_tile = vru_alpha_ops_for_tile * vru_costs['alpha_computation_per_pixel']
        approx_sort_latency = 0
        if total_gaussians_in_tile > gfeat_buffer_size:
            approx_sort_latency = total_gaussians_in_tile * gsu_costs['qsu_op_per_gaussian']
        
        num_chunks = math.ceil(rendered_gaussians_in_tile / gfeat_buffer_size)
        remaining_gaussians = rendered_gaussians_in_tile
        precise_sort_work_per_chunk, vru_work_per_chunk_parallel = [], []
        
        for _ in range(num_chunks):
            chunk_size = min(gfeat_buffer_size, remaining_gaussians)
            if chunk_size <= 0: break
            
            work = math.ceil(chunk_size / gsu_config['bsu_channels']) * gsu_costs['bsu_op_per_16_gaussians']
            precise_sort_work_per_chunk.append(work)
            
            vru_work_fraction = chunk_size / rendered_gaussians_in_tile if rendered_gaussians_in_tile > 0 else 0
            vru_serial_work_for_chunk = vru_serial_work_for_tile * vru_work_fraction
            
            vru_work_per_chunk_parallel.append(vru_serial_work_for_chunk / num_vr_cores if num_vr_cores > 0 else 0)
            remaining_gaussians -= chunk_size
            
        if not precise_sort_work_per_chunk: continue
        
        first_chunk_gsu_time = precise_sort_work_per_chunk[0]
        latency_for_tile = approx_sort_latency + first_chunk_gsu_time + sum(vru_work_per_chunk_parallel)
        
        total_gsu_vru_workload_per_pipe[pipe_idx] += latency_for_tile
        total_gsu_approx_workload += approx_sort_latency
        total_gsu_precise_workload += sum(precise_sort_work_per_chunk)
        total_vru_parallel_workload += sum(vru_work_per_chunk_parallel)
    
    max_pipe_latency = max(total_gsu_vru_workload_per_pipe) if total_gsu_vru_workload_per_pipe else 0
    avg_pipe_latency = np.mean(total_gsu_vru_workload_per_pipe) if total_gsu_vru_workload_per_pipe else 0
    gsu_vru_compute_latency = max_pipe_latency
    
    gsu_mem_latency = gsu_metadata_bytes / bytes_per_cycle if bytes_per_cycle > 0 else 0
    vru_mem_latency = vru_feature_bytes / bytes_per_cycle if bytes_per_cycle > 0 else 0
    mem_transfer_latency = gsu_mem_latency + vru_mem_latency
    
    gsu_vru_stage_latency = max(gsu_vru_compute_latency, mem_transfer_latency)
    gsu_vru_bottleneck_source = "Compute" if gsu_vru_compute_latency > mem_transfer_latency else "Memory"
    
    total_system_latency = ccu_stage_latency + gsu_vru_stage_latency
    fps = clk_hz / total_system_latency if total_system_latency > 0 else 0
    
    return {
        "fps": fps, "time_per_frame_ms": 1000/fps if fps > 0 else 0, "total_latency": total_system_latency,
        "ccu_stage_latency": ccu_stage_latency, "ccu_bottleneck_source": ccu_bottleneck_source,
        "ccu_compute_breakdown": ccu_compute_breakdown, "ccu_compute_latency": ccu_compute_latency,
        "ccu_mem_load_latency": mem_load_latency, "ccu_write_latency": ccu_write_latency,
        "raw_bytes": raw_bytes, "ccu_total_output_bytes": ccu_total_output_bytes,
        "gsu_vru_stage_latency": gsu_vru_stage_latency, "gsu_vru_bottleneck_source": gsu_vru_bottleneck_source,
        "gsu_vru_compute_latency": gsu_vru_compute_latency,
        "mem_transfer_latency": mem_transfer_latency,
        "gsu_read_latency": gsu_mem_latency, "vru_read_latency": vru_mem_latency,
        "gsu_metadata_bytes": gsu_metadata_bytes, "vru_feature_bytes": vru_feature_bytes,
        "avg_pipe_latency": avg_pipe_latency, "max_pipe_latency": max_pipe_latency,
        "gsu_approx_workload_avg": total_gsu_approx_workload / num_gsu_vru_pipes if num_gsu_vru_pipes > 0 else 0,
        "gsu_precise_workload_avg": total_gsu_precise_workload / num_gsu_vru_pipes if num_gsu_vru_pipes > 0 else 0,
        "vru_rasterize_workload_avg": total_vru_parallel_workload / num_gsu_vru_pipes if num_gsu_vru_pipes > 0 else 0
    }

def save_report_to_json(frame_idx, performance_stats, op_counts, config, output_dir="reports"):
    os.makedirs(output_dir, exist_ok=True)
    report_data = {
        "frame_index": frame_idx,
        "performance_summary": performance_stats,
        "operation_counts": op_counts,
        "config_used": { "hardware_architecture": config["hardware_architecture"], "cycle_costs": config["cycle_costs"] }
    }
    report_filename = os.path.join(output_dir, f"report_frame_{frame_idx:05d}.json")
    with open(report_filename, 'w') as f:
        json.dump(report_data, f, indent=4, sort_keys=True)

def generate_full_report_text(frame_idx, stats, op_counts, total_rendered, config):
    lines = []
    lines.append("\n" + "="*85)
    lines.append(f"Analysis Report for Frame: {frame_idx}")
    lines.append("\n[ Performance Summary ]")
    lines.append(f"- Estimated FPS: {stats.get('fps', 0):.2f}")
    lines.append(f"- Time per Frame: {stats.get('time_per_frame_ms', 0):.2f} ms")
    lines.append(f"- Total System Latency: {int(stats.get('total_latency', 0)):,} cycles ({config['simulation_global_settings']['clock_frequency_ghz']} GHz clock)")
    lines.append("\n[ System-Level Pipeline Breakdown ]")
    lines.append(f"1. CCU Stage Latency:        {int(stats.get('ccu_stage_latency', 0)):>12,} cycles (Bottleneck: {stats.get('ccu_bottleneck_source', 'N/A')})")
    lines.append(f"2. GSU-VRU Stage Latency:   {int(stats.get('gsu_vru_stage_latency', 0)):>12,} cycles (Bottleneck: {stats.get('gsu_vru_bottleneck_source', 'N/A')})")
    lines.append("\n[ CCU Detailed Analysis ]")
    lines.append(f"- Total Parallel Pipelines: {config['hardware_architecture']['ccu_config'].get('num_parallel_ccu_pipelines', 1)}")
    lines.append(f"- Data & Operation Counts:")
    lines.append(f"  - Input Raw Gaussians:        {op_counts.get('ccu_input_gaussians', 0):>15,}")
    lines.append(f"  - Output Processed Gaussians: {op_counts.get('ccu_visible_gaussians', 0):>15,}")
    lines.append(f"- Compute Latency Breakdown (per pipeline):")
    ccu_breakdown = stats.get('ccu_compute_breakdown', {})
    if ccu_breakdown:
        bottleneck_task = max(ccu_breakdown, key=ccu_breakdown.get)
        for name, latency in ccu_breakdown.items():
            is_bottleneck = " (Bottleneck)" if name == bottleneck_task else ""
            lines.append(f"  - {name:<20}: {int(latency):>12,} cycles{is_bottleneck}")
    lines.append(f"- Total Stage Latency & Data Breakdown:")
    lines.append(f"  - Final Compute Latency:   {int(stats.get('ccu_compute_latency', 0)):>12,} cycles")
    lines.append(f"  - Memory Load (In):          {int(stats.get('ccu_mem_load_latency', 0)):>12,} cycles ({stats.get('raw_bytes',0)/1e6:.2f} MB)")
    lines.append(f"  - Memory Write (Out):        {int(stats.get('ccu_write_latency', 0)):>12,} cycles ({stats.get('ccu_total_output_bytes',0)/1e6:.2f} MB Total)")
    lines.append(f"    > For GSU (Sort Keys):     {stats.get('gsu_metadata_bytes',0)/1e6:>12.2f} MB")
    lines.append(f"    > For VRU (Render Data):   {stats.get('vru_feature_bytes',0)/1e6:>12.2f} MB")
    lines.append("\n[ GSU-VRU Detailed Analysis ]")
    lines.append(f"- Total Parallel Pipelines: {config['hardware_architecture']['gsu_config'].get('num_pipelines', 1)}")
    lines.append(f"- Input Data & Operation Counts:")
    lines.append(f"  - Gaussians for Sorting (total):   {op_counts.get('gsu_total_gaussians_sorted', 0):>10,}")
    lines.append(f"  - Gaussians for Rendering (proc.): {op_counts.get('ccu_visible_gaussians', 0):>10,}")
    lines.append(f"  - Total Rendered in Tiles:         {total_rendered:>10,}")
    lines.append(f"  - Avg Pixel Contributors:          {op_counts.get('avg_max_contrib', 0):>10,.2f}")
    lines.append(f"- Pipeline Load Balancing:")
    lines.append(f"  - Average Latency per Pipeline:  {int(stats.get('avg_pipe_latency', 0)):>12,} cycles")
    lines.append(f"  - Max Latency (Bottleneck Pipe): {int(stats.get('max_pipe_latency', 0)):>9,} cycles")
    lines.append(f"- Average Workload Breakdown (per pipeline, in cycles):")
    approx_avg = stats.get('gsu_approx_workload_avg', 0)
    precise_avg = stats.get('gsu_precise_workload_avg', 0)
    vru_avg = stats.get('vru_rasterize_workload_avg', 0)
    lines.append(f"  - Approx Sort (Overlapped):      {int(approx_avg):>9,}")
    lines.append(f"  - Precise Sort (Overlapped):     {int(precise_avg):>9,}")
    lines.append(f"  - VRU Rasterize (Overlapped):    {int(vru_avg):>9,} <- VR Cores Applied")
    workloads = {"Approx Sort": approx_avg, "Precise Sort": precise_avg, "VRU Rasterize": vru_avg}
    dominant_task = max(workloads, key=workloads.get)
    lines.append(f"  - Dominant Overlapped Task:        {dominant_task}")
    lines.append(f"- Final Compute Latency: {int(stats.get('gsu_vru_compute_latency', 0)):>18,} cycles")
    lines.append(f"- Memory Transfer Latency Breakdown:")
    gsu_bytes = stats.get('gsu_metadata_bytes', 0)
    vru_bytes = stats.get('vru_feature_bytes', 0)
    lines.append(f"  - GSU Read (Sort Keys):          {int(stats.get('gsu_read_latency', 0)):>9,} cycles ({gsu_bytes/1e6:.2f} MB)")
    lines.append(f"  - VRU Read (Render Data):        {int(stats.get('vru_read_latency', 0)):>9,} cycles ({vru_bytes/1e6:.2f} MB)")
    lines.append(f"  - Total Memory Latency:          {int(stats.get('mem_transfer_latency', 0)):>9,} cycles")
    lines.append("="*85)
    return "\n".join(lines)

# --- Main Execution Block ---
def main(log_scene_path, start_frame=None, end_frame=None):
    if not os.path.isdir(log_scene_path):
        print(f"Error: The specified scene log directory does not exist: {log_scene_path}")
        return

    print(f"[*] Analyzing logs from: {os.path.abspath(log_scene_path)}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))

    config_path = os.path.join(project_root, "src", "config.json")
    config_analyze_path = os.path.join(project_root, "src", "config_analyze.json")

    scene_name = os.path.basename(os.path.normpath(log_scene_path))
    report_base_dir = os.path.join(project_root, "report")
    scene_report_dir = os.path.join(report_base_dir, f"{scene_name}_reports")
    json_report_dir = os.path.join(scene_report_dir, "json_reports")
    
    os.makedirs(scene_report_dir, exist_ok=True)
    os.makedirs(json_report_dir, exist_ok=True)
    
    culling_log_path = os.path.join(log_scene_path, "culling_log.csv")
    contrib_log_dir = os.path.join(log_scene_path, "pixel_contrib_summary")
    tile_distrib_log_dir = os.path.join(log_scene_path, "unsorted_grouped") 
    
    try:
        with open(config_path, 'r') as f: config = json.load(f)
        with open(config_analyze_path, 'r') as f: config_analyze = json.load(f)
        for key, value in config_analyze.items():
            if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                config[key].update(value)
            else: config[key] = value
    except FileNotFoundError as e:
        print(f"Error: Config file not found. Ensure config.json and config_analyze.json are in the 'src' directory: {e.filename}"); return
        
    culling_data = read_csv_to_list(culling_log_path)
    if not culling_data:
        print(f"Error: Main culling log not found or empty at {culling_log_path}. Exiting."); return

    if start_frame is None and end_frame is None: title = "Analyzing All Frames"
    else: title = f"Analyzing Frames from {start_frame or 0} to {end_frame or 'End'}"
    print("\n" + "="*85 + f"\n GSCore Performance Estimator ({title})\n" + "="*85)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_log_filename = os.path.join(scene_report_dir, f"analysis_report_{scene_name}_{timestamp}.txt")
    
    breakdown_csv_filename = os.path.join(scene_report_dir, f"performance_breakdown_{scene_name}_{timestamp}.csv")
    
    with open(summary_log_filename, 'w') as f:
        f.write(f"GSCore Performance Analysis Report - {datetime.now()}\n")
        f.write(f"Analysis Range: {title}\n")
        f.write(f"Log Source: {os.path.abspath(log_scene_path)}\n")

    all_frame_stats = []
    breakdown_data_for_csv = [] 

    for frame_data in culling_data:
        frame_idx = frame_data['View_Index']
        if start_frame is not None and frame_idx < start_frame: continue
        if end_frame is not None and frame_idx > end_frame: continue
        
        contrib_filename = f"contrib_summary_{frame_idx:05d}.csv"
        tile_distrib_filename = f"unsorted_{frame_idx:05d}.csv"
        contrib_log_path = os.path.join(contrib_log_dir, contrib_filename)
        tile_distrib_log_path = os.path.join(tile_distrib_log_dir, tile_distrib_filename)
        
        contrib_data = read_csv_to_list(contrib_log_path)
        tile_stats_data = read_tile_distribution_csv(tile_distrib_log_path)
        
        if not contrib_data or not tile_stats_data:
            print(f"  - Warning: Skipping frame {frame_idx} due to missing log files.")
            continue
        
        op_counts = {
            'ccu_input_gaussians': frame_data['Total_Gaussians'],
            'ccu_visible_gaussians': frame_data['Gaussians_In_View'],
            'ccu_frustum_culling_check': frame_data['Total_Gaussians'],
            'ccu_covariance_computation': frame_data['Gaussians_In_View'],
            'ccu_sh_evaluation': frame_data['Gaussians_In_View'],
            'ccu_tile_intersection_ops': int(frame_data['Gaussians_In_View'] * 2.5),
            'ccu_subtile_generation_ops': int(frame_data['Gaussians_In_View'] * 2.5 * 4.0),
            'gsu_total_gaussians_sorted': sum(stats.get('total', 0) for stats in tile_stats_data.values()),
            'avg_max_contrib': np.mean([row['Max_Contributor_Count'] for row in contrib_data])
        }
        
        frame_stats = _calculate_performance_for_counts(op_counts, tile_stats_data, config)
        all_frame_stats.append((frame_idx, frame_stats))
        
        # [수정] 그래프용 CSV 데이터를 GSU와 VRU 세부 분석용으로 재구성
        breakdown_row = {
            'frame_idx': frame_idx,
            'fps': round(frame_stats.get('fps', 0), 2),
            'total_latency_cycles': int(frame_stats.get('total_latency', 0)),
            
            # Stage Latencies
            'ccu_stage_latency': int(frame_stats.get('ccu_stage_latency', 0)),
            'gsu_vru_stage_latency': int(frame_stats.get('gsu_vru_stage_latency', 0)),

            # CCU Stage Bottleneck Components
            'ccu_compute_latency': int(frame_stats.get('ccu_compute_latency', 0)),
            'ccu_mem_load_latency': int(frame_stats.get('ccu_mem_load_latency', 0)),
            'ccu_mem_write_latency': int(frame_stats.get('ccu_write_latency', 0)),

            # GSU-VRU Stage Bottleneck Components
            'gsu_vru_compute_latency': int(frame_stats.get('gsu_vru_compute_latency', 0)),
            'gsu_mem_read_latency': int(frame_stats.get('gsu_read_latency', 0)),
            'vru_mem_read_latency': int(frame_stats.get('vru_read_latency', 0)),
            
            # GSU vs VRU Average Compute Workload Breakdown (per pipeline)
            'gsu_approx_sort_avg_cycles': int(frame_stats.get('gsu_approx_workload_avg', 0)),
            'gsu_precise_sort_avg_cycles': int(frame_stats.get('gsu_precise_workload_avg', 0)),
            'vru_rasterize_avg_cycles': int(frame_stats.get('vru_rasterize_workload_avg', 0)),
        }
        breakdown_data_for_csv.append(breakdown_row)

        save_report_to_json(frame_idx, frame_stats, op_counts, config, output_dir=json_report_dir)

        total_rendered = sum(s.get('rendered', 0) for s in tile_stats_data.values())
        report_text = generate_full_report_text(frame_idx, frame_stats, op_counts, total_rendered, config)
        with open(summary_log_filename, 'a') as f:
            f.write(report_text)

    if breakdown_data_for_csv:
        save_breakdown_to_csv(breakdown_data_for_csv, breakdown_csv_filename)

    print("\n[ Console Performance Summary ]")
    if not all_frame_stats:
        print("  - No frames were successfully analyzed within the specified range.")
    else:
        for frame_idx, stats in all_frame_stats:
            print(f"  - Frame {frame_idx:<4}: FPS: {stats['fps']:<8.2f} Total Latency: {int(stats['total_latency']):>12,} cycles")
    if len(all_frame_stats) > 1:
        avg_fps = np.mean([s['fps'] for _, s in all_frame_stats])
        print(f"\n[ Average FPS Over Analyzed Sequence: {avg_fps:.2f} ]")
    
    print(f"\n[+] Comprehensive analysis report saved to: {summary_log_filename}")
    print(f"[+] Raw data for each frame saved in '{json_report_dir}/' directory.")
    if breakdown_data_for_csv:
        print(f"[+] Performance breakdown for graphing saved to: {breakdown_csv_filename}")
    print("\n" + "="*85)

# --- 스크립트 실행 부분 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GSCore Performance Estimator based on CSV logs.")
    parser.add_argument("-p", "--path", type=str, required=True, 
                        help="Path to the specific scene log directory (e.g., 'log/flame_steak_cu'). This argument is required.")
    parser.add_argument("-s", "--start", type=int, default=None, 
                        help="The starting frame number to analyze (inclusive).")
    parser.add_argument("-e", "--end", type=int, default=None, 
                        help="The ending frame number to analyze (inclusive).")
    args = parser.parse_args()
    
    main(log_scene_path=args.path, start_frame=args.start, end_frame=args.end)

