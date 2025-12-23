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
    from sorting_algorithms_dram import CostTracker, incremental_sort_merge, get_id
except ImportError:
    print("Error: 'sorting_algorithms_dram.py' not found."); exit()

def read_csv_to_list(path):
    data = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cleaned_row = {k.strip(): v.strip() for k, v in row.items()}
                for k, v in cleaned_row.items():
                    try: cleaned_row[k] = int(v)
                    except (ValueError, TypeError):
                        try: cleaned_row[k] = float(v)
                        except (ValueError, TypeError): pass
                data.append(cleaned_row)
    except FileNotFoundError: return None
    return data

def read_and_parse_tile_csv(path):
    _, per_tile_data = {}, defaultdict(list)
    if not path or not os.path.exists(path): return None, None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    tile_id, count = int(row['Tile_ID']), int(row['Gaussian_Count'])
                    if count > 0 and 'Gaussian_IDs' in row and 'Depths' in row:
                        ids_str, depths_str = row.get('Gaussian_IDs', '').strip(), row.get('Depths', '').strip()
                        if ids_str and depths_str:
                            ids = [int(i) for i in ids_str.split(' ')]
                            depths = [float(d) for d in depths_str.split(' ')]
                            if len(ids) == len(depths): per_tile_data[tile_id] = list(zip(ids, depths))
                except (ValueError, KeyError, TypeError): continue
    except FileNotFoundError: return None, None
    return _, per_tile_data

def save_list_of_dicts_to_csv(data, file_path):
    if not data: return
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader(); writer.writerows(data)
    print(f"[+] Detailed data for analysis saved to: {file_path}")

def bytes_to_cycles(num_bytes, bytes_per_cycle):
    return math.ceil(num_bytes / bytes_per_cycle) if bytes_per_cycle > 0 else 0

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
    
    input_gaussians = op_counts.get('ccu_input_gaussians', 0)
    visible_gaussians = op_counts.get('ccu_visible_gaussians', 0)
    gsu_gaussians = op_counts.get('gsu_total_gaussians_sorted', 0)
    
    cull_work = input_gaussians * costs['ccu'].get('frustum_culling_check', 0)
    cov_work = visible_gaussians * costs['ccu'].get('ccu_covariance_computation', 0)
    sh_work = visible_gaussians * costs['ccu'].get('ccu_sh_evaluation', 0)
    ccu_compute_latency = max(cull_work, cov_work, sh_work) / num_ccu_pipes if num_ccu_pipes > 0 else 0

    dram_read_ccu_bytes = input_gaussians * mem_config.get('bytes_per_raw_gaussian', 59)
    dram_write_gsu_bytes = gsu_gaussians * mem_config.get('bytes_per_sort_entry', 6)
    
    num_vru_params = 9
    bytes_per_vru_param = 2 # 16-bit
    bytes_per_processed_gaussian = num_vru_params * bytes_per_vru_param # 9 * 2 = 18 Bytes
    dram_write_vru_bytes = visible_gaussians * bytes_per_processed_gaussian
    
    mem_load_latency = bytes_to_cycles(dram_read_ccu_bytes / num_ccu_pipes, bytes_per_cycle) if num_ccu_pipes > 0 else 0
    ccu_write_latency = bytes_to_cycles((dram_write_gsu_bytes + dram_write_vru_bytes) / num_ccu_pipes, bytes_per_cycle) if num_ccu_pipes > 0 else 0
    
    ccu_latencies = {"Compute": ccu_compute_latency, "Memory Load (In)": mem_load_latency, "Memory Write (Out)": ccu_write_latency}
    ccu_stage_latency = max(ccu_latencies.values())
    ccu_bottleneck_source = max(ccu_latencies, key=ccu_latencies.get)
    
    total_system_latency = ccu_stage_latency + (gsu_vru_latency_override or 0)
    fps = clk_hz / total_system_latency if total_system_latency > 0 else 0
    
    return {"fps": fps, "time_per_frame_ms": 1000/fps if fps > 0 else 0, 
            "total_latency": total_system_latency, "ccu_stage_latency": ccu_stage_latency, 
            "ccu_bottleneck_source": ccu_bottleneck_source, "gsu_vru_stage_latency": gsu_vru_latency_override,
            "dram_read_ccu_bytes": dram_read_ccu_bytes, "dram_write_gsu_bytes": dram_write_gsu_bytes, 
            "dram_write_vru_bytes": dram_write_vru_bytes,
            "ccu_compute_latency": ccu_compute_latency, "ccu_mem_load_latency": mem_load_latency,
            "ccu_mem_write_latency": ccu_write_latency}

def _calculate_pipelined_latency(cost_bd, vru_latency_for_tile):
    method_name = cost_bd.get('method_name', 'Full Sort (GSCore)')
    if method_name == "Full Sort (GSCore)":
        init_latency = cost_bd.get('approx_sort_cost', 0)
        num_chunks = cost_bd.get('num_chunks', 0)
        if num_chunks == 0: return init_latency, "N/A"
        sort_per_chunk = cost_bd.get('precise_sort_cost_per_chunk', 0)
        raster_per_chunk = vru_latency_for_tile / num_chunks if num_chunks > 0 else 0
        
        if num_chunks > 0:
            pipeline_time = min(sort_per_chunk, raster_per_chunk) + max(sort_per_chunk, raster_per_chunk) * num_chunks
        else:
            pipeline_time = 0
            
        return init_latency + pipeline_time, "Sorting (Precise)" if sort_per_chunk > raster_per_chunk else "Rasterization"
    else:
        init_latency = cost_bd['separation'].final_cycles + cost_bd['sort_new'].final_cycles
        pipeline_time = max(cost_bd['merge'].final_cycles, vru_latency_for_tile)
        return init_latency + pipeline_time, "Sorting (Merge)" if cost_bd['merge'].final_cycles > vru_latency_for_tile else "Rasterization"

def generate_full_report_text(frame_idx, stats, sort_stats):
    lines = [f"\n{'='*85}", f"Analysis Report for Frame: {frame_idx}",
             "\n[ Performance Summary (Proposed, Pipelined) ]",
             f"- Estimated FPS: {stats.get('fps_pipelined', 0):.2f}",
             f"- Total System Latency: {int(stats.get('total_latency_pipelined', 0)):,} cycles",
             "\n[ System-Level Breakdown (Pipelined) ]",
             f"1. CCU Stage Latency:       {int(stats.get('ccu_stage_latency', 0)):>15,} cycles (Bottleneck: {stats.get('ccu_bottleneck_source', 'N/A')})",
             f"2. GSU-VRU Stage Latency:   {int(stats.get('gsu_vru_stage_latency_pipelined', 0)):>15,} cycles",
             f"\n[ GSU DRAM Read (Bytes per Frame) ]",
             f"- Incremental Method : {int(sort_stats.get('dram_read_gsu_incr_bytes', 0)):>15,} bytes",
             f"- Baseline Method    : {int(sort_stats.get('dram_read_gsu_base_bytes', 0)):>15,} bytes"]
    lines.append(f"{'='*85}")
    return "\n".join(lines)

def process_frame(frame_idx, frame_data, config, log_dirs, start_frame, hash_sim_mode):
    _, current_tile_data = read_and_parse_tile_csv(os.path.join(log_dirs['current_tile'], f"unsorted_{frame_idx:05d}.csv"))
    if not current_tile_data: return None

    is_first_frame = (frame_idx == start_frame)
    _, prev_tile_data = (None, None) if is_first_frame else read_and_parse_tile_csv(os.path.join(log_dirs['prev_tile'], f"sorted_{frame_idx-1:05d}.csv"))
    
    contrib_data = read_csv_to_list(os.path.join(log_dirs['contrib'], f"contrib_summary_{frame_idx:05d}.csv"))
    avg_max_contrib = np.mean([row['Max_Contributor_Count'] for row in contrib_data]) if contrib_data else 0
    
    bytes_per_sort_entry = config['hardware_architecture']['memory_config'].get('bytes_per_sort_entry', 6)
    
    gsu_config = config['hardware_architecture']['gsu_config']
    screen_width = config['simulation_global_settings'].get('render_width_pixels', 1920)
    screen_height = config['simulation_global_settings'].get('render_height_pixels', 1080)
    tile_width = gsu_config.get('tile_size_x', 16); tile_height = gsu_config.get('tile_size_y', 16)
    group_dim_x = gsu_config.get('group_processing_dim_x', 2); group_dim_y = gsu_config.get('group_processing_dim_y', 2)
    tiles_x = math.ceil(screen_width / tile_width); tiles_y = math.ceil(screen_height / tile_height)
    
    all_tile_ids_present = set(current_tile_data.keys())
    tile_groups = [[ty * tiles_x + tx for j in range(group_dim_y) for i in range(group_dim_x) if (tx := group_i + i) < tiles_x and (ty := group_j + j) < tiles_y and (ty * tiles_x + tx) in all_tile_ids_present] for group_j in range(0, tiles_y, group_dim_y) for group_i in range(0, tiles_x, group_dim_x)]
    tile_groups = [group for group in tile_groups if group]
    
    incr_seq_l, incr_pipe_l, base_seq_l, base_pipe_l = [], [], [], []
    
    # [수정] GSU SRAM 비트 및 DRAM 인덱스 카운터 추가
    incr_dram_b, base_dram_b, incr_sram_b, base_sram_b, still_g, new_g = 0, 0, 0, 0, 0, 0
    incr_dram_read_indices, incr_dram_write_indices = 0, 0
    base_dram_read_indices, base_dram_write_indices = 0, 0
    
    incr_breakdown, base_breakdown = defaultdict(int), defaultdict(int)

    for group in tile_groups:
        if not is_first_frame and prev_tile_data:
            unique_gids = {get_id(item) for tid in group for item in current_tile_data.get(tid, [])}
            unique_gids.update({get_id(item) for tid in group for item in prev_tile_data.get(tid, [])})

            additional_bytes_per_entry = 0.5 
            read_multiplier = 2
            cost_per_unique_gid = (bytes_per_sort_entry + additional_bytes_per_entry) * read_multiplier
            incr_dram_b += len(unique_gids) * cost_per_unique_gid
        else:
            incr_dram_b += sum(len(current_tile_data.get(tid, [])) for tid in group) * bytes_per_sort_entry
        
        g_incr_seq, g_incr_pipe, g_base_seq, g_base_pipe = [],[],[],[]
        g_incr_details, g_base_details = [], []
        for tid in group:
            curr = current_tile_data.get(tid, []); prev = [] if is_first_frame or not prev_tile_data else prev_tile_data.get(tid, [])
            vru_lat = _calculate_vru_latency_for_tile(len(curr), avg_max_contrib, config)
            
            if not is_first_frame:
                p_ids = {get_id(i) for i in prev}; c_ids = {get_id(i) for i in curr}
                new_g += len(c_ids - p_ids); still_g += len(c_ids & p_ids)

            _, bd_i = incremental_sort_merge(prev, curr, config)
            # [수정] Incremental SRAM 비트 및 DRAM 인덱스 합산
            incr_sram_b += bd_i.get('gsu_sram_bits_total', 0)
            incr_dram_read_indices += bd_i.get('dram_read_indices', 0)
            incr_dram_write_indices += bd_i.get('dram_write_indices', 0)
            
            g_incr_seq.append(bd_i['total'].compute_cycles + vru_lat)
            pipe_i, _ = _calculate_pipelined_latency(bd_i, vru_lat); g_incr_pipe.append(pipe_i)
            g_incr_details.append({'sep': bd_i['separation'].final_cycles, 'sgs': bd_i['sort_new'].final_cycles, 
                                   'merge': bd_i['merge'].final_cycles, 'vru': vru_lat})

            _, bd_b = incremental_sort_merge([], curr, config)
            # [수정] Baseline SRAM 비트 및 DRAM 인덱스 합산
            base_sram_b += bd_b.get('gsu_sram_bits_total', 0)
            base_dram_read_indices += bd_b.get('dram_read_indices', 0)
            base_dram_write_indices += bd_b.get('dram_write_indices', 0)
            
            base_dram_b += bd_b.get('dram_bytes', 0)
            g_base_seq.append(bd_b['total'].final_cycles + vru_lat)
            pipe_b, _ = _calculate_pipelined_latency(bd_b, vru_lat); g_base_pipe.append(pipe_b)
            g_base_details.append({'gsu': bd_b['total'].final_cycles, 'vru': vru_lat})

        if g_incr_pipe:
            idx = np.argmax(g_incr_pipe); incr_pipe_l.append(g_incr_pipe[idx])
            for k,v in g_incr_details[idx].items(): incr_breakdown[k] += v
        if g_base_pipe:
            idx = np.argmax(g_base_pipe); base_pipe_l.append(g_base_pipe[idx])
            for k,v in g_base_details[idx].items(): base_breakdown[k] += v
        if g_incr_seq: incr_seq_l.append(max(g_incr_seq))
        if g_base_seq: base_seq_l.append(max(g_base_seq))

    incr_latency_pipelined = sum(incr_pipe_l)
    incr_latency_sequential = sum(incr_seq_l)
    base_latency_pipelined = sum(base_pipe_l)
    base_latency_sequential = sum(base_seq_l)
    
    total_gaussians_in_tiles = sum(len(v) for v in current_tile_data.values())
    op_counts = {'ccu_input_gaussians': frame_data.get('Total_Gaussians', 0), 'ccu_visible_gaussians': frame_data.get('Gaussians_In_View', 0),
                 'gsu_total_gaussians_sorted': total_gaussians_in_tiles}
    
    stats_pipe_incr = _calculate_performance_for_counts(op_counts, config, gsu_vru_latency_override=incr_latency_pipelined)
    stats_seq_incr = _calculate_performance_for_counts(op_counts, config, gsu_vru_latency_override=incr_latency_sequential)

    stats_for_report = {'fps_pipelined': stats_pipe_incr['fps'], 'total_latency_pipelined': stats_pipe_incr['total_latency'], 'ccu_stage_latency': stats_pipe_incr['ccu_stage_latency'], 'gsu_vru_stage_latency_pipelined': stats_pipe_incr['gsu_vru_stage_latency']}
    sort_stats_for_report = {"dram_read_gsu_incr_bytes": incr_dram_b, "dram_read_gsu_base_bytes": base_dram_b}
    report_text = generate_full_report_text(frame_idx, stats_for_report, sort_stats_for_report)
    
    gsu_incr_sep = int(incr_breakdown.get('sep', 0))
    gsu_incr_sgs = int(incr_breakdown.get('sgs', 0))
    gsu_incr_merge = int(incr_breakdown.get('merge', 0))
    vru_on_incr_lat = int(incr_breakdown.get('vru', 0))
    gsu_base_sort = int(base_breakdown.get('gsu', 0))
    vru_on_base_lat = int(base_breakdown.get('vru', 0))
    
    # [수정] CSV 저장을 위한 상세 데이터 구조체 (SRAM 비트 및 DRAM 인덱스 개수 추가)
    csv_row = {
        'frame_idx': frame_idx,
        'fps_pipelined': round(stats_pipe_incr['fps'], 2),
        'total_latency_pipelined': int(stats_pipe_incr['total_latency']),
        'total_latency_sequential': int(stats_seq_incr['total_latency']),
        'ccu_stage_latency': int(stats_pipe_incr['ccu_stage_latency']),
        
        'gsu_vru_incremental_latency_sequential': int(incr_latency_sequential),
        'gsu_vru_incremental_latency_pipelined': int(incr_latency_pipelined),
        'gsu_incremental_latency_sep': gsu_incr_sep,
        'gsu_incremental_latency_sgs': gsu_incr_sgs,
        'gsu_incremental_latency_merge': gsu_incr_merge,
        'vru_latency_on_incremental': vru_on_incr_lat,
        'gsu_incremental_latency_total': gsu_incr_sep + gsu_incr_sgs + gsu_incr_merge,
        
        'gsu_vru_baseline_latency_sequential': int(base_latency_sequential),
        'gsu_vru_baseline_latency_pipelined': int(base_latency_pipelined),
        'gsu_baseline_latency_total': gsu_base_sort,
        'vru_latency_on_baseline': vru_on_base_lat,
        
        'dram_read_ccu_bits': int(stats_pipe_incr['dram_read_ccu_bytes'] * 8),
        'dram_write_gsu_from_ccu_bits': int(stats_pipe_incr['dram_write_gsu_bytes'] * 8),
        'dram_write_vru_from_ccu_bits': int(stats_pipe_incr['dram_write_vru_bytes'] * 8),
        'dram_read_gsu_baseline_bits': int(base_dram_b*8),
        'dram_read_gsu_incremental_bits': int(incr_dram_b*8),
        
        'gsu_sram_access_incremental_bits': int(incr_sram_b),
        'gsu_sram_access_baseline_bits': int(base_sram_b), 
        
        # ▼▼▼ [수정] DRAM 접근 인덱스 개수 추가 ▼▼▼
        'dram_read_indices_incremental': int(incr_dram_read_indices),
        'dram_write_indices_incremental': int(incr_dram_write_indices),
        'dram_read_indices_baseline': int(base_dram_read_indices),
        'dram_write_indices_baseline': int(base_dram_write_indices),
        # ▲▲▲ [수정] 완료 ▲▲▲
        
        'still_gaussians': still_g,
        'new_gaussians': new_g
    }
            
    return (frame_idx, report_text, csv_row)

def main(log_scene_path, start_frame=None, end_frame=None):
    if not os.path.isdir(log_scene_path): print(f"Error: Log directory does not exist: {log_scene_path}"); return
    print(f"[*] Analyzing logs from: {os.path.abspath(log_scene_path)}")
    scene_name = os.path.basename(os.path.normpath(log_scene_path)); timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    scene_report_dir = os.path.join("report", f"{scene_name}_reports_{timestamp}"); os.makedirs(scene_report_dir, exist_ok=True)
    log_dirs = {'contrib': os.path.join(log_scene_path, "pixel_contrib_summary"), 'current_tile': os.path.join(log_scene_path, "unsorted_with_depth"), 'prev_tile': os.path.join(log_scene_path, "sorted_with_depth")}
    
    try:
        # [수정] 병합된 config_merged.json을 사용하도록 변경
        config_path = "src/config_merged.json"
        with open(config_path, 'r', encoding='utf-8') as f: 
            config = json.load(f)
            
    except FileNotFoundError as e: 
        print(f"Error: Config file not found: {e.filename}"); return
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {config_path}")
        
    hash_sim_mode = "Detailed" if config.get('simulation_global_settings', {}).get('detailed_hash_sim', False) else "Fast"
    culling_data = read_csv_to_list(os.path.join(log_scene_path, "culling_log.csv"))
    if not culling_data: print(f"Error: Main culling log not found"); return

    start_frame = start_frame if start_frame is not None else culling_data[0].get('View_Index', 0)
    end_frame = end_frame if end_frame is not None else culling_data[-1].get('View_Index', 0)
    title = f"Analyzing Frames {start_frame} to {end_frame}"
    print(f"\n{'='*85}\n GSCore Performance Estimator ({title})\n{'='*85}")

    summary_log_filename = os.path.join(scene_report_dir, "analysis_summary.txt")
    csv_output_path = os.path.join(scene_report_dir, "performance_details_over_frames.csv")
    with open(summary_log_filename, 'w') as f: f.write(f"GSCore Analysis - {datetime.now()}\nRange: {title}\nSource: {os.path.abspath(log_scene_path)}\nHash Simulation Mode: {hash_sim_mode}\n")

    tasks = [(fd['View_Index'], fd, config, log_dirs, start_frame, hash_sim_mode) for fd in culling_data if start_frame <= fd['View_Index'] <= end_frame]
    all_csv_data = []
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_frame, *task): task for task in tasks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc="Processing Frames"):
            try:
                if (result := future.result()):
                    frame_idx, report_text, csv_row = result; all_csv_data.append(csv_row)
                    with open(summary_log_filename, 'a') as f: f.write(report_text)
            except Exception as e:
                import traceback; traceback.print_exc(); print(f'\nFrame {futures[future][0]} generated an exception: {e}')

    all_csv_data.sort(key=lambda x: x['frame_idx'])
    save_list_of_dicts_to_csv(all_csv_data, csv_output_path)
    
    if all_csv_data:
        avg_fps = sum(r['fps_pipelined'] for r in all_csv_data) / len(all_csv_data)
        total_incr_dram = sum(r['dram_read_gsu_incremental_bits'] for r in all_csv_data)
        total_base_dram = sum(r['dram_read_gsu_baseline_bits'] for r in all_csv_data)
        
        print("\n[ Quick Summary (from estimator) ]")
        print(f"- Average FPS (Pipelined, Incremental): {avg_fps:.2f}")
        if total_base_dram > 0:
            savings = (1 - total_incr_dram / total_base_dram) * 100
            print(f"- Total GSU DRAM Read Savings: {savings:.2f}%")
            
    print(f"\n[+] Full text report saved to: {summary_log_filename}")
    print("\n" + "="*85)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GSCore Performance Estimator with parallel processing.")
    parser.add_argument("-p", "--path", type=str, required=True, help="Path to the scene log directory.")
    parser.add_argument("-s", "--start", type=int, help="Starting frame number.")
    parser.add_argument("-e", "--end", type=int, help="Ending frame number.")
    args = parser.parse_args()
    main(log_scene_path=args.path, start_frame=args.start, end_frame=args.end)