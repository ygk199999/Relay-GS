import math
from collections import defaultdict

class CostTracker:
    """Compute와 Memory 사이클을 별도로 추적하고 병목을 기반으로 최종 사이클을 결정합니다."""
    def __init__(self, compute_cycles=0, memory_cycles=0, dram_bytes=0):
        self.compute_cycles = compute_cycles
        self.memory_cycles = memory_cycles
        self.dram_bytes = dram_bytes
        self.final_cycles = max(compute_cycles, memory_cycles)

    def add_compute_cycles(self, count=1):
        self.compute_cycles += count
        self.final_cycles = max(self.compute_cycles, self.memory_cycles)

    def add_memory_cycles(self, count=1, dram_bytes_added=0):
        self.memory_cycles += count
        self.dram_bytes += dram_bytes_added
        self.final_cycles = max(self.compute_cycles, self.memory_cycles)

    def __repr__(self):
        return f"Cost(Final={self.final_cycles}, Comp={self.compute_cycles}, Mem={self.memory_cycles}, DRAM={self.dram_bytes})"

def get_id(item): return item[0]
def get_depth(item): return item[1]

def bytes_to_cycles(num_bytes, mem_config, config):
    clk_ghz = config['simulation_global_settings'].get('clock_frequency_ghz', 1.0)
    dram_bw_gb_s = mem_config.get('dram_bandwidth_gb_per_sec', 32)
    bytes_per_cycle = (dram_bw_gb_s * 1e9) / (clk_ghz * 1e9) if clk_ghz > 0 else 0
    return math.ceil(num_bytes / bytes_per_cycle) if bytes_per_cycle > 0 else 0


# ==============================================================================
# === GSCore 2-Stage 정렬 모델 (DRAM 바이트 계산 포함) ===
# ==============================================================================
def full_sort(list_to_sort, config):
    cost = CostTracker()
    num_elements = len(list_to_sort)
    
    if num_elements == 0: 
        sorted_list = []
    else:
        sorted_list = sorted(list_to_sort, key=lambda item: get_depth(item))
    
    gsu_costs_config = config['cycle_costs']['gsu']
    gsu_config = config['hardware_architecture']['gsu_config']
    mem_config = config['hardware_architecture']['memory_config']
    
    qsu_latency_per_element = gsu_costs_config.get('qsu_op_per_gaussian', 4)
    bsu_latency = gsu_costs_config.get('bsu_op_per_16_gaussians', 10)
    bsu_throughput = gsu_costs_config.get('bsu_throughput_per_chunk', 1)
    num_pivots = gsu_config.get('qsu_pivot_count', 7)
    num_subsets = num_pivots + 1
    gfeat_buffer_size = config['hardware_architecture']['vru_config'].get('gfeat_buffer_size_gaussians', 256)
    bsu_channels = gsu_config.get('bsu_channels', 16)
    bytes_per_sort_entry = mem_config.get('bytes_per_sort_entry', 6)
    
    total_compute_cycles = 0
    total_dram_bytes = 0
    
    approx_sort_cost = 0
    precise_sort_cost_per_chunk_avg = 0
    num_chunks = math.ceil(num_elements / gfeat_buffer_size) if gfeat_buffer_size > 0 else 0

    if num_elements > 0:
        read_bytes = num_elements * bytes_per_sort_entry
        write_bytes = num_elements * bytes_per_sort_entry
        total_dram_bytes = read_bytes + write_bytes
        cost.add_memory_cycles(bytes_to_cycles(total_dram_bytes, mem_config, config), total_dram_bytes)

        if num_elements > gfeat_buffer_size:
            num_rounds_approx = math.ceil(math.log(num_elements / gfeat_buffer_size, num_subsets)) if num_subsets > 1 else 1
            num_rounds_approx = max(1, num_rounds_approx)
            cost_per_round_approx = num_elements * qsu_latency_per_element
            approx_sort_cost = num_rounds_approx * cost_per_round_approx
        
        total_compute_cycles += approx_sort_cost
        elements_processed = 0
        total_precise_sort_cost = 0

        for i in range(num_chunks):
            current_chunk_size = min(gfeat_buffer_size, num_elements - elements_processed)
            if current_chunk_size <= 0: break
            
            qsu_precise_cycles = 0
            if current_chunk_size > bsu_channels:
                num_rounds_precise = math.ceil(math.log(current_chunk_size / bsu_channels, num_subsets)) if num_subsets > 1 else 1
                num_rounds_precise = max(1, num_rounds_precise)
                cost_per_round_precise = current_chunk_size * qsu_latency_per_element
                qsu_precise_cycles = num_rounds_precise * cost_per_round_precise

            num_bsu_chunks_in_chunk = math.ceil(current_chunk_size / bsu_channels) if bsu_channels > 0 else 0
            bsu_precise_cycles = 0
            if num_bsu_chunks_in_chunk > 0:
                bsu_precise_cycles = bsu_latency + (num_bsu_chunks_in_chunk - 1) * bsu_throughput
            
            total_precise_sort_cost += qsu_precise_cycles + bsu_precise_cycles
            elements_processed += current_chunk_size
        
        if num_chunks > 0:
            precise_sort_cost_per_chunk_avg = total_precise_sort_cost / num_chunks

        total_compute_cycles += total_precise_sort_cost
    
    cost.add_compute_cycles(total_compute_cycles)
    
    breakdown = {
        'total': cost, 
        'method_name': "Full Sort (GSCore)",
        'approx_sort_cost': approx_sort_cost,
        'precise_sort_cost_per_chunk': precise_sort_cost_per_chunk_avg,
        'num_chunks': num_chunks,
        'dram_bytes': total_dram_bytes, 
        'separation': CostTracker(), 
        'sort_new': CostTracker(), 
        'merge': CostTracker(),
        'overflow_fallback': False,
        'gsu_sram_bits_sep': 0,
        'gsu_sram_bits_sgs': 0,
        'gsu_sram_bits_total': 0,
        
        # ▼▼▼ [수정] DRAM 접근 인덱스 개수 추가 ▼▼▼
        'dram_read_indices': num_elements,
        'dram_write_indices': num_elements
        # ▲▲▲ [수정] 완료 ▲▲▲
    }
    return sorted_list, breakdown


# ==============================================================================
# === [수정됨] 32비트/48비트 접근을 분리하는 해시 시뮬레이션 ===
# ==============================================================================
class HardwareHashTableSim:
    """
    [SRAM 비트 수 수정] Build(Write=48bit)와 Probe(Read=32bit)의 SRAM 접근 비트를 분리 계산합니다.
    """
    def __init__(self, table_size=2048, build_op_cost=3, probe_op_cost=1, bits_per_entry=48, bits_per_id=32):
        self.table_size = table_size
        self.sram = [None] * table_size
        self.build_op_cost_per_hop = build_op_cost 
        self.probe_op_cost_per_hop = probe_op_cost
        self.bits_per_entry = bits_per_entry # 48 bits (전체 쓰기)
        self.bits_per_id = bits_per_id       # 32 bits (ID 읽기)

    def _hash_function(self, gid):
        return gid % self.table_size

    def build(self, item):
        gid = get_id(item)
        hash_addr = self._hash_function(gid)
        cycles = 0
        sram_bits = 0 
        start_addr = hash_addr
        
        cycles_read = 0
        while self.sram[hash_addr] is not None:
            cycles_read += self.build_op_cost_per_hop
            sram_bits += self.bits_per_id 
            hash_addr = (hash_addr + 1) % self.table_size
            
            if hash_addr == start_addr: 
                return -1, 0
        
        cycles_read += self.build_op_cost_per_hop
        sram_bits += self.bits_per_id 
        
        cycles_write = 1
        sram_bits += self.bits_per_entry 
        
        self.sram[hash_addr] = item 
        return cycles_read + cycles_write, sram_bits

    def probe(self, gid):
        hash_addr = self._hash_function(gid)
        cycles = 0
        sram_bits = 0 
        start_addr = hash_addr
        
        while True:
            cycles += self.probe_op_cost_per_hop
            sram_bits += self.bits_per_id 
            current_item = self.sram[hash_addr]

            if current_item is None:
                return (None, cycles, sram_bits)
            
            if get_id(current_item) == gid:
                cycles += 1
                return (current_item, cycles, sram_bits)
            
            hash_addr = (hash_addr + 1) % self.table_size

            if hash_addr == start_addr:
                return (None, cycles, sram_bits)

# ==============================================================================
# === 'Sort New' 함수 (SRAM 접근 비트 반환 추가) ===
# ==============================================================================
def sort_new_sgs(list_to_sort, config):
    sorted_list, cost_bd = full_sort(list_to_sort, config)
    sort_new_cost = cost_bd['total']
    
    sort_new_cost.dram_bytes = 0
    sort_new_cost.memory_cycles = 0 
    sort_new_cost.final_cycles = sort_new_cost.compute_cycles
    
    sram_bits_sgs = cost_bd['dram_bytes'] * 8
    
    return sorted_list, sort_new_cost, sram_bits_sgs

# ==============================================================================
# === [수정됨] 'incremental_sort_merge' (SRAM 비트 합산) ===
# ==============================================================================
def incremental_sort_merge(list_t_minus_1_depth_sorted, list_t_id_sorted, config):
    costs_config = config['cycle_costs']['gsu']
    mem_config = config['hardware_architecture']['memory_config']
    bytes_per_sort_entry = mem_config.get('bytes_per_sort_entry', 6)
    
    bits_per_sort_entry = bytes_per_sort_entry * 8 # 48 bits
    bits_per_id = 32 # 32비트(4바이트) ID
    
    # === GSCore (Baseline) 또는 첫 프레임일 경우 ===
    if not list_t_minus_1_depth_sorted:
        sorted_list, cost_bd = full_sort(list_t_id_sorted, config)
        return sorted_list, cost_bd

    # === 우리 방식 (Hash)일 경우 ===
    sim_settings = config['simulation_global_settings']
    use_detailed_hash_sim = sim_settings.get('detailed_hash_sim', False)

    L = len(list_t_minus_1_depth_sorted)
    M = len(list_t_id_sorted)
    
    # === [OVERFLOW] 1. SRAM A (4096) 한계 검사 ===
    sram_a_limit = 4096 
    if M > sram_a_limit:
        sorted_list, cost_bd = full_sort(list_t_id_sorted, config)
        cost_bd['overflow_fallback'] = True
        return sorted_list, cost_bd
    
    compute_cycles_sep = 0
    gsu_sram_bits_sep = 0 
    
    build_op_latency = costs_config.get('sep_hash_build_op_per_gaussian', 3)
    probe_op_throughput = costs_config.get('sep_hash_probe_op_per_gaussian', 1)
    probe_op_latency = costs_config.get('sep_hash_probe_latency', 2)

    overflow_detected = False

    if use_detailed_hash_sim:
        # === [OVERFLOW] 2. SRAM B (4096) 한계 검사 (Detailed) ===
        sram_b_limit = 4096
        hash_table_sim = HardwareHashTableSim(
            table_size=sram_b_limit, 
            build_op_cost=build_op_latency,
            probe_op_cost=probe_op_throughput,
            bits_per_entry=bits_per_sort_entry, # 48
            bits_per_id=bits_per_id             # 32
        )
        total_build_cycles = 0
        total_build_sram_bits = 0 
        map_t_sim = {} 
        for item_t in list_t_id_sorted:
            cycles, sram_bits = hash_table_sim.build(item_t) 
            if cycles == -1: 
                overflow_detected = True
                break
            total_build_cycles += cycles
            total_build_sram_bits += sram_bits
        
        if overflow_detected:
            sorted_list, cost_bd = full_sort(list_t_id_sorted, config)
            cost_bd['overflow_fallback'] = True
            return sorted_list, cost_bd
            
        total_probe_cycles = 0
        total_probe_sram_bits = 0 
        still_present_elements, ids_from_prev_frame = [], set()

        if L > 0:
            total_probe_cycles = probe_op_latency 
            for item_tm1 in list_t_minus_1_depth_sorted:
                item_id = get_id(item_tm1)
                ids_from_prev_frame.add(item_id)
                found_item, cycles, sram_bits = hash_table_sim.probe(item_id)
                total_probe_cycles += cycles
                total_probe_sram_bits += sram_bits
            total_probe_cycles -= probe_op_throughput 

            if total_probe_cycles < probe_op_latency: 
                total_probe_cycles = probe_op_latency

        compute_cycles_sep = total_build_cycles + total_probe_cycles
        gsu_sram_bits_sep = total_build_sram_bits + total_probe_sram_bits
        
        map_t = {get_id(item): item for item in list_t_id_sorted}
        still_present_elements, new_elements, ids_from_prev_frame = [], [], set()
        for item_tm1 in list_t_minus_1_depth_sorted:
            item_id = get_id(item_tm1); ids_from_prev_frame.add(item_id)
            if item_id in map_t: still_present_elements.append(map_t[item_id])
        new_ids = set(map_t.keys()) - ids_from_prev_frame
        new_elements = [map_t[id_val] for id_val in new_ids]

    else: # Fast Mode (수식 기반)
        # === [OVERFLOW] 3. SRAM B (2048) 한계 검사 (Fast) ===
        sram_b_limit = 2048
        if M > sram_b_limit:
            sorted_list, cost_bd = full_sort(list_t_id_sorted, config)
            cost_bd['overflow_fallback'] = True
            return sorted_list, cost_bd

        compute_cycles_build = M * build_op_latency
        compute_cycles_probe = probe_op_latency + (L - 1) * probe_op_throughput if L > 0 else 0
        compute_cycles_sep = compute_cycles_build + compute_cycles_probe

        gsu_sram_bits_sep = (M * bits_per_sort_entry) + (L * bits_per_id)

        # 논리적 분리
        map_t = {get_id(item): item for item in list_t_id_sorted}
        still_present_elements, new_elements, ids_from_prev_frame = [], [], set()
        for item_tm1 in list_t_minus_1_depth_sorted:
            item_id = get_id(item_tm1); ids_from_prev_frame.add(item_id)
            if item_id in map_t: still_present_elements.append(map_t[item_id])
        new_ids = set(map_t.keys()) - ids_from_prev_frame
        new_elements = [map_t[id_val] for id_val in new_ids]
        
    # [DRAM] 1. Separation 비용
    read_bytes_prev = L * bytes_per_sort_entry
    read_bytes_curr = M * bytes_per_sort_entry
    sep_dram_bytes = read_bytes_prev + read_bytes_curr
    separation_cost = CostTracker(compute_cycles=compute_cycles_sep)
    separation_cost.add_memory_cycles(bytes_to_cycles(sep_dram_bytes, mem_config, config), sep_dram_bytes)

    # --- 2. Sort New (GSCore 방식 사용) ---
    sorted_new, sort_new_cost, sram_bits_sgs = sort_new_sgs(new_elements, config)
    
    
    # --- 3. Merge 비용 (Sorted List 쓰기) ---
    final_list, p_still, p_new = [], 0, 0
    while p_still < len(still_present_elements) and p_new < len(sorted_new):
        if get_depth(still_present_elements[p_still]) < get_depth(sorted_new[p_new]):
            final_list.append(still_present_elements[p_still]); p_still += 1
        else: final_list.append(sorted_new[p_new]); p_new += 1
    final_list.extend(still_present_elements[p_still:]); final_list.extend(sorted_new[p_new:])
    
    final_list_len = len(final_list) # [수정] 최종 개수 계산
    merge_latency = costs_config.get('merge_pipeline_stages', 3)
    merge_throughput = 1 
    compute_cycles_merge = (merge_latency + (final_list_len - 1) * merge_throughput) if final_list_len > 0 else 0
    
    # [DRAM] Merge 비용
    write_bytes_final = final_list_len * bytes_per_sort_entry
    merge_dram_bytes = write_bytes_final
    merge_cost = CostTracker(compute_cycles=compute_cycles_merge)
    merge_cost.add_memory_cycles(bytes_to_cycles(merge_dram_bytes, mem_config, config), merge_dram_bytes)
    
    # --- 최종 비용 집계 ---
    total_dram_bytes = separation_cost.dram_bytes + sort_new_cost.dram_bytes + merge_cost.dram_bytes
    total_cost = CostTracker(compute_cycles=separation_cost.compute_cycles + sort_new_cost.compute_cycles + merge_cost.compute_cycles,
                             memory_cycles=separation_cost.memory_cycles + sort_new_cost.memory_cycles + merge_cost.memory_cycles,
                             dram_bytes=total_dram_bytes)
    
    total_cost.final_cycles = max(total_cost.compute_cycles, total_cost.memory_cycles)

    # [수정] 최종 breakdown에 SRAM 및 DRAM 인덱스 개수 추가
    breakdown = {
        'total': total_cost,
        'method_name': "Incremental Merge (Hash)",
        'separation': separation_cost,
        'sort_new': sort_new_cost,
        'merge': merge_cost,
        'dram_bytes': total_dram_bytes,
        'overflow_fallback': False,
        'gsu_sram_bits_sep': gsu_sram_bits_sep,
        'gsu_sram_bits_sgs': sram_bits_sgs,
        'gsu_sram_bits_total': gsu_sram_bits_sep + sram_bits_sgs,
        
        # ▼▼▼ [수정] DRAM 접근 인덱스 개수 추가 ▼▼▼
        'dram_read_indices': L + M,
        'dram_write_indices': final_list_len
        # ▲▲▲ [수정] 완료 ▲▲▲
    }
    return final_list, breakdown