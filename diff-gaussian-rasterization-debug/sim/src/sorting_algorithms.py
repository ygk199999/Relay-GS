import math

class CostTracker:
    """Compute와 Memory 사이클을 별도로 추적하고 병목을 기반으로 최종 사이클을 결정합니다."""
    def __init__(self, compute_cycles=0, memory_cycles=0):
        self.compute_cycles = compute_cycles
        self.memory_cycles = memory_cycles
        self.final_cycles = max(compute_cycles, memory_cycles)

    def add_compute_cycles(self, count=1):
        self.compute_cycles += count
        self.final_cycles = max(self.compute_cycles, self.memory_cycles)

    def add_memory_cycles(self, count=1):
        self.memory_cycles += count
        self.final_cycles = max(self.compute_cycles, self.memory_cycles)

    def __repr__(self):
        return f"Cost(Final={self.final_cycles}, Comp={self.compute_cycles}, Mem={self.memory_cycles})"

def get_id(item): return item[0]
def get_depth(item): return item[1]

# ==============================================================================
# === [수정됨] GSCore 2-Stage 정렬 모델 (우리 기준 적용) ===
# ==============================================================================
def full_sort(list_to_sort, config):
    """
    [GCore 논리 최종본] GSCore의 2-Stage 계층 정렬을 시뮬레이션합니다.
    - Stage 1: QSU로 N -> 256개짜리 청크로 분할
    - Stage 2: 256개 청크를 (QSU + BSU)로 정밀 정렬
    - "우리 기준" (SRAM 지연 시간)을 반영합니다.
    - 파이프라인 분석을 위해 상세 내역(breakdown)을 반환합니다.
    """
    cost = CostTracker()
    num_elements = len(list_to_sort)
    
    # --- 논리적 정렬 (결과 반환용) ---
    if num_elements == 0: 
        sorted_list = []
    else:
        sorted_list = sorted(list_to_sort, key=lambda item: get_depth(item))
    
    # === 비용 계산 설정 로드 ===
    gsu_costs_config = config['cycle_costs']['gsu']
    gsu_config = config['hardware_architecture']['gsu_config']
    
    # GSCore 논문 및 "우리 기준" 값
    qsu_latency_per_element = gsu_costs_config.get('qsu_op_per_gaussian', 4)  # QSU 항목당 4 사이클 (Stall 반영)
    bsu_latency = gsu_costs_config.get('bsu_op_per_16_gaussians', 10)         # BSU 파이프라인 10 사이클 Latency
    bsu_throughput = gsu_costs_config.get('bsu_throughput_per_chunk', 1)     # BSU 파이프라인 1 사이클 Throughput
    
    num_pivots = gsu_config.get('qsu_pivot_count', 7)
    num_subsets = num_pivots + 1  # 8-way
    
    # GSCore 논문 청크 크기
    gfeat_buffer_size = config['hardware_architecture']['vru_config'].get('gfeat_buffer_size_gaussians', 256)
    bsu_channels = gsu_config.get('bsu_channels', 16)
    
    total_compute_cycles = 0
    
    # 상세 내역 (Breakdown)
    approx_sort_cost = 0
    precise_sort_cost_per_chunk_avg = 0 # 청크 1개 정렬에 드는 평균 비용
    num_chunks = math.ceil(num_elements / gfeat_buffer_size) if gfeat_buffer_size > 0 else 0

    if num_elements > 0:
        # =======================================================================
        # === [GSCore 논리] Stage 1: Approximate Sort (N -> 256개 청크) ===
        # =======================================================================
        if num_elements > gfeat_buffer_size:
            num_rounds_approx = math.ceil(math.log(num_elements / gfeat_buffer_size, num_subsets)) if num_subsets > 1 else 1
            num_rounds_approx = max(1, num_rounds_approx)
            cost_per_round_approx = num_elements * qsu_latency_per_element
            approx_sort_cost = num_rounds_approx * cost_per_round_approx
        
        total_compute_cycles += approx_sort_cost

        # =======================================================================
        # === [GSCore 논리] Stage 2: Precise Sort (256개 청크 -> 16개 청크) ===
        # =======================================================================
        elements_processed = 0
        total_precise_sort_cost = 0

        for i in range(num_chunks):
            current_chunk_size = min(gfeat_buffer_size, num_elements - elements_processed)
            if current_chunk_size <= 0: break
            
            # --- 2a. QSU로 256개를 16개로 쪼개기 ---
            qsu_precise_cycles = 0
            if current_chunk_size > bsu_channels:
                num_rounds_precise = math.ceil(math.log(current_chunk_size / bsu_channels, num_subsets)) if num_subsets > 1 else 1
                num_rounds_precise = max(1, num_rounds_precise)
                cost_per_round_precise = current_chunk_size * qsu_latency_per_element
                qsu_precise_cycles = num_rounds_precise * cost_per_round_precise

            # --- 2b. BSU로 16개짜리 덩어리들 정렬 ---
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
    
    # 파이프라인 분석을 위한 상세 내역 반환
    breakdown = {
        'total': cost, # 이것은 '순차' 실행 시 총 비용
        'method_name': "Full Sort (GSCore)",
        # GSCore 파이프라인 분석용
        'approx_sort_cost': approx_sort_cost,
        'precise_sort_cost_per_chunk': precise_sort_cost_per_chunk_avg,
        'num_chunks': num_chunks,
        # 해시 방식과 포맷을 맞추기 위한 빈 값
        'separation': CostTracker(), 
        'sort_new': CostTracker(), 
        'merge': CostTracker()
    }
    return sorted_list, breakdown


class HardwareHashTableSim:
    """
    '선형 탐색(Linear Probing)'을 사용하는 해시 테이블 하드웨어의 
    동작을 시뮬레이션하고, 실제 '연산 사이클'을 계산합니다.
    """
    def __init__(self, table_size=2048, build_op_cost=1, probe_op_cost=1):
        self.table_size = table_size
        self.sram = [None] * table_size
        # [우리 기준 적용] build/probe op cost는 각 'hop'의 지연 시간(Latency)을 의미
        self.build_op_cost_per_hop = build_op_cost 
        self.probe_op_cost_per_hop = probe_op_cost 
    
    def _hash_function(self, gid):
        return gid % self.table_size

    def build(self, item):
        """
        SRAM에 (gid, item)을 저장하고, 충돌 포함 '쓰기'에 걸린 사이클을 반환합니다.
        "우리 기준": Latency = 3 (읽기 2 + 쓰기 1)
        """
        gid = get_id(item)
        hash_addr = self._hash_function(gid)
        cycles = 0
        start_addr = hash_addr
        
        # 1. 빈 슬롯 찾기 (읽기)
        while self.sram[hash_addr] is not None:
            cycles += self.build_op_cost_per_hop # 1 hop (읽기) 비용
            hash_addr = (hash_addr + 1) % self.table_size
            
            if hash_addr == start_addr: 
                raise Exception(f"Hash Table Full. GID {gid} failed to build.") 
        
        # 2. 찾은 후, 첫 읽기 비용 추가
        cycles += self.build_op_cost_per_hop # 마지막 빈 슬롯 '읽기' 비용
        
        # 3. 쓰기 (Write) 비용은 1로 가정 (config에 포함 안 됨)
        cycles += 1 # 1-cycle write latency
        
        self.sram[hash_addr] = item 
        return cycles

    def probe(self, gid):
        """
        SRAM에서 gid를 찾고, (찾은 item, '조회'에 걸린 사이클)을 반환합니다.
        "우리 기준": Latency = 2 (읽기 2) (파이프라인 시 Throughput 1)
        """
        hash_addr = self._hash_function(gid)
        cycles = 0
        start_addr = hash_addr
        
        while True:
            # 1. 읽기 (Read) 비용
            cycles += self.probe_op_cost_per_hop # 1 hop (읽기) 비용
            current_item = self.sram[hash_addr]

            if current_item is None:
                return (None, cycles) # 2a. Miss
            
            if get_id(current_item) == gid:
                # 2b. Hit (읽기 완료)
                return (current_item, cycles) 
            
            hash_addr = (hash_addr + 1) % self.table_size

            if hash_addr == start_addr:
                return (None, cycles) # Miss


def incremental_sort_merge(list_t_minus_1_depth_sorted, list_t_id_sorted, config):
    """
    [수정됨] GSCore 또는 해시 기반 정렬을 수행하고
    파이프라인 분석을 위한 상세 내역을 반환합니다.
    """
    costs_config = config['cycle_costs']['gsu']
    mem_config = config['hardware_architecture']['memory_config']
    clk_ghz = config['simulation_global_settings'].get('clock_frequency_ghz', 1.0)
    dram_bw_gb_s = mem_config.get('dram_bandwidth_gb_per_sec', 32)
    # [우리 기준 적용] 정렬 항목 크기
    bytes_per_sort_entry = mem_config.get('bytes_per_sort_entry', 6)
    
    bytes_per_cycle = (dram_bw_gb_s * 1e9) / (clk_ghz * 1e9) if clk_ghz > 0 else 0
    
    def bytes_to_cycles(num_bytes):
        return math.ceil(num_bytes / bytes_per_cycle) if bytes_per_cycle > 0 else 0

    # === GSCore (Baseline) 또는 첫 프레임일 경우 ===
    if not list_t_minus_1_depth_sorted:
        sorted_list, cost_bd = full_sort(list_t_id_sorted, config)
        read_bytes = len(list_t_id_sorted) * bytes_per_sort_entry
        write_bytes = len(list_t_id_sorted) * bytes_per_sort_entry
        cost_bd['total'].add_memory_cycles(bytes_to_cycles(read_bytes + write_bytes))
        return sorted_list, cost_bd # GSCore 상세 내역 반환

    # === 우리 방식 (Hash)일 경우 ===
    sim_settings = config.get('simulation_global_settings', {})
    use_detailed_hash_sim = sim_settings.get('detailed_hash_sim', False)

    L = len(list_t_minus_1_depth_sorted)
    M = len(list_t_id_sorted)
    
    compute_cycles_sep = 0
    
    # [우리 기준 적용] config에서 Latency / Throughput 값 가져오기
    # Build는 파이프라인 안되므로 Latency (예: 3)
    build_op_latency = costs_config.get('sep_hash_build_op_per_gaussian', 3)
    # Probe는 파이프라인되므로 Throughput (예: 1)
    probe_op_throughput = costs_config.get('sep_hash_probe_op_per_gaussian', 1)
    # Probe의 초기 Latency (예: 2)
    probe_op_latency = costs_config.get('sep_hash_probe_latency', 2)

    if use_detailed_hash_sim:
        # Detailed Sim: 각 hop의 비용으로 계산
        # (간단한 모델: config의 'probe'를 hop 비용으로, 'build'를 hop 비용으로 사용)
        hash_table_sim = HardwareHashTableSim(
            table_size=max(M * 2, 2048),
            build_op_cost=build_op_latency, # Build는 Latency가 hop 비용
            probe_op_cost=probe_op_throughput # Probe는 Throughput이 hop 비용
        )
        total_build_cycles = 0
        map_t_sim = {} 
        for item_t in list_t_id_sorted:
            cycles = hash_table_sim.build(item_t)
            total_build_cycles += cycles
            map_t_sim[get_id(item_t)] = item_t
        
        total_probe_cycles = 0
        still_present_elements, ids_from_prev_frame = [], set()

        for item_tm1 in list_t_minus_1_depth_sorted:
            item_id = get_id(item_tm1)
            ids_from_prev_frame.add(item_id)
            found_item, cycles = hash_table_sim.probe(item_id)
            total_probe_cycles += cycles
            if found_item is not None:
                still_present_elements.append(found_item)

        compute_cycles_sep = total_build_cycles + total_probe_cycles
        new_ids = set(map_t_sim.keys()) - ids_from_prev_frame
        new_elements = [map_t_sim[id_val] for id_val in new_ids]

    else: # Fast Mode (수식 기반)
        # [우리 기준 적용] Build는 Latency*M, Probe는 Latency + (L-1)*Throughput
        compute_cycles_build = M * build_op_latency
        compute_cycles_probe = probe_op_latency + (L - 1) * probe_op_throughput if L > 0 else 0
        compute_cycles_sep = compute_cycles_build + compute_cycles_probe

        # 논리적 분리
        map_t = {get_id(item): item for item in list_t_id_sorted}
        still_present_elements, new_elements, ids_from_prev_frame = [], [], set()
        for item_tm1 in list_t_minus_1_depth_sorted:
            item_id = get_id(item_tm1); ids_from_prev_frame.add(item_id)
            if item_id in map_t: still_present_elements.append(map_t[item_id])
        new_ids = set(map_t.keys()) - ids_from_prev_frame
        new_elements = [map_t[id_val] for id_val in new_ids]
        
    read_bytes_sep = (L + M) * bytes_per_sort_entry
    memory_cycles_sep = bytes_to_cycles(read_bytes_sep)
    separation_cost = CostTracker(compute_cycles=compute_cycles_sep, memory_cycles=memory_cycles_sep)

    # --- 2. Sort New (GSCore 방식 사용) ---
    sorted_new, sort_new_cost_bd = full_sort(new_elements, config) # full_sort 호출
    sort_new_cost = sort_new_cost_bd['total'] # total 비용만 가져옴
    read_bytes_sort = len(new_elements) * bytes_per_sort_entry
    write_bytes_sort = len(new_elements) * bytes_per_sort_entry
    sort_new_cost.add_memory_cycles(bytes_to_cycles(read_bytes_sort + write_bytes_sort))
    
    # --- 3. Merge ---
    final_list_len = len(still_present_elements) + len(sorted_new)
    merge_latency = costs_config.get('merge_pipeline_stages', 3)
    merge_throughput = 1 # 1 사이클/항목
    compute_cycles_merge = (merge_latency + (final_list_len - 1) * merge_throughput) if final_list_len > 0 else 0
    
    read_bytes_merge = final_list_len * bytes_per_sort_entry
    write_bytes_merge = final_list_len * bytes_per_sort_entry
    memory_cycles_merge = bytes_to_cycles(read_bytes_merge + write_bytes_merge)
    merge_cost = CostTracker(compute_cycles=compute_cycles_merge, memory_cycles=memory_cycles_merge)

    # 논리적 병합
    final_list, p_still, p_new = [], 0, 0
    while p_still < len(still_present_elements) and p_new < len(sorted_new):
        if get_depth(still_present_elements[p_still]) < get_depth(sorted_new[p_new]):
            final_list.append(still_present_elements[p_still]); p_still += 1
        else: final_list.append(sorted_new[p_new]); p_new += 1
    final_list.extend(still_present_elements[p_still:]); final_list.extend(sorted_new[p_new:])
    
    # --- 최종 비용 집계 (순차 실행 기준) ---
    total_final_cycles = separation_cost.final_cycles + sort_new_cost.final_cycles + merge_cost.final_cycles
    total_cost = CostTracker()
    total_cost.compute_cycles = separation_cost.compute_cycles + sort_new_cost.compute_cycles + merge_cost.compute_cycles
    total_cost.memory_cycles = separation_cost.memory_cycles + sort_new_cost.memory_cycles + merge_cost.memory_cycles
    total_cost.final_cycles = total_final_cycles # '순차' 총합

    # 파이프라인 분석을 위한 상세 내역 반환
    breakdown = {
        'total': total_cost,
        'method_name': "Incremental Merge (Hash)",
        # 해시 파이프라인 분석용
        'separation': separation_cost,
        'sort_new': sort_new_cost,
        'merge': merge_cost,
        'num_elements_merged': final_list_len,
        # GSCore 방식과 포맷을 맞추기 위한 빈 값
        'approx_sort_cost': 0,
        'precise_sort_cost_per_chunk': 0,
        'num_chunks': 0
    }
    return final_list, breakdown