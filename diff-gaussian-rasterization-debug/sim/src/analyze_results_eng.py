import pandas as pd
import numpy as np
import argparse
import os

def analyze_performance(csv_path):
    """
    (DRAM 에너지 모델 수정)
    DRAM 에너지를 [칩 자체(pJ/bit)]와 [인터페이스(W, 1인덱스=1사이클)]로 분리하여 합산합니다.
    """
    # ==============================================================================
    # === 사용자 설정 영역 (User Configuration Area) ===
    # ==============================================================================
    CONFIG = {
        "clock_frequency_ghz":0.3, # 1.0 GHz (분석용)

        "power_consumption_watts": {
            # On-chip 모듈
            "ccu": 0.011876*4,
            "vru_base": 0.01875,
            "vru_fast": 0.0188,
            "gsu_baseline_logic": 0.015*4,
            "gsu_incremental_logic": 7.15211E-06*4,
            "gsu_sram_energy_pj_per_bit": 0.15*4,
            "gsu_sram_static_power_w": 0.011,  # [중요] 이 값은 예시입니다. 실제 값으로 교체하세요.

            
            # ▼▼▼ [수정] DRAM 인터페이스 모듈 (신규) ▼▼▼
            "tile_distributor_w": 3.52212E-06,  # GSU가 DRAM에서 인덱스를 읽어올 때(L+M) 활성화
            "grouped_merge_unit_w": 2.45416E-05,   # GSU가 DRAM에 인덱스를 쓸 때(N) 활성화
            
            # ▼▼▼ [롤백] DRAM 칩 자체 에너지 (pJ/bit) ▼▼▼
            "dram_energy_pj_per_bit":  7
            # ▲▲▲ [수정] 완료 ▲▲▲
        }
    }
    # ==============================================================================

    # --- 1. 데이터 로드 및 첫 프레임 제외 ---
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

    # 첫 번째 프레임(index 0)을 평균 계산에서 제외
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

    clk_hz = CONFIG["clock_frequency_ghz"] * 1.0e9
    power_config = CONFIG.get("power_consumption_watts")

    # --- 2. 컬럼 확인 ---
    required_cols = [
        'gsu_sram_access_incremental_bits', 'gsu_sram_access_baseline_bits',
        'dram_read_indices_incremental', 'dram_write_indices_incremental',
        'dram_read_indices_baseline', 'dram_write_indices_baseline',
        'dram_read_gsu_baseline_bits', 'dram_read_gsu_incremental_bits'
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"\nError: CSV file is missing required columns: {missing_cols}")
        print("Please ensure 'estimate_performance_our_hw.py' generated these columns.")
        return

    # --- 3. 속도(Speed) 분석 ---
    # (이전과 동일)
    df['total_latency_base_seq'] = df['ccu_stage_latency'] + df['gsu_vru_baseline_latency_sequential']
    df['vru_incremental_latency_half'] = df['vru_latency_on_incremental'] / 2.0
    df['total_latency_incr_seq_vru_fast'] = df['ccu_stage_latency'] + df['gsu_incremental_latency_total'] + df['vru_incremental_latency_half']
    gsu_init_latency = df['gsu_incremental_latency_sep'] + df['gsu_incremental_latency_sgs']
    pipelined_part_vru_fast = np.maximum(df['gsu_incremental_latency_merge'], df['vru_incremental_latency_half'])
    df['total_latency_incr_pipe_vru_fast'] = df['ccu_stage_latency'] + gsu_init_latency + pipelined_part_vru_fast
    
    avg_fps = {
        "Baseline GSU + Base VRU (Seq)": clk_hz / df['total_latency_base_seq'].mean(),
        "Incremental GSU + Base VRU (Seq)": clk_hz / df['total_latency_sequential'].mean(),
        "Incremental GSU + Fast VRU (Seq)": clk_hz / df['total_latency_incr_seq_vru_fast'].mean(),
        "Incremental GSU + Fast VRU (Pipe)": clk_hz / df['total_latency_incr_pipe_vru_fast'].mean(),
    }
    baseline_key = "Baseline GSU + Base VRU (Seq)"
    baseline_fps = avg_fps[baseline_key]
    speedup = {scenario: fps / baseline_fps for scenario, fps in avg_fps.items()}

    # --- 4. 에너지(Energy) 분석 (DRAM 모델 수정) ---
    if power_config:
        # 컴포넌트별 활성 시간(초) 계산
        df['time_ccu'] = df['ccu_stage_latency'] / clk_hz
        df['time_gsu_baseline'] = df['gsu_baseline_latency_total'] / clk_hz
        df['time_vru_baseline'] = df['vru_latency_on_baseline'] / clk_hz
        df['time_gsu_incremental_total'] = df['gsu_incremental_latency_total'] / clk_hz
        df['time_gsu_init'] = (df['gsu_incremental_latency_sep'] + df['gsu_incremental_latency_sgs']) / clk_hz
        df['time_gsu_merge'] = df['gsu_incremental_latency_merge'] / clk_hz
        df['time_vru_base_on_prop'] = df['vru_latency_on_incremental'] / clk_hz
        df['time_vru_fast_on_prop'] = df['vru_incremental_latency_half'] / clk_hz

        # DRAM 인터페이스 모듈 활성 시간 (1 인덱스 = 1 사이클)
        df['time_distributor_baseline'] = df['dram_read_indices_baseline'] / clk_hz
        df['time_merge_unit_baseline'] = df['dram_write_indices_baseline'] / clk_hz
        df['time_distributor_incremental'] = df['dram_read_indices_incremental'] / clk_hz
        df['time_merge_unit_incremental'] = df['dram_write_indices_incremental'] / clk_hz

        # 컴포넌트별 에너지 소비량(Joule) 계산
        # On-Chip
        df['energy_ccu'] = power_config['ccu'] * df['time_ccu']
        df['energy_vru_baseline'] = power_config['vru_base'] * df['time_vru_baseline']
        df['energy_vru_base_prop'] = power_config['vru_base'] * df['time_vru_base_on_prop']
        df['energy_vru_fast_prop'] = power_config['vru_fast'] * df['time_vru_fast_on_prop']
        # GSU Logic
        df['energy_gsu_baseline_logic'] = power_config['gsu_baseline_logic'] * df['time_gsu_baseline']
        df['energy_gsu_incremental_logic_total'] = power_config['gsu_incremental_logic'] * df['time_gsu_incremental_total']
        df['energy_gsu_init_logic'] = power_config['gsu_incremental_logic'] * df['time_gsu_init']
        df['energy_gsu_merge_logic'] = power_config['gsu_incremental_logic'] * df['time_gsu_merge']
        # GSU SRAM
        gsu_sram_pj_to_j = power_config['gsu_sram_energy_pj_per_bit'] * 1e-12
        df['energy_sram_gsu_baseline'] = gsu_sram_pj_to_j * df['gsu_sram_access_baseline_bits']
        df['energy_sram_gsu_incremental'] = gsu_sram_pj_to_j * df['gsu_sram_access_incremental_bits']
        
        # ▼▼▼ [수정] DRAM 인터페이스 에너지 (W) ▼▼▼
        df['energy_distributor_baseline'] = power_config['tile_distributor_w'] * df['time_distributor_baseline']
        df['energy_merge_unit_baseline'] = power_config['grouped_merge_unit_w'] * df['time_merge_unit_baseline']
        df['energy_distributor_incremental'] = power_config['tile_distributor_w'] * df['time_distributor_incremental']
        df['energy_merge_unit_incremental'] = power_config['grouped_merge_unit_w'] * df['time_merge_unit_incremental']

        # ▼▼▼ [수정] GSU SRAM 정적(Static) 에너지 계산 추가 ▼▼▼
        # 각 시나리오의 총 실행 시간(초) 동안 SRAM 정적 전력이 소모됨
        gsu_sram_static_w = power_config.get('gsu_sram_static_power_w', 0)
        df['energy_sram_static_baseline'] = gsu_sram_static_w * (df['total_latency_base_seq'] / clk_hz)
        df['energy_sram_static_incr_seq_base_vru'] = gsu_sram_static_w * (df['total_latency_sequential'] / clk_hz)
        df['energy_sram_static_incr_seq_fast_vru'] = gsu_sram_static_w * (df['total_latency_incr_seq_vru_fast'] / clk_hz)
        df['energy_sram_static_incr_pipe_fast_vru'] = gsu_sram_static_w * (df['total_latency_incr_pipe_vru_fast'] / clk_hz)
        # ▲▲▲ 수정 완료 ▲▲▲
        
        # ▼▼▼ [롤백] DRAM 칩 자체 에너지 (pJ/bit) ▼▼▼
        dram_pj_to_j = power_config.get('dram_energy_pj_per_bit', 0) * 1e-12
        df['energy_dram_baseline'] = dram_pj_to_j *(df['dram_read_gsu_baseline_bits']+df['dram_read_ccu_bits']+2*df['dram_write_gsu_from_ccu_bits']+2*df['dram_write_vru_from_ccu_bits'])
        df['energy_dram_incremental'] = dram_pj_to_j * (df['dram_read_gsu_incremental_bits']+df['dram_read_ccu_bits']+2*df['dram_write_gsu_from_ccu_bits']+2*df['dram_write_vru_from_ccu_bits'])
        # ▲▲▲ [수정] 완료 ▲▲▲

        # ▼▼▼ [수정] 시나리오별 총 에너지 (SRAM 정적 에너지 합산) ▼▼▼
        # 1. Baseline
        df['total_energy_baseline'] = (
            df['energy_ccu'] + 
            df['energy_gsu_baseline_logic'] + df['energy_sram_gsu_baseline'] + 
            df['energy_vru_baseline'] + 
            df['energy_distributor_baseline'] + df['energy_merge_unit_baseline'] +
            df['energy_dram_baseline'] +
            df['energy_sram_static_baseline'] # SRAM 정적 에너지 추가
        )

        # 2. Incremental GSU + Base VRU (Sequential)
        df['total_energy_incr_seq_base_vru'] = (
            df['energy_ccu'] + 
            df['energy_gsu_incremental_logic_total'] + df['energy_sram_gsu_incremental'] + 
            df['energy_vru_base_prop'] + 
            df['energy_distributor_incremental'] + df['energy_merge_unit_incremental'] +
            df['energy_dram_incremental'] +
            df['energy_sram_static_incr_seq_base_vru'] # SRAM 정적 에너지 추가
        )
        
        # 3. Incremental GSU + Fast VRU (Sequential)
        df['total_energy_incr_seq_fast_vru'] = (
            df['energy_ccu'] + 
            df['energy_gsu_incremental_logic_total'] + df['energy_sram_gsu_incremental'] + 
            df['energy_vru_fast_prop'] + 
            df['energy_distributor_incremental'] + df['energy_merge_unit_incremental'] +
            df['energy_dram_incremental'] +
            df['energy_sram_static_incr_seq_fast_vru'] # SRAM 정적 에너지 추가
        )
        
        # 4. Incremental GSU + Fast VRU (Pipeline)
        df['total_energy_incr_pipe_fast_vru'] = (
            df['energy_ccu'] + 
            df['energy_gsu_init_logic'] + df['energy_gsu_merge_logic'] + 
            df['energy_sram_gsu_incremental'] + 
            df['energy_vru_fast_prop'] + 
            df['energy_distributor_incremental'] + df['energy_merge_unit_incremental'] +
            df['energy_dram_incremental'] +
            df['energy_sram_static_incr_pipe_fast_vru'] # SRAM 정적 에너지 추가
        )

        avg_energy_uj = {
            baseline_key: df['total_energy_baseline'].mean() * 1e6,
            "Incremental GSU + Base VRU (Seq)": df['total_energy_incr_seq_base_vru'].mean() * 1e6,
            "Incremental GSU + Fast VRU (Seq)": df['total_energy_incr_seq_fast_vru'].mean() * 1e6,
            "Incremental GSU + Fast VRU (Pipe)": df['total_energy_incr_pipe_fast_vru'].mean() * 1e6,
        }
        
        baseline_energy = avg_energy_uj[baseline_key]
        energy_savings = {scenario: (1 - energy / baseline_energy) * 100 for scenario, energy in avg_energy_uj.items()}
    else:
        print("\nWarning: Power consumption data not found. Energy analysis skipped.")
        avg_energy_uj = {}
        energy_savings = {}

    # --- 5. 모듈 동작 시간 검증 출력 (평균값으로 변경, DRAM 모델 수정) ---
    print("\n" + "="*85)
    print(f" Module Activity Time & Energy Verification (Average over {len(df)} frames)")
    print("="*85)
    
    if len(df) > 0 and power_config:
        mean_frame = df.mean(axis=0) # 모든 컬럼의 평균
        
        # ▼▼▼ [수정] 검증 로직 변경 ▼▼▼
        print(f"\n[Baseline Scenario]")
        print(f"  CCU active time:         {mean_frame['time_ccu']*1e6:>10.2f} us  (Energy: {mean_frame['energy_ccu']*1e6:>8.2f} uJ)")
        print(f"  GSU_baseline_logic act:  {mean_frame['time_gsu_baseline']*1e6:>10.2f} us  (Energy: {mean_frame['energy_gsu_baseline_logic']*1e6:>8.2f} uJ)")
        print(f"  GSU_SRAM access energy:                          (Energy: {mean_frame['energy_sram_gsu_baseline']*1e6:>8.2f} uJ)")
        print(f"  VRU_baseline active:     {mean_frame['time_vru_baseline']*1e6:>10.2f} us  (Energy: {mean_frame['energy_vru_baseline']*1e6:>8.2f} uJ)")
        print(f"  Tile_Distributor (Read): {mean_frame['time_distributor_baseline']*1e6:>10.2f} us  (Energy: {mean_frame['energy_distributor_baseline']*1e6:>8.2f} uJ)")
        print(f"  Grouped_Merge (Write):   {mean_frame['time_merge_unit_baseline']*1e6:>10.2f} us  (Energy: {mean_frame['energy_merge_unit_baseline']*1e6:>8.2f} uJ)")
        print(f"  DRAM Chip (GSU) energy:                        (Energy: {mean_frame['energy_dram_baseline']*1e6:>8.2f} uJ)")
        print(f"  Total sequential time:   {(mean_frame['time_ccu'] + mean_frame['time_gsu_baseline'] + mean_frame['time_vru_baseline'])*1e6:>10.2f} us")
        print(f"  Total energy:                                  (Total:  {mean_frame['total_energy_baseline']*1e6:>8.2f} uJ)")
        
        print(f"\n[Incremental + Fast VRU (Sequential)]")
        print(f"  CCU active time:         {mean_frame['time_ccu']*1e6:>10.2f} us  (Energy: {mean_frame['energy_ccu']*1e6:>8.2f} uJ)")
        print(f"  GSU_incr_logic active:   {mean_frame['time_gsu_incremental_total']*1e6:>10.2f} us  (Energy: {mean_frame['energy_gsu_incremental_logic_total']*1e6:>8.2f} uJ)")
        print(f"  GSU_SRAM access energy:                          (Energy: {mean_frame['energy_sram_gsu_incremental']*1e6:>8.2f} uJ)")
        print(f"  VRU_fast active:         {mean_frame['time_vru_fast_on_prop']*1e6:>10.2f} us  (Energy: {mean_frame['energy_vru_fast_prop']*1e6:>8.2f} uJ)")
        print(f"  Tile_Distributor (Read): {mean_frame['time_distributor_incremental']*1e6:>10.2f} us  (Energy: {mean_frame['energy_distributor_incremental']*1e6:>8.2f} uJ)")
        print(f"  Grouped_Merge (Write):   {mean_frame['time_merge_unit_incremental']*1e6:>10.2f} us  (Energy: {mean_frame['energy_merge_unit_incremental']*1e6:>8.2f} uJ)")
        print(f"  DRAM Chip (GSU) energy:                        (Energy: {mean_frame['energy_dram_incremental']*1e6:>8.2f} uJ)")
        print(f"  Total sequential time:   {(mean_frame['time_ccu'] + mean_frame['time_gsu_incremental_total'] + mean_frame['time_vru_fast_on_prop'])*1e6:>10.2f} us")
        print(f"  Total energy:                                  (Total:  {mean_frame['total_energy_incr_seq_fast_vru']*1e6:>8.2f} uJ)")
        
        print(f"\n[Incremental + Fast VRU (Pipeline)]")
        print(f"  CCU active time:         {mean_frame['time_ccu']*1e6:>10.2f} us  (Energy: {mean_frame['energy_ccu']*1e6:>8.2f} uJ)")
        print(f"  GSU_init_logic active:   {mean_frame['time_gsu_init']*1e6:>10.2f} us  (Energy: {mean_frame['energy_gsu_init_logic']*1e6:>8.2f} uJ)")
        print(f"  GSU_merge_logic active:  {mean_frame['time_gsu_merge']*1e6:>10.2f} us  (Energy: {mean_frame['energy_gsu_merge_logic']*1e6:>8.2f} uJ) [PARALLEL]")
        print(f"  GSU_SRAM access energy:                          (Energy: {mean_frame['energy_sram_gsu_incremental']*1e6:>8.2f} uJ)")
        print(f"  VRU_fast active:         {mean_frame['time_vru_fast_on_prop']*1e6:>10.2f} us  (Energy: {mean_frame['energy_vru_fast_prop']*1e6:>8.2f} uJ) [PARALLEL]")
        print(f"  Tile_Distributor (Read): {mean_frame['time_distributor_incremental']*1e6:>10.2f} us  (Energy: {mean_frame['energy_distributor_incremental']*1e6:>8.2f} uJ)")
        print(f"  Grouped_Merge (Write):   {mean_frame['time_merge_unit_incremental']*1e6:>10.2f} us  (Energy: {mean_frame['energy_merge_unit_incremental']*1e6:>8.2f} uJ)")
        print(f"  DRAM Chip (GSU) energy:                        (Energy: {mean_frame['energy_dram_incremental']*1e6:>8.2f} uJ)")
        pipeline_overlap = max(mean_frame['time_gsu_merge'], mean_frame['time_vru_fast_on_prop']) * 1e6
        print(f"  Parallel overlap time:   {pipeline_overlap:>10.2f} us  (max of GSU_merge, VRU_fast)")
        print(f"  Total pipeline time:     {mean_frame['total_latency_incr_pipe_vru_fast']/clk_hz*1e6:>10.2f} us")
        print(f"  Total energy:                                  (Total:  {mean_frame['total_energy_incr_pipe_fast_vru']*1e6:>8.2f} uJ)")
        # ▲▲▲ [수정] 완료 ▲▲▲

    # --- 6. 최종 리포트 출력 ---
    if len(df) > 0 and power_config:
        mean_frame = df.mean(axis=0) # 모든 컬럼의 평균값 계산

        # 정규화의 기준이 될 '베이스라인 총 에너지' (단위: Joule)
        # 이 값이 0이면 나눗셈 오류가 발생하므로, 0보다 큰지 확인해야 함
        baseline_total_energy = mean_frame.get('total_energy_baseline', 0)

        # --- 리포트 출력용 헬퍼 함수 정의 (수정됨) ---
        def print_energy_scenario_report(title, on_chip_components, off_chip_components, total_energy_key, baseline_total_energy):
            print(f"--- Energy Scenario: {title} ---")
            can_normalize = baseline_total_energy > 0

            # --- On-Chip Energy ---
            print("  [On-Chip Energy]")
            on_chip_total_joules = 0
            for name, energy_key in on_chip_components:
                energy_joules = mean_frame.get(energy_key, 0)
                on_chip_total_joules += energy_joules
                energy_uJ = energy_joules * 1e6

                if can_normalize:
                    normalized_percent = (energy_joules / baseline_total_energy) * 100
                    print(f"    - {name:<27}: {energy_uJ:10.2f} uJ ({normalized_percent:5.1f}%)")
                else:
                    print(f"    - {name:<27}: {energy_uJ:10.2f} uJ")

            on_chip_total_uJ = on_chip_total_joules * 1e6
            if can_normalize:
                on_chip_norm_percent = (on_chip_total_joules / baseline_total_energy) * 100
                print(f"    - {'On-Chip Subtotal':<27}: {on_chip_total_uJ:10.2f} uJ ({on_chip_norm_percent:5.1f}%)")
            else:
                 print(f"    - {'On-Chip Subtotal':<27}: {on_chip_total_uJ:10.2f} uJ")
            print("") # 줄바꿈

            # --- Off-Chip Energy ---
            print("  [Off-Chip Energy]")
            off_chip_total_joules = 0
            for name, energy_key in off_chip_components:
                energy_joules = mean_frame.get(energy_key, 0)
                off_chip_total_joules += energy_joules
                energy_uJ = energy_joules * 1e6

                if can_normalize:
                    normalized_percent = (energy_joules / baseline_total_energy) * 100
                    print(f"    - {name:<27}: {energy_uJ:10.2f} uJ ({normalized_percent:5.1f}%)")
                else:
                    print(f"    - {name:<27}: {energy_uJ:10.2f} uJ")

            off_chip_total_uJ = off_chip_total_joules * 1e6
            if can_normalize:
                off_chip_norm_percent = (off_chip_total_joules / baseline_total_energy) * 100
                print(f"    - {'Off-Chip Subtotal':<27}: {off_chip_total_uJ:10.2f} uJ ({off_chip_norm_percent:5.1f}%)")
            else:
                print(f"    - {'Off-Chip Subtotal':<27}: {off_chip_total_uJ:10.2f} uJ")

            print("\n" + "  " + "-"*50)

            # --- 종합 결과 ---
            total_energy_joules = mean_frame.get(total_energy_key, 0)
            total_energy_uJ = total_energy_joules * 1e6

            if can_normalize:
                total_norm_percent = (total_energy_joules / baseline_total_energy) * 100
                print(f"  - Avg. Total Energy         : {total_energy_uJ:10.2f} uJ ({total_norm_percent:5.1f}%)")
            else:
                print(f"  - Avg. Total Energy         : {total_energy_uJ:10.2f} uJ")

            # 에너지 절감률은 이전과 동일하게 표시 (이것이 더 직관적)
            if can_normalize and total_energy_key != 'total_energy_baseline':
                savings = (1 - total_energy_joules / baseline_total_energy) * 100
                print(f"  - Energy Savings vs. Base   : {savings:18.2f}%")
            print("\n")


        # --- 각 시나리오별 컴포넌트 리스트 정의 (이전과 동일) ---
        on_chip_base = [
        ("CCU", 'energy_ccu'),
        ("GSU Logic (Baseline)", 'energy_gsu_baseline_logic'),
        ("GSU SRAM (Dynamic)", 'energy_sram_gsu_baseline'),
        ("GSU SRAM (Static)", 'energy_sram_static_baseline'), # 추가
        ("VRU (Baseline)", 'energy_vru_baseline'),
        ("DRAM I/F Read (On-Chip)", 'energy_distributor_baseline'),
        ("DRAM I/F Write (On-Chip)", 'energy_merge_unit_baseline'),
    ]
        off_chip_base = [("DRAM Chip Access", 'energy_dram_baseline')]

        on_chip_incr_base_vru = [
            ("CCU", 'energy_ccu'),
            ("GSU Logic (Incremental)", 'energy_gsu_incremental_logic_total'),
            ("GSU SRAM", 'energy_sram_gsu_incremental'),
            ("GSU SRAM (Static)", 'energy_sram_static_incr_seq_base_vru'), # 추가
            ("VRU (Base)", 'energy_vru_base_prop'),
            ("DRAM I/F Read (On-Chip)", 'energy_distributor_incremental'),
            ("DRAM I/F Write (On-Chip)", 'energy_merge_unit_incremental'),
        ]
        off_chip_incr = [
            ("DRAM Chip Access", 'energy_dram_incremental')
        ]

        on_chip_incr_fast_vru = [
            ("CCU", 'energy_ccu'),
            ("GSU Logic (Incremental)", 'energy_gsu_incremental_logic_total'),
            ("GSU SRAM", 'energy_sram_gsu_incremental'),
            ("GSU SRAM (Static)", 'energy_sram_static_incr_seq_base_vru'), # 추가
            ("VRU (Fast)", 'energy_vru_fast_prop'),
            ("DRAM I/F Read (On-Chip)", 'energy_distributor_incremental'),
            ("DRAM I/F Write (On-Chip)", 'energy_merge_unit_incremental'),
        ]
        on_chip_incr_fast_vru_seq = [
        ("CCU", 'energy_ccu'),
        ("GSU Logic (Incremental)", 'energy_gsu_incremental_logic_total'),
        ("GSU SRAM (Dynamic)", 'energy_sram_gsu_incremental'),
        ("GSU SRAM (Static)", 'energy_sram_static_incr_seq_fast_vru'), # 추가
        ("VRU (Fast)", 'energy_vru_fast_prop'),
        ("DRAM I/F Read (On-Chip)", 'energy_distributor_incremental'),
        ("DRAM I/F Write (On-Chip)", 'energy_merge_unit_incremental'),    ]

        on_chip_incr_fast_vru_pipe = [
        ("CCU", 'energy_ccu'),
        ("GSU Logic (Init)", 'energy_gsu_init_logic'),
        ("GSU Logic (Merge)", 'energy_gsu_merge_logic'),
        ("GSU SRAM (Dynamic)", 'energy_sram_gsu_incremental'),
        ("GSU SRAM (Static)", 'energy_sram_static_incr_pipe_fast_vru'), # 추가
        ("VRU (Fast)", 'energy_vru_fast_prop'),
        ("DRAM I/F Read (On-Chip)", 'energy_distributor_incremental'),
        ("DRAM I/F Write (On-Chip)", 'energy_merge_unit_incremental'),    ]


        # --- 최종 리포트 출력 ---
        print("\n\n" + "="*60)
        print(" Energy Breakdown & Efficiency Analysis".center(60))
        print(" (Energy values are normalized to Baseline Total Energy)".center(60))
        print(f" Source: {os.path.basename(csv_path)}".center(60))
        print("="*60 + "\n")

        baseline_key = "Baseline GSU + Base VRU (Seq)"
        print_energy_scenario_report(
            baseline_key,
            on_chip_base, off_chip_base,
            'total_energy_baseline', baseline_total_energy
        )

        print_energy_scenario_report(
            "Incremental GSU + Base VRU (Seq)",
            on_chip_incr_base_vru, off_chip_incr,
            'total_energy_incr_seq_base_vru', baseline_total_energy
        )

        print_energy_scenario_report(
            "Incremental GSU + Fast VRU (Seq)",
            on_chip_incr_fast_vru, off_chip_incr,
            'total_energy_incr_seq_fast_vru', baseline_total_energy
        )

        print_energy_scenario_report("Incremental GSU + Fast VRU (Seq)", on_chip_incr_fast_vru_seq, off_chip_incr, 'total_energy_incr_seq_fast_vru', baseline_total_energy)

        print_energy_scenario_report("Incremental GSU + Fast VRU (Pipe)", on_chip_incr_fast_vru_pipe, off_chip_incr, 'total_energy_incr_pipe_fast_vru', baseline_total_energy)

    else:
        print("No power config found or no data available to generate the energy report.")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyzes the detailed performance CSV for Speed and Energy based on hardware-specific power models.")
    parser.add_argument("csv_path", type=str, help="Path to the 'performance_details_over_frames.csv' file.")
    args = parser.parse_args()
    
    analyze_performance(args.csv_path)