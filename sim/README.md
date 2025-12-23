# 하드웨어 성능 시뮬레이터 (Hardware Performance Simulator)

**주의: 이 폴더는 `legacy_sim` 폴더에 있는 정렬 알고리즘 시뮬레이터와는 다른, 전체 렌더링 파이프라인의 하드웨어 성능(FPS 등)을 예측하기 위한 시뮬레이터입니다.**

이 디렉토리의 도구들은 `render_cycle.py`를 통해 생성된 상세 로그(.csv)를 사용하여, 제안하는 하드웨어 아키텍처 상에서 4D 가우시안 스플래팅 렌더링이 어느 정도의 성능을 보일지 예측하고 분석하는 것을 목표로 합니다.

## 주요 파일 및 워크플로우

### `run_simulation.ipynb` (자동화 노트북)

이 폴더의 핵심적인 실행 파일입니다. 아래의 두 단계 프로세스를 자동으로 실행하여 성능 예측을 완료합니다.

1.  **로그 생성**: `sim/src/render_cycle.py`를 실행하여 렌더링 과정의 상세한 동작을 `.csv` 로그 파일로 생성합니다.
2.  **성능 분석**: `sim/src/estimate_performance_our.py`를 실행하여 생성된 로그를 분석하고, 최종 예상 성능(FPS 등)을 계산합니다.

### 사용법

Jupyter Notebook 또는 Jupyter Lab 환경에서 `sim/run_simulation.ipynb` 파일을 열고, 상단의 설정 셀에서 분석하고자 하는 모델 경로(`MODEL_PATH`) 등을 수정한 뒤, 전체 셀을 실행하면 됩니다.

### 수동 실행 (참고용)

노트북을 사용하지 않고 수동으로 실행할 경우의 명령어 예시는 다음과 같습니다.

1.  **로그 생성 (`render_cycle.py`)**
    ```shell
    python sim/src/render_cycle.py -m output/N3V/sear_steak --skip_train --debug --python_sorting
    ```
    *   위 명령을 실행하면 `sim/log/[모델명]_[타임스탬프]` 형태의 로그 폴더가 생성됩니다.

2.  **성능 분석 (`estimate_performance_our.py`)**
    ```shell
    # 위 단계에서 생성된 로그 폴더 경로를 -p 인자로 전달합니다.
    python sim/src/estimate_performance_our.py -p sim/log/sear_steak_python_sorting_20251107-153000
    ```

3.  **성능 분석 (`estimate_performance.py`)**

    `render_cycle.py`로 생성된 로그를 분석하여 하드웨어 성능을 예측합니다. `estimate_performance_our.py`와 달리, 이 스크립트는 단일 하드웨어 구성에 대한 상세 분석을 수행합니다.

    ```shell
    # 로그 폴더 경로를 -p 인자로 전달합니다.
    python sim/src/estimate_performance.py -p sim/log/[모델명]_[타임스탬프]

    # 특정 프레임 구간만 분석하려면 -s(시작)와 -e(끝) 인자를 사용합니다.
    python sim/src/estimate_performance.py -p sim/log/[모델명]_[타임스탬프] -s 10 -e 50
    ```
