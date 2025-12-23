# GSCore Simulator (`sim/src`)

이 디렉토리는 GSCore 하드웨어 아키텍처 시뮬레이터의 핵심 소스 코드를 포함하고 있습니다. 각 파일의 주요 역할은 다음과 같습니다.

## 설정 파일

- **`config.json`**: 시뮬레이션의 전반적인 설정을 담당하는 주 설정 파일입니다. 시뮬레이션 전역 설정, 하드웨어 아키텍처(CCU, GSU, VRU 유닛의 개수 및 파이프라인 구성), 메모리 대역폭, 알고리즘 파라미터(알파 값, 렌더링 거리 등)를 정의합니다.
- **`config_analyze.json`**: 성능 분석 스크립트에서 사용하는 추가 설정 파일입니다. 각 하드웨어 유닛의 연산 당 사이클 비용(cycle costs)과 같이 분석에 특화된 파라미터를 정의합니다.

## 핵심 시뮬레이션 로직

- **`simulator.py`**: GSCore 시뮬레이터의 메인 로직을 포함합니다. CCU, GSU, VRU 각 렌더링 파이프라인 단계를 시뮬레이션하며, 멀티프로세싱을 통해 병렬 처리를 관리합니다.
  - `_ccu_worker`: Frustum Culling, Covariance 및 SH 계산 등 CCU의 작업을 시뮬레이션합니다.
  - `_simulate_tile_worker`: GSU의 정렬(Sorting)과 VRU의 래스터라이제이션(Rasterization) 작업을 시뮬레이션합니다.
- **`loader.py`**: 시뮬레이션에 필요한 데이터(카메라 파라미터, Gaussian 데이터)를 읽어오고, 시뮬레이션에 적합한 형식으로 변환하는 역할을 합니다.
- **`sorting_algorithms.py`**: GSU(Gaussian Splatting Unit)에서 사용되는 정렬 알고리즘을 모델링합니다.
  - `full_sort`: 데이터를 매번 처음부터 다시 정렬하는 방식의 비용을 계산합니다.
  - `incremental_sort_merge`: 이전 프레임의 정렬 결과를 재활용하여 변경된 부분만 업데이트하는 증분 정렬 방식의 비용을 계산합니다.

## 렌더링 및 로그 생성

- **`render_cycle.py`**: 4D Gaussian Splatting 모델의 렌더링 과정을 관리하는 스크립트입니다. 각 뷰(카메라)를 순회하며 렌더링을 수행하고, 디버그 모드가 활성화된 경우 분석에 필요한 다양한 로그(컬링 정보, 정렬 전/후 데이터, 픽셀별 기여자 수 등)를 CSV 파일 형태로 저장합니다.
- **`cycle_logger.py`**: 시뮬레이션 중 발생하는 다양한 연산(operation)의 횟수를 실시간으로 추적하고 집계하는 `OperationCounter` 클래스를 제공합니다.

## 성능 분석 스크립트

- **`estimate_performance.py`**: `render_cycle.py`가 생성한 로그 파일들을 종합하여 시스템의 성능을 분석하는 스크립트입니다. 프레임별 FPS, 각 파이프라인 단계의 지연 시간(latency), 병목 지점 등을 계산하여 상세한 리포트를 생성합니다.
- **`estimate_performance_our.py`**: `estimate_performance.py`와 유사한 성능 분석 스크립트이지만, `sorting_algorithms.py`에 정의된 증분 정렬(Incremental Sort)과 전체 정렬(Full Sort) 방식의 성능을 비교 분석하는 기능이 특화되어 있습니다.
- **`post_sim.py`**: 시뮬레이션 로그를 파싱하여 간소화된 성능 분석을 수행하는 스크립트입니다.
