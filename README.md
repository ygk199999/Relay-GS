# Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting
### [Project page](https://fudan-zvg.github.io/4d-gaussian-splatting/) | [Paper](https://arxiv.org/abs/2310.10642)
> [**Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting**](https://arxiv.org/abs/2310.10642),  
> Zeyu Yang, Hongye Yang, Zijie Pan, [Li Zhang](https://lzrobots.github.io)  
> **Fudan University**  
> **ICLR 2024**

> [**4D Gaussian Splatting: Modeling Dynamic Scenes with Native 4D Primitives**](https://arxiv.org/abs/2412.20720),  
> Zeyu Yang, Zijie Pan, Xiatian Zhu, [Li Zhang](https://lzrobots.github.io), Jianfeng Feng, Yu-Gang Jiang, Philip H.S. Torr  
> **Fudan University, University of Surrey, University of Oxford**  
> **Arxiv preprint**


**This repository is the official implementation of "Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting".** In this paper, we propose coherent integrated modeling of the space and time dimensions for dynamic scenes by formulating unbiased 4D Gaussian primitives along with a dedicated rendering pipeline.


## 🛠️ Pipeline
<div align="center">
  <img src="assets/pipeline.png"/>
</div><br/>


## Get started

### Environment

The hardware and software requirements are the same as those of the [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), which this code is built upon. To setup the environment, please run the following command:

```shell
git clone https://github.com/fudan-zvg/4d-gaussian-splatting
cd 4d-gaussian-splatting
conda env create --file environment.yml
conda activate 4dgs
```

### Data preparation

**DyNeRF dataset:**

Download the [Neural 3D Video dataset](https://github.com/facebookresearch/Neural_3D_Video) and extract each scene to `data/N3V`. After that, preprocess the raw video by executing:

```shell
python scripts/n3v2blender.py data/N3V/$scene_name
```

**DNeRF dataset:**

The dataset can be downloaded from [drive](https://drive.google.com/file/d/19Na95wk0uikquivC7uKWVqllmTx-mBHt/view?usp=sharing) or [dropbox](https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip?dl=0). Then, unzip each scene into `data/dnerf`.


### Running

After the installation and data preparation, you can train the model by running:

```shell
python train.py --config $config_path
```

## 🎥 Videos

### 🎞️ Demo

[![Demo Video](https://i3.ytimg.com/vi/3cXC9e4CujM/maxresdefault.jpg)](https://www.youtube.com/embed/3cXC9e4CujM)

### 🎞️ Dynamic novel view synthesis

https://github.com/fudan-zvg/4d-gaussian-splatting/assets/45744267/5e163b88-4f70-4157-b9f5-8431b13c26b7

### 🎞️ Bullet time

https://github.com/fudan-zvg/4d-gaussian-splatting/assets/45744267/ac5bc3b2-dd17-446d-9ee6-6efcc871eb84

### 🎞️ Free view synthesis from a teleporting camera

https://github.com/fudan-zvg/4d-gaussian-splatting/assets/45744267/6bd0b57b-4857-4722-9851-61250a2521ab

### 🎞️ Monocular dynamic scene reconstruction

https://github.com/fudan-zvg/4d-gaussian-splatting/assets/45744267/2c79974c-1867-4ce6-848b-5d31679b6067

<img width="837" height="457" alt="image" src="https://github.com/user-attachments/assets/3e261e56-9bc5-4cac-b13c-63d2d4f286f4" />

여기서 부터 diff-gaussian-rasterzation-dubug 파일을 사용하셔야 합니다 폴더 뒤의 -dubug 삭제하고 폴더를 덮어 씌우세요 

# 렌더링 및 시뮬레이션 스크립트 가이드

이 문서는 `render_sort.py`와 `render_cycle.py` 스크립트의 사용법, 주요 차이점, 그리고 `sim/` 폴더의 시뮬레이션 도구와의 연관성을 설명합니다.

## 개요

이 프로젝트에는 세 가지 주요 렌더링 스크립트가 있습니다.
- `render.py`: 표준 렌더링을 위한 기본 스크립트입니다.
- `render_sort.py`: 다양한 정렬 알고리즘의 성능과 정확성을 분석하고 디버깅하기 위한 스크립트입니다.
- `render_cycle.py`: 제안하는 하드웨어 아키텍처의 성능(예: FPS)을 예측하기 위해, 렌더링 파이프라인의 상세한 로그를 생성하는 스크립트입니다.

## 스크립트 비교

| 기능 | `render_sort.py` | `render_cycle.py` |
| :--- | :--- | :--- |
| 주요 목적 | 정렬 알고리즘 분석 및 디버깅 | 하드웨어 성능 시뮬레이션을 위한 로그 생성 |
| 핵심 기능 | C++ / Python 정렬기 간 전환 | 렌더링 과정의 상세 정보(가우시안 ID, 깊이 등)를 CSV로 기록 |
| 디버그 출력 | 타일 별 가우시안 데이터 (`.npz` 형식) | 타일 별 가우시안 데이터 (`.csv` 형식), 컬링, 기여자 수 등 |
| 주요 플래그 | `--use_python_sorting`, `--resort_interval`, `--debug` | `--debug`, `--log_with_depth` |
| 사용 사례 | "Python으로 구현한 새로운 정렬 알고리즘이 C++ 정렬과 어떻게 다른지 확인하고 싶을 때" | "특정 씬을 렌더링할 때 제안하는 하드웨어에서 몇 FPS가 나올지 예측하고 싶을 때" |
| 출력 위치 | `renders_vanilla/`, `renders_npz_cpp/` 등 | `[씬이름]_[모드]_[타임스탬프]/` |

---

## 1. `render_sort.py` - 정렬 알고리즘 분석

이 스크립트는 가우시안 스플래팅의 핵심 단계인 정렬(sorting) 과정을 심층적으로 분석하고 디버깅하기 위해 설계되었습니다. C++ 기반의 고도로 최적화된 정렬과 Python 기반의 유연한 정렬 구현을 선택적으로 사용할 수 있습니다.

### 주요 기능

- 전환 가능한 정렬 (`--use_python_sorting`): 이 플래그를 사용하면 Python으로 구현된 정렬 로직을 활성화할 수 있습니다. 이를 통해 새로운 정렬 알고리즘을 쉽게 테스트하고 디버깅할 수 있습니다. 플래그가 없으면 기본 C++ 래스터라이저의 정렬을 사용합니다.
- 증분 재정렬 (`--resort_interval N`): Python 정렬 사용 시, `N` 프레임마다 전체 가우시안을 강제로 재정렬하여 시간적 일관성을 테스트할 수 있습니다.
- 상세 디버깅 (`--debug`): 정렬 직전의 가우시안 데이터를 타일 단위로 저장합니다. 각 타일에 어떤 가우시안들이 포함되고, 그들의 깊이(depth) 값이 무엇인지 `.npz` 파일로 기록하여 정렬 알고리즘의 입력값을 정확히 확인할 수 있습니다.

### 사용 예시

```shell
# Python 기반 정렬, 10프레임마다 재정렬, 디버그 모드 활성화
python render_sort.py \
    --config configs/dynerf/coffee_martini.yaml \
    --iteration 7000 \
    --use_python_sorting \
    --resort_interval 10 \
    --debug
```
- 결과:
  - 렌더링된 이미지는 `renders_incremental_resort_10/` 폴더에 저장됩니다.
  - 디버그 로그는 `renders_npz_python/` 폴더에 프레임별 `.npz` 파일로 저장됩니다.

---

## 2. `render_cycle.py` - 하드웨어 시뮬레이션 로그 생성

이 스크립트는 렌더링 파이프라인의 각 단계에서 어떤 일들이 일어나는지 매우 상세한 로그를 `.csv` 파일로 생성하는 데 특화되어 있습니다. 이 로그는 `sim/` 폴더의 하드웨어 성능 시뮬레이터의 입력으로 사용됩니다.

### 주요 기능

- 로그 생성 (`--debug`): 이 플래그를 활성화하면 렌더링 과정에서 발생하는 데이터를 로그 파일로 저장합니다.
- 로그 상세도 조절 (`--log_with_depth`):
  - 활성화 시: Python 기반 정렬 시뮬레이션을 위해 `Tile_ID, Gaussian_Count, Gaussian_IDs, Depths` 형식의 로그를 생성합니다. 깊이(depth) 정보가 포함되어 더 정확한 시뮬레이션이 가능합니다.
  - 비활성화 시 (기본값): CUDA 정렬을 가정하고 `Tile_ID, Gaussian_Count, Gaussian_IDs` 형식의 로그를 생성합니다.
- 다양한 로그 타입:
  - `unsorted/`: 정렬 전 각 타일에 할당된 가우시안 목록
  - `sorted/`: 정렬 후 각 타일에 할당된 가우시안 목록
  - `culling_log.csv`: 뷰 프러스텀 컬링 후 보이는 가우시안의 수
  - `pixel_contrib_summary/`: 각 픽셀에 기여하는 가우시안의 수

### 사용 예시

```shell
# 깊이 정보를 포함한 상세 로그 생성 (시뮬레이션용)
python render_cycle.py \
    -m output/N3V/sear_steak \
    --skip_train \
    --debug \
    --log_with_depth
```
- 결과: `sear_steak_log_with_depth_[타임스탬프]/` 와 같은 폴더가 생성되고, 그 안에 프레임별 `.csv` 로그 파일들이 저장됩니다.

---

## 3. `sim/` 폴더와의 연계성

`sim/` 폴더의 도구들은 `render_cycle.py`를 통해 생성된 로그를 사용하여 특정 하드웨어 아키텍처에서의 렌더링 성능(FPS)을 예측합니다.

### 워크플로우

1.  로그 생성: `render_cycle.py`를 실행하여 분석하고자 하는 씬의 `.csv` 로그를 생성합니다.
    ```shell
    python render_cycle.py -m [모델_경로] --debug --log_with_depth
    ```
2.  성능 분석: `sim/src/` 안의 `estimate_performance_our.py` 또는 `estimate_performance.py` 스크립트를 실행하여 로그를 분석하고 예상 성능을 계산합니다.
    ```shell
    # 이전 단계에서 생성된 로그 폴더를 -p 인자로 전달
    python sim/src/estimate_performance_our.py -p [로그_폴더_경로]
    ```

### 자동화된 시뮬레이션

`sim/run_simulation.ipynb` 노트북을 사용하면 위 1, 2번 과정을 자동으로 실행할 수 있습니다. 노트북 내의 경로 설정만 수정하고 전체 셀을 실행하면 최종 성능 예측 결과까지 한 번에 얻을 수 있습니다.

```
