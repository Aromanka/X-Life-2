<p align="center">
  <img src="./src/xlife.png" width="300" alt="X-Life System Logo">
</p>

<h1 align="center">X-Life</h1>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.7](https://img.shields.io/badge/PyTorch-2.7-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Unity 2022.3](https://img.shields.io/badge/Unity-2022.3.x-000000?logo=unity&logoColor=white)](https://unity.com/)
[![Citation](https://img.shields.io/badge/Citation-CITATION.cff-lightgrey)](./CITATION.cff)

<p align="center">
  <strong>X-Life</strong> is a metabolic world-model system for adaptive lifestyle medicine that translates continuous physiological data and sensing into personalised, quantitative dietary and postprandial activity prescriptions, concurrently meeting individual lifestyle preferences, contextual settings and safety constraints.
</p>

## 📋 Table of Contents

- [📋 Table of Contents](#-table-of-contents)
- [🌟 Overview](#-overview)
- [🚀 Quick Start](#-quick-start)
  - [System Requirements](#system-requirements)
    - [Software Dependencies](#software-dependencies)
    - [Hardware Requirements](#hardware-requirements)
  - [Installation](#installation)
    - [1. Clone the Repository](#1-clone-the-repository)
    - [2. Create Python Environment](#2-create-python-environment)
    - [3. Install Python Dependencies](#3-install-python-dependencies)
    - [4. Install Neo4j Service](#4-install-neo4j-service)
- [🏗️ Core Modules](#️-core-modules)
  - [1. 🕸️ Lifestyle Prescription Generation Module](#1-️-lifestyle-prescription-generation-module)
    - [Configuration Steps](#configuration-steps)
    - [Test Pipelines](#test-pipelines)
  - [2. 🧠 Diet/Exercise Sub-world model and Ranking Module](#2--dietexercise-sub-world-model-and-ranking-module)
  - [3. 🛡️ Safety Module](#3-️-safety-module)
  - [4. 🧬 Multi-omics Model](#4--multi-omics-model)
  - [5. 🥽 AR Deployment](#5--ar-deployment)
    - [Hardware / Platform](#hardware--platform)
    - [Software Environment](#software-environment)
    - [Dependencies](#dependencies)
      - [A. Vuplex 3D WebView (Required, Not Included)](#a-vuplex-3d-webview-required-not-included)
      - [B. XREAL XR Plugin (Required, Already Configured)](#b-xreal-xr-plugin-required-already-configured)
      - [C. Unity Package Manager Dependencies (Auto-installed)](#c-unity-package-manager-dependencies-auto-installed)
    - [Project Setup](#project-setup)
      - [4.1 Configure the WebView Initial URL](#41-configure-the-webview-initial-url)
      - [4.2 (Optional) Configure iFLYTEK Speech Recognition](#42-optional-configure-iflytek-speech-recognition)
    - [Build APK](#build-apk)
      - [Pre-build Checklist](#pre-build-checklist)
      - [Build Steps in Unity (Android)](#build-steps-in-unity-android)
    - [Troubleshooting](#troubleshooting)
- [📊 Minimal Dataset](#-minimal-dataset)
- [📄 License](#-license)

## 🌟 Overview

This repository contains the Python code and sample dataset for the paper **"A metabolic world model system for personalised lifestyle medicine"**. The Unity AR client is distributed separately as the external `XLIFE-AR.zip` archive described below.


## 🚀 Quick Start

### System Requirements

#### Software Dependencies
- **Python**: 3.10+
- **PyTorch**: 2.7.0
- **JDK**: 17.0.18+
- **Unity3D**: 2022.3.x
- **Neo4j**: 5.26.21+

#### Hardware Requirements
- **Training**: NVIDIA A800 or equivalent GPU
- **AR Deployment**: Android 14+ compatible device

### Installation

#### 1. Clone the Repository
```bash
git clone --branch fix/xlife-audit-11-items https://github.com/LittleQBerry/X-Life.git
cd X-Life
```

#### 2. Create Python Environment
```bash
# Using conda (recommended)
conda create -n xlife python=3.10 -y
conda activate xlife
```

#### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Install Neo4j Service
```bash
# Install JDK
sudo apt install openjdk-17-jdk
# Verify JDK version
java -version
```

Download the Neo4j Community 5.26.21 tarball from the [Neo4j Deployment Center](https://neo4j.com/deployment-center/?gdb-selfmanaged) and start the service:
```bash
tar zxf neo4j-community-5.26.21-unix.tar.gz
cd neo4j-community-5.26.21/bin/
./neo4j start
```

## 🏗️ Core Modules


### 1. 🕸️ Lifestyle Prescription Generation Module

The knowledge-graph-guided lifestyle prescription generation module generates personalized diet and exercise prescriptions by grounding LLM outputs in a hybrid vector-graph metabolic knowledge base, safeguarded by a safety module that enforces clinical safety through semantic auditing and deterministic constraints.

#### Configuration Steps

**1. Configure Database and Models**
Configure Neo4j database, the remote LLM API, and optional local model paths in `./kg_agents/config.json`:

```json
{
    "neo4j": {
        "uri": "bolt://127.0.0.1:7687",
        "username": "your_username",
        "password": "your_password"
    },
    "api_model": {
        "api_key": "your_api_key",
        "base_url": "your_base_url",
        "model": "your_api_model"
    },
    "local_model_path": "your_local_LLM_path",
    "local_emb_path": "your_local_embedding_model_path"
}
```

**2. Prepare Guidelines**
Prepare guidelines following the descriptions in our paper:
```
kg_agents/
└── data/
    ├── diet/
    │   ├── diet_guideline_1.pdf
    │   └── ...
    └── exer/
        ├── exercise_guideline_1.pdf
        └── ...
```

**3. Extract and Embed Entities**
Make sure the Neo4j server has started:
```bash
cd kg_agents
# Extract knowledge from ./data
python -m core.build_kg
# Import knowledge graph
python -m core.import_kg
# (Optional) Embedding knowledge graph
python -m core.embed_kg
```

#### Test Pipelines

**Generation and Assessment Pipelines**
```bash
# Run these commands from the repository root.
cd kg_agents
# Test diet prescription pipeline
python -m pipeline.diet_pipeline --bn 1 --vn 5 --query "I want a sandwich with just veggies, no meat." --use_vector --rag_topk 5
# Test exercise prescription pipeline
python -m pipeline.exer_pipeline --bn 1 --vn 4 --query "I want to do some back exercises at the gym." --use_vector --rag_topk 5
```

**Start Flask Service**
```bash
cd kg_agents
python server.py
```

**API Interface Testing**
```bash
# Diet prescription generation
curl -X POST http://localhost:5000/api/v1/diet/generate-only \
  -H "Content-Type: application/json" \
  -d '{"user_metadata":{},"user_requirement":{},"user_query":"a balanced lunch"}'

# Exercise prescription generation
curl -X POST http://localhost:5000/api/v1/exercise/generate-only \
  -H "Content-Type: application/json" \
  -d '{"user_metadata":{},"user_requirement":{},"user_query":"a moderate walk"}'

# Safety module assess prescription
curl -X POST http://localhost:5000/api/v1/safety/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "plan_type": "diet",
    "user_metadata": {},
    "plan": {}
  }'
```

For detailed prompt sample of prescription generation, prescription assessment and knowledge graph extraction, you can refer to [kg_agents/sample_prompts.md](./kg_agents/sample_prompts.md)

### 2. 🧠 Diet/Exercise Sub-world model and Ranking Module

Diet/Exercise sub-world models predict latent metabolic states for diet and exercise candidates. The public model now exposes separate action-conditioned transition modules with trajectory and dynamics branches, cross-attention fusion, a future glucose head, and a quantitative `ranking_score` output.

1. Data Preparation.
Prepare the user metadata and physiological trajectory in the following format:
```bash
[
    {
        "subject_id": "XXX",
        "subject_name": "XXX",
        "metadata": {
        },
        "meals": [
          {
            "food_data": {
              "food_id": "XXXX",
              "date":"XXXX" ,
              "meal_type":"XXXX" ,
              "data": {
                "time":,
                "food_items": [
                  {
                    "img_names":
                  }
                ]
              }
            },
            "exercise_data_after_meal": {},
            "cgm_preprandial": [],
            "cgm_postprandial": [],
            "sleep_data": {}
          }
        ],
        "exercise_events": [
          {
            "event_id": "exercise_001",
            "recorded_at": "YYYY-MM-DDTHH:MM:SS",
            "cgm_pre_exercise": [],
            "cgm_post_exercise": []
          }
        ],
        "nocturnal_hypoglycemia": [
          {
            "date": "XXXX",
            "cgm_pre_sleep": [],
            "cgm_sleep": []
          }
        ]
      }
]
```
The bundled minimal dataset is under `./minimal_dataset/`. It contains the current de-identified review fixture: three pseudonymous participants, two meals, two exercise events, two nocturnal windows, four meal photos and the corresponding observed CGM trajectories. `SampleImageDataset` reads meal photos from `minimal_dataset/images/` and also converts top-level `exercise_events` with CGM windows into standalone exercise samples. The raw meal/exercise windows contain 12 pre-event and 36 post-event readings; the current model example uses a fixed 32-point target length (`max_future_len=32`), so the raw JSON retains more post-event points than the default model input. The bundle is a format and smoke-test fixture; it is not sufficient for model training, which requires at least two participants and a project-specific training split.

2.Model config. In `world_model+ranking_module/model_config.py`, the default research configuration resolves to `Qwen/Qwen3-8B` for text features and `Qwen/Qwen2.5-VL-72B-Instruct` for the diet image-language encoder. The download helpers accept the selected configuration and pass the resolved local paths through to the model. The `cnn-backup` vision option is intended for lightweight verification and is not the research model.
```bash
vim world_model+ranking_module/model_config.py
```
3.Training. The `train()` entry point supports the manuscript's explicit training phases through `training_phase`: `encoder_pretraining`, `world_model`, `preference_alignment`, and `participant_update`. The world-model phase freezes the action encoders and trains transition, fusion, decoder and ranking modules. Data splitting is participant-level with five reproducible seeds; all samples from a participant stay in one partition. `split_index` selects one recorded split; running and aggregating all five splits remains an experiment-level step. CGM padding is tracked with validity masks and excluded from loss and metrics. The bundled `minimal_dataset` contains three participants and is a format/smoke-test fixture with only two meals and two exercise events; it is not a statistically meaningful training cohort. The checked-in `main.py` uses the `default` research configuration, which may download Qwen3-8B and Qwen2.5-VL-72B weights; edit the `get_training_config(...)` and path assignments in `main.py` to select another configuration or dataset. Users can change `image_dir`, `data_path`, `save_dir` and the desired training phase before running. From the repository root:
```bash
cd world_model+ranking_module
python main.py
```

4.Evaluation. `python main.py` runs training and then calls `trainer.evaluate()` on the validation split; it is not a checkpoint-only evaluation command. A caller that constructs `CGMPredictionTrainerWithImage` with a saved `model_path` can call `trainer.evaluate()` directly to evaluate an existing checkpoint. The evaluation returns predicted CGM values plus MSE, MAE, RMSE, R² and Pearson metrics.
```bash
cd world_model+ranking_module
python main.py
```

5.Multi-stage alignment. `CGMPredictionTrainerWithImage.dpo_loss()` implements the score-margin objective for preferred/rejected prescriptions from the same participant state, with an optional frozen reference-model margin. The public sample dataset does not contain preference pairs, so preference alignment requires a project-specific dataset and caller that supplies those pairs; the helper does not treat safety scores as learned preference scores.

When a trained world model is available, inject a callable as `world_model_scorer` into `DietPipeline` or `ExercisePipeline` (or pass it to `generate`). The pipeline first removes candidates with `is_safe=False`, then stores the learned value as `_ranking_score` and ranks only the remaining candidates. Without an injected scorer, it uses the safety score as an explicit fallback and does not claim to have learned a physiological ranking.

The transition, trajectory, dynamics, fusion, ranking, training-phase and DPO additions provide stable interfaces and a small differentiable reference path for integration tests. They are not a claim that the paper's unpublished losses, checkpoints or preference data have been reconstructed. Exact CGM patch settings, auxiliary omics decoders, encoder-pretraining labels, preference-pair construction/reference checkpoints, and the participant-update data contract still need to be supplied for a paper-faithful reproduction.


### 3. 🛡️ Safety Module

A safety module based on retrieval-augmented generation is designed to enforce safety constraints and reduce hallucination risk of lifestyle prescriptions. The runtime model is the API model configured in `kg_agents/config.json`; Qwen3-8B is the manuscript reference model rather than a hard-coded dependency. The designed system prompt is in [kg_agents/sample_prompts.md](./kg_agents/sample_prompts.md).

### 4. 🧬 Multi-omics Model

The model supports multi-omics-based glucose prediction. From the repository root, run:
```bash
python omics_model/train_omics_modality_models.py \
  --labels labels.csv \
  --covariates covariates.csv \
  --metagenome metagenome.csv \
  --lipid_metabolome lipid_metabolome.csv \
  --proteome proteome.csv \
  --outdir results_omics_models \
  --key_col case_id \
  --group_col subject_id \
  --day_col day_index \
  --target_col iAUC_2h_true_baseline_sub_pos_dx5 \
  --test_size 0.2 \
  --seed 42 \
  --lgb_n_splits 5 \
  --lgb_n_estimators 2000 \
  --lgb_top_k 300 \
  --optuna_trials 200 \
  --calib_bins 10 \
  --calib_strategy quantile
```

### 5. 🥽 AR Deployment

The external Unity project targets **XREAL AR** devices to:
- Embed a web page as the main UI via **Vuplex 3D WebView** (WebView scene)
- Trigger **XREAL RGB Camera** capture / screen recording from the web page (camera capture scene)
- Send capture results (photos/videos, local file paths / local HTTP URLs, etc.) back to the web page via `postMessage`

#### Hardware / Platform

- Target platform: Android
- Device: XREAL devices / runtimes that support XREAL XR Plugin (and any Android device that can install an APK)
- Debugging: USB cable + USB debugging enabled (required for Build & Run)


#### Software Environment

- **Unity**: `2022.3.x` (this project uses `2022.3.61t8` in `ProjectSettings/ProjectVersion.txt`)
- **Unity Hub modules**: Android Build Support (SDK / NDK / OpenJDK)
- **Optional**: `adb` (usually not needed if you use Unity's Build & Run; needed for CLI install/debugging)
- **Download**: Please download the complete Unity project [XLIFE-AR.zip](https://drive.google.com/file/d/1Ul-z-lDQSRvFw9aqk6uxIVhMamKpvAJM/view?usp=sharing)

#### Dependencies

The Python/Neo4j modules in this repository are separate from the Unity client. The Unity project is distributed as the external `XLIFE-AR.zip` archive linked below; Unity `Assets/`, `Packages/`, and `ProjectSettings/` are not included in this repository.

> The two most common blockers are: **Vuplex (commercial, not included in the repo)** + **XREAL XR Plugin (provided as a local tar package)**.

##### A. Vuplex 3D WebView (Required, Not Included)

- Purchase and import (Unity Asset Store): [Vuplex 3D WebView for Android (with Gecko Engine)](https://assetstore.unity.com/packages/tools/network/vuplex-3d-webview-for-android-with-gecko-engine-245867)
- After importing, you should see: `Assets/Vuplex/WebView/...`, and be able to use `CanvasWebViewPrefab` in scenes
- Note: if you want to run WebView directly in the macOS/Windows Editor, you may need to install the native plugins for those platforms per Vuplex docs; building for Android only is fine

##### B. XREAL XR Plugin (Required, Provided in the External Unity Archive)

- Included in the external `XLIFE-AR.zip` archive: `Packages/com.xreal.xr.tar.gz`
- The external Unity project's `Packages/manifest.json` references the local package: `"com.xreal.xr": "@com.xreal.xr.tar.gz"`
- To upgrade the plugin: replace `Packages/com.xreal.xr.tar.gz` and keep the `manifest.json` reference consistent

##### C. Unity Package Manager Dependencies (Auto-installed)

After opening the project, Unity resolves dependencies from `Packages/manifest.json` automatically (e.g., Input System / AR Foundation / XR Interaction Toolkit). In most cases you do not need to install anything manually in Package Manager.

#### Project Setup

##### 4.1 Configure the WebView Initial URL

1. Open the scene: `Assets/Scenes/WebView.unity`
2. Select `CanvasWebViewPrefab` in the scene hierarchy
3. Set `InitialUrl` to your web URL
   - Current example value: `https://xlife.chat/MainPage/AR`

##### 4.2 (Optional) Configure iFLYTEK Speech Recognition

**REQUIRED if you want to use speech recognition features.**

The speech recognition config file is: `Assets/Resources/XunfeiConfig.json`

1. Register at [iFLYTEK Open Platform](https://www.xfyun.cn/) and create an application
2. Obtain your API credentials (appId, apiKey, apiSecret)
3. Edit `Assets/Resources/XunfeiConfig.json` and fill in:
   - `appId`: Your application ID
   - `apiKey`: Your API Key
   - `apiSecret`: Your API Secret

Alternatively, you can create a ScriptableObject config in Unity Editor:

- Right-click in Project window → Create → XLIFE AR → Xunfei Speech Recognition Config
- Set the credentials in the Inspector
- Place the config asset in `Assets/Resources/` folder

#### Build APK

##### Pre-build Checklist

- Vuplex is imported correctly (otherwise you will see compilation errors like missing `Vuplex.WebView` namespace)
- `Packages/com.xreal.xr.tar.gz` exists and Unity Package Manager shows no errors
- Scenes are included in Build Settings (this project should already be configured, but double-check):
  - `Assets/Scenes/WebView.unity`
  - `Assets/Scenes/RGBCameraAndCapture.unity`

##### Build Steps in Unity (Android)

1. `File -> Build Settings...`
2. Select `Android`, then click `Switch Platform`
3. In `Scenes In Build`, make sure `WebView` and `RGBCameraAndCapture` are enabled (recommended: put `WebView` first)
4. `Edit -> Project Settings -> XR Plug-in Management`
   - In the `Android` tab, enable **XREAL** (XREAL XR Loader)
5. `Player Settings -> Other Settings` (recommended values; adjust as needed)
   - `Scripting Backend`: IL2CPP
   - `Target Architectures`: ARM64
6. Connect an Android device (Developer Mode + USB debugging enabled), then click `Build And Run`
   - Or click `Build` to export an `.apk` and install it manually

> Note: this project includes a custom manifest at `Assets/Plugins/Android/AndroidManifest.xml`, which declares network/microphone/screen-capture related permissions, including the Android 14+ foreground service permission `FOREGROUND_SERVICE_MEDIA_PROJECTION`.

#### Troubleshooting

- `The type or namespace name 'Vuplex' could not be found`: Vuplex 3D WebView is not imported
- WebView is blank / cannot load: check `InitialUrl`, device network, and whether cleartext HTTP is allowed (if using `http`)
- Camera / screen recording does not work in Editor: XREAL capture APIs typically only work on real Android devices (there is also a mock branch in scripts)
- Screen recording issues on Android 14+: make sure you did not overwrite/remove the foreground service permission declarations in `Assets/Plugins/Android/AndroidManifest.xml`


## 📊 Minimal Dataset

The current de-identified minimal dataset is available at [`./minimal_dataset/`](./minimal_dataset/). It contains three pseudonymous participants and the following observed records:

- 2 meals, including four meal photographs and pre/post-meal CGM windows;
- 2 exercise events, including observed post-event CGM windows;
- 2 nocturnal windows with rule-derived positive/negative hypoglycemia labels;
- `manifest.json` with event counts and SHA-256 hashes for the bundled images.

The JSON bundle keeps the repository's patient-level layout (`subject_id`, `metadata`, `meals`, and `nocturnal_hypoglycemia`) and adds top-level `exercise_events` plus explicit CGM windows. `recorded_at` is a proxy event time; it is not a verified first-bite or exercise-start time. Night labels are rule-derived CGM labels, not clinician-adjudicated diagnoses. The bundled photographs and trajectories are provided for code verification and functional demonstration, subject to the release permissions documented in [`minimal_dataset/README.md`](./minimal_dataset/README.md).

Validate the bundle from the repository root with:

```bash
python3 minimal_dataset/validate_bundle.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
