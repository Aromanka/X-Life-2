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
      - [B. XREAL XR Plugin (Required, Provided in the External Unity Archive)](#b-xreal-xr-plugin-required-provided-in-the-external-unity-archive)
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
git clone https://github.com/2025XLife/X-Life.git
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
The maintained role service reads model paths, API role settings, device assignments and supervision/retrieval schemas from [example_config.json](./example_config.json). Set the API credential in the environment variable named by the configuration, then start the service from the workspace root:

```bash
python extra_code/server.py --config X-Life-2/example_config.json
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
python -m core.build_kg_2_steps
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

Set `world_model.checkpoint_path` to the trained ranking checkpoint. For `/diet/generate` and `/exercise/generate`, include the current event as `world_model_context` with `subject_id`, `metadata`, `sleep_data` and `cgm_pre`.

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

Diet/Exercise sub-world models simulate future glucose trajectories and rank candidate diet and exercise prescriptions using lifestyle records and physiological data.

**1. Data Preparation**

Prepare participant metadata, meal images and descriptions, exercise records and CGM trajectories following [minimal_dataset/sample_data.json](./minimal_dataset/sample_data.json). Provide the corresponding training annotations and preference pairs for encoder pre-training and alignment.

**2. Model Configuration**

Edit [example_config.json](./example_config.json) to configure the model, data windows, training, alignment, inference and external role service. All maintained Python entry points read this file relative to the repository, independently of the current working directory. Use `--config /path/to/config.json` or `XLIFE_CONFIG` to select another complete configuration. Missing required fields raise an error; Python code does not supply model or training parameter values.

```bash
python 'world_model+ranking_module/train.py' --config example_config.json --help
```

> **Note** The Qwen2.5-VL-72B-Instruct backbone used for diet-action representation is supervised fine-tuned on structured dietary information, including nutrition analysis, ingredients and preparation methods. The resulting diet-information-tuned model (together with its projection head) is the intended diet action encoder.
**3. Training**

Run the following commands from the repository root with your training data:

Start the role server in a separate terminal first. Action models load on demand, so the service can run during annotation and encoder training; both action artifact directories must exist before world-model training.

```bash
python ../extra_code/server.py --config example_config.json
```

```bash
python 'world_model+ranking_module/encode_annotations.py' \
  --input data/annotations.json --output data/train.json \
  --config example_config.json

python 'world_model+ranking_module/train_action_encoder.py' --config example_config.json \
  --domain diet --data data/diet_annotations.json --backend-id diet-encoder-v1 --output outputs/diet_encoder

python 'world_model+ranking_module/train_action_encoder.py' --config example_config.json \
  --domain exercise --data data/exercise_annotations.json --backend-id exercise-encoder-v1 --output outputs/exercise_encoder

python 'world_model+ranking_module/train.py' --config example_config.json \
  --phase single --data data/train.json --output outputs/world
```

**4. Multi-stage Alignment**

Recommendation requirements use explicit `--requirement` JSON first, then the optional `inference.requirement` object. When absent or null, the client calls `POST /v1/requirement/default` with mode, context and constraints. The role service returns `{"requirement": {...}}` from `server.default_requirements[mode]` in the standard result/provenance envelope. The example service defaults are 500 kcal for diet, 20 minutes for exercise, and both for joint actions; configure these values for your deployment. Requirements are never randomly generated. HTTP failures produce `method_status=requirement_failed` with no recommendations.

Train fusion after both single domains, then align the ranking head through professional and physiological preferences. User feedback has a separate entry point:

```bash
python 'world_model+ranking_module/train.py' --config example_config.json \
  --phase fusion --data data/train.json --init-checkpoint outputs/world/checkpoint.pt --output outputs/fusion

python 'world_model+ranking_module/train.py' --config example_config.json \
  --phase professional --data data/professional_pairs.json --init-checkpoint outputs/fusion/checkpoint.pt \
  --output outputs/professional
```

For physiological alignment, use `--phase physiological`, initialize from the professional checkpoint and supply `--replay-data`. For user feedback, run `train_user_feedback.py --config example_config.json` with the physiological checkpoint and both replay datasets.

The ranking head produces unbounded logits. All alignment stages compute DPO from policy and frozen-reference log-softmax probabilities; their pairwise differences equal logit differences. At inference, `model.rank` returns softmax scores over one complete candidate set sharing the same user state. The ranking CLI collects logits across batches before normalization and writes `scores` in candidate input order. Recommendation output exposes `ranking_score` for learned ranking, normalized across all successfully simulated candidates before safety filtering and Top-K selection. Scores sum to one over that set and depend on its composition; they are not absolute suitability or safety probabilities. Internal logits remain unbounded and are not exported. Ranking v2 checkpoint weights and alignment resume semantics are unchanged.

Training options are available through `python 'world_model+ranking_module/train.py' --help`. LLM action encoders are enabled by default. The default server loads `outputs/diet_encoder` and `outputs/exercise_encoder` relative to the selected configuration file, using the separately trained artifacts above. `model.action_backend_id` specifies each domain's expected artifact ID; existing configurations with a shared string ID remain supported. Configure model paths, role settings, device assignments and supervision/retrieval schemas before starting `../extra_code/server.py --config example_config.json`. Missing artifacts or mismatched IDs raise an error. Existing numeric-action world-model checkpoints require explicitly setting `http.use_llm_action_encoders` to false (or `XLIFE_USE_LLM_ACTION_ENCODERS=0`); using the LLM action representation requires training the world model with that representation. API credentials are read only from the environment variable named in the configuration.

### 3. 🛡️ Safety Module

A safety module based on retrieval-augmented generation enforces safety constraints for lifestyle prescriptions. Configure its API role and retrieval schema in [example_config.json](./example_config.json). The maintained prompts and review service are in the sibling `extra_code` directory.

Knowledge retrieval uses the manuscript's Qwen3-Embedding-8B by default. Configure `server.embedding_model` (a Hugging Face model ID or local directory), `server.embedding_device`, `server.embedding_max_length` and `server.embedding_query_instruction` in the same configuration file. Both knowledge retrieval dimensions are 4096; the world-model dimension remains independent. The service uses query instructions, last-token pooling and normalized embeddings as documented in the [official model card](https://huggingface.co/Qwen/Qwen3-Embedding-8B). Graph entity embeddings are indexed in memory and queried by cosine similarity. `--backend mock` is an explicitly marked test path; the normal service does not replace encoder failures with random vectors.

### 4. 🧬 Multi-omics Model

Structured omics annotations provide auxiliary supervision during world-model training. The omics prediction head is removed during inference. See [docs/METHOD.md](./docs/METHOD.md) for the training procedure.

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

The minimal dataset is available at [`./minimal_dataset/`](./minimal_dataset/). It contains three pseudonymous participants with:

- 2 meal events and four meal photographs;
- 2 exercise events with pre/post-event CGM trajectories;
- 2 nocturnal windows with rule-derived hypoglycemia labels;
- `manifest.json` with event counts and image hashes.

Data descriptions are provided in [minimal_dataset/README.md](./minimal_dataset/README.md).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
