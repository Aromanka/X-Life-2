<p align="center">
  <img src="./src/xlife.png" width="200" alt="X-Life System Logo">
</p>

<h1 align="center">X-Life</h1>

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&amp;logoColor=white" alt="Python 3.10"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.7-EE4C2C?logo=pytorch&amp;logoColor=white" alt="PyTorch 2.7"></a>
  <a href="https://unity.com/"><img src="https://img.shields.io/badge/Unity-2022.3.x-000000?logo=unity&amp;logoColor=white" alt="Unity 2022.3"></a>
  <a href="./CITATION.cff"><img src="https://img.shields.io/badge/Citation-CITATION.cff-lightgrey" alt="Citation"></a>
</p>

**X-Life** is a metabolic world-model system for adaptive lifestyle medicine that translates continuous physiological data and sensing into personalised, quantitative dietary and postprandial activity prescriptions, concurrently meeting individual lifestyle preferences, contextual settings and safety constraints.

<a name="table-of-contents"></a>

## 📑 Table of Contents

- [🔎 Overview](#overview)
- [🚀 Quick Start](#quick-start)
- [🧩 Core Modules](#core-modules)
  - [Lifestyle Prescription Generation Module](#lifestyle-prescription-generation-module)
  - [Diet/Exercise Sub-world model and Ranking Module](#dietexercise-sub-world-model-and-ranking-module)
  - [Safety Module](#safety-module)
- [🥽 AR Deployment](#ar-deployment)
- [📊 minimum Dataset](#minimum-dataset)
- [📄 License](#license)

<a name="overview"></a>

## 🔎 Overview

This repository contains the Python code and minimum dataset for **"A metabolic world model system for personalised lifestyle medicine"**. The Unity AR client is distributed separately as `XLIFE-AR.zip`.


<a name="quick-start"></a>

## 🚀 Quick Start

### System Requirements

#### Software Dependencies

- **Python**: 3.10+
- **PyTorch**: >=2.7,<3
- **Unity3D**: 2022.3.x

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

Install the Python dependencies for the world model, ranking module and agents:

```bash
pip install -r requirements.txt
```

<a name="core-modules"></a>

## 🧩 Core Modules

### Lifestyle Prescription Generation Module

This module generates diet and exercise prescriptions from a hybrid graph and vector knowledge base.

#### Knowledge Graph Construction

Prepare guideline files and extract the knowledge graph:

```bash
python 'world_model+ranking_module/knowledge_cli.py' --config config.json \
  --input data/guidelines --output data/knowledge.json
```

### Diet/Exercise Sub-world model and Ranking Module

The sub-world models simulate glucose trajectories; the ranking module scores candidate prescriptions.

**1. Data Preparation**

Prepare participant metadata, meals, exercise records and CGM trajectories using [minimum_dataset/sample_data.json](./minimum_dataset/sample_data.json). The training and preference files below are user-prepared inputs.

Action-encoder training files contain `text`, optional `images`, `annotations` and `factors`, ordered according to `schema.actions` in `config.json`.

**2. Action Encoder Pre-training**

Pre-train the diet and exercise encoders separately using the Qwen2.5-VL-72B-Instruct backbone:

```bash
python 'world_model+ranking_module/train_action_encoder.py' --config config.json \
  --model Qwen/Qwen2.5-VL-7B-Instruct --domain diet \
  --data data/diet_annotations.json --output outputs/diet_encoder \
  --backend-id diet-sft-v1

python 'world_model+ranking_module/train_action_encoder.py' --config config.json \
  --model Qwen/Qwen2.5-VL-7B-Instruct --domain exercise \
  --data data/exercise_annotations.json --output outputs/exercise_encoder \
  --backend-id exercise-sft-v1
```

Configure the role service to use these saved encoders, and set `model.action_backend_id.diet` and `model.action_backend_id.exercise` to the matching backend IDs in `config.json`. Enable `http.use_llm_action_encoders` for these encoders.

**3. World Model Training**

Train both sub-world models, then train their fusion module:

```bash
python 'world_model+ranking_module/train.py' --config config.json \
  --phase single --domain both --data data/train.json --output outputs/single

python 'world_model+ranking_module/train.py' --config config.json \
  --phase fusion --data data/train.json \
  --init-checkpoint outputs/single/checkpoint.pt --output outputs/fusion
```

**4. Multi-stage Alignment**

Run health professional alignment, physiological response alignment and user feedback alignment in order:

```bash
python 'world_model+ranking_module/train.py' --config config.json \
  --phase ${ALIGNMENT_PHASE} --data data/${ALIGNMENT_PHASE}_pairs.json \
  --init-checkpoint outputs/fusion/checkpoint.pt --output outputs/professional

python 'world_model+ranking_module/train.py' --config config.json \
  --phase physiological --data data/physiological_pairs.json \
  --replay-data data/professional_pairs.json \
  --init-checkpoint outputs/professional/checkpoint.pt --output outputs/physiological

python 'world_model+ranking_module/train_user_feedback.py' --config config.json \
  --data data/user_feedback.json --observation-data data/user_observations.json \
  --expert-replay-data data/professional_pairs.json \
  --physiological-replay-data data/physiological_pairs.json \
  --init-checkpoint outputs/physiological/checkpoint.pt --output outputs/user
```

For physiological alignment, set `alignment.replay_fraction` between 0.2 and 0.3 and provide enough current pairs to realise that ratio. Each user feedback run uses feedback and measured observations from one participant.

**5. Prescription Generation and Ranking**

```bash
python 'world_model+ranking_module/inference.py' --config config.json \
  --data data/train.json --checkpoint outputs/user/checkpoint.pt \
  --knowledge data/knowledge.json --mode diet --ranking learned \
  --top-k 3 --output outputs/diet_recommendations.json
```

Set `--mode exercise` or `--mode joint` for exercise or combined prescriptions. Use each script's `--help` for available arguments.

### Safety Module

Configure the locally deployed DeepSeek-R1 model for the `safety` role in `config.json`. Safety review is included in the prescription inference pipeline.

The role server normalizes clinical context and candidate actions into safety input schema v1: `schema_version`, `as_of`, `mode` and `facts`. Each fact contains `name`, `value`, `unit`, `source` and optional `observed_at`. Rules and semantic review use these same facts; accepted responses return them under `inputs`.

Configure `safety.field_map` for source-specific metadata names and categories, `safety.glucose_unit` for historical CGM, and `safety.high_intensity_thresholds` for each supported intensity scale. `safety.required_facts` and `safety.max_glucose_age_minutes` control required inputs and glucose freshness. Missing inputs needed by an applicable rule reject that check. Dataset adapters and field mapping run in the external role server, not the world model.

<a name="ar-deployment"></a>

## 🥽 AR Deployment

The external Unity project targets **XREAL AR** devices. It embeds the web UI with **Vuplex 3D WebView**, captures images and video with the **XREAL RGB Camera**, and sends the results back to the web page through `postMessage`.

**Download:** [XLIFE-AR.zip](https://drive.google.com/file/d/1Ul-z-lDQSRvFw9aqk6uxIVhMamKpvAJM/view?usp=sharing) (complete Unity project).

### Hardware / Platform

- Target platform: Android
- Device: XREAL devices / runtimes that support XREAL XR Plugin (and any Android device that can install an APK)
- Debugging: USB cable + USB debugging enabled (required for Build & Run)


### Software Environment

- **Unity**: `2022.3.x` (this project uses `2022.3.61t8` in `ProjectSettings/ProjectVersion.txt`)
- **Unity Hub modules**: Android Build Support (SDK / NDK / OpenJDK)
- **Optional**: `adb` (usually not needed if you use Unity's Build & Run; needed for CLI install/debugging)

The Python and Neo4j modules are separate from the Unity client. The external `XLIFE-AR.zip` archive contains the Unity project; `Assets/`, `Packages/` and `ProjectSettings/` are not included here.

> Vuplex must be purchased and imported separately. The XREAL XR Plugin is included in the external archive.

<details>
<summary>📦 Dependencies and plugin installation</summary>

### Vuplex 3D WebView

- Purchase and import (Unity Asset Store): [Vuplex 3D WebView for Android (with Gecko Engine)](https://assetstore.unity.com/packages/tools/network/vuplex-3d-webview-for-android-with-gecko-engine-245867)
- After importing, you should see: `Assets/Vuplex/WebView/...`, and be able to use `CanvasWebViewPrefab` in scenes
- Note: if you want to run WebView directly in the macOS/Windows Editor, you may need to install the native plugins for those platforms per Vuplex docs; building for Android only is fine

### XREAL XR Plugin

- Included in the external `XLIFE-AR.zip` archive: `Packages/com.xreal.xr.tar.gz`
- The external Unity project's `Packages/manifest.json` references the local package: `"com.xreal.xr": "@com.xreal.xr.tar.gz"`
- To upgrade the plugin: replace `Packages/com.xreal.xr.tar.gz` and keep the `manifest.json` reference consistent

### Unity Package Manager Dependencies

After opening the project, Unity resolves the dependencies in `Packages/manifest.json` automatically.

</details>

<details>
<summary>🛠️ Project setup</summary>

### 1. Configure the WebView Initial URL

1. Open the scene: `Assets/Scenes/WebView.unity`
2. Select `CanvasWebViewPrefab` in the scene hierarchy
3. Set `InitialUrl` to your web URL
   - Current example value: `https://xlife.chat/MainPage/AR`

### 2. (Optional) Configure iFLYTEK Speech Recognition

Required for speech recognition.

The speech recognition config file is: `Assets/Resources/XunfeiConfig.json`

1. Register at [iFLYTEK Open Platform](https://www.xfyun.cn/) and create an application
2. Obtain your API credentials (appId, apiKey, apiSecret)
3. Edit `Assets/Resources/XunfeiConfig.json` and fill in:
   - `appId`: Your application ID
   - `apiKey`: Your API Key
   - `apiSecret`: Your API Secret

Alternatively, create a ScriptableObject config in Unity:

- Right-click in Project window → Create → XLIFE AR → Xunfei Speech Recognition Config
- Set the credentials in the Inspector
- Place the config asset in `Assets/Resources/` folder

</details>

<details>
<summary>📱 Build and export an Android APK</summary>

### Pre-build Checklist

- Vuplex is imported correctly (otherwise you will see compilation errors like missing `Vuplex.WebView` namespace)
- `Packages/com.xreal.xr.tar.gz` exists and Unity Package Manager shows no errors
- Scenes are included in Build Settings (this project should already be configured, but double-check):
  - `Assets/Scenes/WebView.unity`
  - `Assets/Scenes/RGBCameraAndCapture.unity`

### Build Steps in Unity (Android)

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

</details>

<a name="minimum-dataset"></a>

## 📊 minimum Dataset

The minimum dataset is available at [`./minimum_dataset/`](./minimum_dataset/). It contains three pseudonymous participants with:

- 2 meal events and four meal photographs;
- 2 exercise events with pre/post-event CGM trajectories;
- 2 nocturnal windows with rule-derived hypoglycemia labels;
- `manifest.json` with event counts and image hashes.

<a name="license"></a>

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
