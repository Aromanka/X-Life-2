# 论文动作编码路径

饮食与运动分别使用经过全参数监督微调的 Qwen2.5-VL-72B-Instruct。饮食接受文字、餐食图片或图片加文字，运动只接受文字。末层隐藏状态在输入 token 上均值池化，监督训练的 factor head 将其映射为结构化生理因素。世界模型中的投影头只读取这些因素，不直接读取隐藏状态。

饮食因素顺序为 carbohydrate/protein/fat/fiber/mass/energy，单位为 g/g/g/g/g/kcal；运动为 duration/energy/intensity，单位为 min/kcal/MET。运动训练标签必须统一为 MET，不混用 Borg 等量表。缺失的监督因素以 null 标记，只在已观测因素上计算损失。

## 编码器预训练

`train_action_encoder.py` 同时使用结构化注释的语言模型损失和生理因素的 log1p 均方误差，更新全部 VLM 参数及 factor head。注释 token 不参与动作隐藏状态池化，避免监督答案泄漏。投影头在后续世界模型单域训练中学习，融合训练和偏好对齐阶段冻结；远端 VLM 和 factor head 在世界模型训练期间始终冻结。

训练数据为 JSON 列表。图片路径相对于数据文件，服务推理图片路径位于服务所在文件系统。饮食样例：

```json
[{"text":"米饭、鸡肉和蔬菜", "images":["meal.jpg"],
  "annotations":{"ingredients":["rice","chicken","vegetables"],"preparation":"steamed","portion_g":350,"energy_kcal":400},
  "factors":[45,20,12,6,350,400]}]
```

运动样例：

```json
[{"text":"餐后20分钟开始步行，持续20分钟，约3 MET", "images":[],
  "annotations":{"type":"walking","intensity_MET":3,"duration_min":20,"meal_delay_min":20},
  "factors":[20,70,3],"intensity_scale":"MET"}]
```

在 X-Life-2 下分别训练两域，使用同一个发布版本标识：

```bash
accelerate launch 'world_model+ranking_module/train_action_encoder.py' \
  --domain diet --data diet_supervision.json --output outputs/diet_encoder \
  --backend-id qwen25vl-actions-sft-v1
accelerate launch 'world_model+ranking_module/train_action_encoder.py' \
  --domain exercise --data exercise_supervision.json --output outputs/exercise_encoder \
  --backend-id qwen25vl-actions-sft-v1
```

产物包含 VLM、processor 和 action_factors.pt。论文路径使用对应域的微调产物。没有 action_factors.pt 的原始 Qwen 目录走预训练模型测试路径：生成 JSON 生理因素，不使用随机初始化的 factor head，并标记 test_backend。

## 世界模型与推理

在工作区根目录启动现有角色服务，动作因素提取始终使用本地 Qwen 产物，其余角色沿用原设置：

```bash
python extra_code/server.py --backend api \
  --diet-encoder X-Life-2/outputs/diet_encoder \
  --exercise-encoder X-Life-2/outputs/exercise_encoder
```

在 X-Life-2 下启用论文路径并重新训练世界模型：

```bash
export XLIFE_USE_LLM_ACTION_ENCODERS=1
python 'world_model+ranking_module/train.py' --data ../data/xlife_cleaned_0207.json \
  --config configs/http_actions.example.json --phase single --domain both \
  --output outputs/actions_single
```

后续 fusion、professional、physiological 沿用原入口，不再阻止 HTTP 动作编码器进入偏好对齐。训练数据中的餐食图片放在 food_data.data.images 列表中；预生成候选和偏好动作使用 diet_images。候选饮食和运动描述分别使用 diet_text、exercise_text。

`POST /v1/action/factors` 接受 domain/text/images，返回 factors/backend_id，运动另返回 intensity_scale=MET。本地投影按固定单位缩放因素后生成 q，再叠加现有 onset 正弦编码。动作缺席时不请求 VLM，返回零动作向量。

默认数值模式仍可用于已有数值 checkpoint；旧 HTTP latent checkpoint 与新因素投影不兼容，需要重新训练。mock 服务仅返回标记为 test_backend 的固定因素，用于接口测试。

## 原始模型测试

当前 exp.sh 不微调大模型，使用 configs/pretrained_actions.json，仅训练本地投影与世界模型。metadata、饮食和运动角色均指向原始 Qwen2.5-VL 目录；从项目根目录启动服务：

```bash
python -u extra_code/server.py --backend api --device cuda \
  --metadata-model /data/home/wanglidi/code/model/Qwen2.5vl \
  --diet-encoder /data/home/wanglidi/code/model/Qwen2.5vl \
  --exercise-encoder /data/home/wanglidi/code/model/Qwen2.5vl
```

另一个终端执行 bash exp.sh。无需先生成微调编码器目录，训练过程不调用处方生成或安全审查 API。原始模型通过确定性解码生成定长因素数组，单位和顺序与微调路径相同；模型输出格式错误、缺失数量或非法数值时明确报错，不填充随机值。该路径只用于当前测试，不代表已完成论文的动作编码器监督微调。
