# X-Life

当前维护主线是 v3 数值动作世界模型：独立饮食/运动残差动力学、三路门控融合、并列生理预测和独立排序头，以及 HTTP 知识生成与安全审查。

Metadata 统一使用 Qwen3-32B 语义编码后对齐 CGM 特征，自动解析嵌套字段与变长数组。数据接口、服务启动和 checkpoint 兼容性见 [Metadata 编码](docs/METADATA.md)。训练前需要启动该服务，或提供已编码 metadata；原数值 metadata checkpoint 需要重新训练。

在 X-Life-2 目录安装唯一依赖清单：`pip install -r requirements.txt`。

## 离线推理

先启动同级 extra_code 的角色服务器。提示词、模型名称和上游设置全部集中在 extra_code。应用只发送结构化 HTTP POST。完整服务器命令见同级 exp_1029_v2.sh。

```bash
export XLIFE_USE_LLM_ACTION_ENCODERS=0
export XLIFE_LLM_URL=http://127.0.0.1:8765
python 'world_model+ranking_module/inference.py' \
  --data ../data/xlife_cleaned_0207.json \
  --checkpoint outputs/exp_1029_v1/subject_fusion_d64/best.pt \
  --mode joint --ranking random --count 5 --output outputs/inference/joint.json
```

流程：真实事件屏蔽动作 → 独立需求生成候选 → 世界模型模拟 → 排序 → 硬规则与语义审查 → 安全 Top 3。生成端仅收到 subject_id、origin、历史 CGM/mask、metadata；原餐食/运动/动作文本/onset/未来标签均不传递。

- `--ranking learned` 默认要求已训练的对应模式排名，不回退到安全评分。`--ranking random` 明确标记 test_only。
- `--mode joint` 要求融合产物；单域 checkpoint 只能模拟已训练域。
- `--source-mode diet|exercise`、`--subject-id`、`--sample-index` 选择真实事件；候选模式可与来源独立。
- `--requirement requirement.json` 替换固定 seed 的随机结构化需求；`--candidates candidates.json` 接收已有候选；`--extract-actions` 调用 HTTP 显式提取。
- `--constraints constraints.json` 配置过敏、禁用运动、定量上限和联合时间约束；不从未命名 metadata 数组推断疾病。
- `--knowledge graph.json` 使用有来源的知识快照；缺图谱明确记录 knowledge_missing。

输出包含共享 context hash、每候选的动作/潜轨迹/预测/安全结论、来源、失败原因和耗时。不足三个输出实际数量，无通过者输出空集。处方格式、合理性和安全规则全部由同级 extra_code 服务实现；仓库内 generation/safety 仅发起 HTTP 请求。服务失联记录失败状态，最终推荐依据服务返回的决策。

处方处理端口为 `/v1/candidate/prepare`，完整安全评估端口为 `/v1/safety/evaluate`，与其余角色共用服务地址。

论文动作编码路径使用全参数微调的 Qwen2.5-VL、末层均值池化、监督生理因素提取和本地动作投影。启用 XLIFE_USE_LLM_ACTION_ENCODERS=1，配置匹配的 action_backend_id 并重新训练世界模型；不能直接切换数值或旧 HTTP latent checkpoint。餐食图片、编码器训练与服务启动见 [动作编码器](docs/ACTION_ENCODERS.md)。

## 训练与评估

```bash
python 'world_model+ranking_module/train.py' --data ../data/xlife_cleaned_0207.json \
  --config configs/numeric64.json --phase single --domain both \
  --epochs 60 --batch-size 16 --device cuda --output outputs/single
python 'world_model+ranking_module/train.py' --data ../data/xlife_cleaned_0207.json \
  --phase fusion --init-checkpoint outputs/single/best.pt \
  --epochs 40 --batch-size 16 --device cuda --output outputs/fusion
python 'world_model+ranking_module/experiment.py' --data ../data/xlife_cleaned_0207.json \
  --models persistence linear dlinear world --config configs/numeric64.json \
  --seeds 42 43 44 45 46 --epochs 10 --device cuda --output outputs/protocol
python -m unittest discover -s 'world_model+ranking_module' -p 'test_*.py'
```

排序采用标准 DPO：无界内部 logits 定义候选选择策略，针对冻结参考策略优化偏好对，结果只输出排序、不输出数值分数。在同一 train.py 使用 --phase professional，再 --phase physiological，--data 传离线偏好对。单域与融合训练产生的随机 ranking 头不代表训练过的排名。

experiment.py 固定 participant 80/20，训练参与者内部留出晚期事件并清除六小时重叠窗口，分别报告 seen/unseen。归一化仅拟合训练事件。报告各模式/历史缺失/MAE/逐曲线 Pearson/预测时段/耗时/CUDA 峰值。默认 unseen 为零样本，--adapt-epochs 单独记录参与者事实微调与清除重叠后的晚期评估。

CBG/夜间仅实现显式时间锚点、单位和观测来源的任务校验，当前实际数据没有经确认的 bedtime/wake 标签，未训练或验证整夜预测。模型改动和验收边界见 [方法说明](docs/METHOD.md)。

旧 v2 agents、trainer 和模型配置迁至同级 extra_code/legacy_xlife 保存，不再进入默认导入路径。v2 checkpoint 不兼容。真实大模型、图谱完整性、五次全量研究实验和论文规模动作编码器训练须另行验证。

Ranking 版本已更新为 2。旧 single/fusion 产物可用于初始化；旧有界分数对齐产物需要从 fusion 重新训练排序头，不能继续原对齐训练。
## 服务端监督与检索 schema

辅助监督与个体化知识检索统一从服务的 `POST /v1/schema` 读取策略。服务通过 `--schema` 选择 JSON 文件；字段路径、目标形状、损失权重和检索查询来源均在服务端设定，见 [RUNTIME_SCHEMA.md](docs/RUNTIME_SCHEMA.md)。

