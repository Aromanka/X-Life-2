# X-Life

当前维护主线是 v3 数值动作世界模型：独立饮食/运动残差动力学、三路门控融合、并列生理预测和独立排序头，以及 HTTP 知识生成与安全审查。

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

输出包含共享 context hash、每候选的动作/潜轨迹/预测/排名/安全结论、来源、失败原因和耗时。不足三个输出实际数量，无通过者输出空集。审查服务故障或格式错误拒绝候选，硬规则否决不能被排序或语义接受覆盖。

HTTP 动作 latent 分支需要 XLIFE_USE_LLM_ACTION_ENCODERS=1、匹配的 action_backend_id/维度及重新训练的投影；不能直接切换数值 checkpoint。详见 [HTTP 契约](docs/HTTP_INFERENCE.md)。

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

排序在同一 train.py 使用 --phase professional，再 --phase physiological，--data 传离线偏好对。单域与融合训练产生的随机 ranking 头不代表训练过的排名。

experiment.py 固定 participant 80/20，训练参与者内部留出晚期事件并清除六小时重叠窗口，分别报告 seen/unseen。归一化仅拟合训练事件。报告各模式/历史缺失/MAE/逐曲线 Pearson/预测时段/耗时/CUDA 峰值。默认 unseen 为零样本，--adapt-epochs 单独记录参与者事实微调与清除重叠后的晚期评估。

CBG/夜间仅实现显式时间锚点、单位和观测来源的任务校验，当前实际数据没有经确认的 bedtime/wake 标签，未训练或验证整夜预测。模型改动和验收边界见 [方法说明](docs/METHOD.md)。

旧 v2 agents、trainer 和模型配置迁至同级 extra_code/legacy_xlife 保存，不再进入默认导入路径。v2 checkpoint 不兼容。真实大模型、图谱完整性、五次全量研究实验和论文规模动作编码器训练须另行验证。
