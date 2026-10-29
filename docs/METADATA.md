# Metadata 语义编码

统一流程是 metadata 字段 → Qwen3-32B 语义向量 → 可训练对齐投影 → 与 CGM 特征拼接。模型主干不指定队列来源、原始字段数量或语义向量宽度。

`dataset_metadata.py` 定义项目共享的数据接口。metadata 可以是嵌套 JSON 对象，包含数值、文本、布尔值、null 和任意长度数组；解析器自动保留字段路径与数组位置，不限定字段名，也不截断字段数量。形如 `{"value":6.1,"unit":"%"}` 的测量会保留单位。没有语义名称的数组位置保留为 `group[0]`，不会猜测医学名称或单位。

也可直接提供统一字段格式：

```json
{"fields":[
  {"name":"clinical.HbA1c","value":6.1,"unit":"%"},
  {"name":"sleep.quality","value":"poor","unit":""},
  {"name":"omics.marker","value":null,"unit":""}
]}
```

字段名必须唯一，值为标量或 null，单位为字符串。嵌套格式自动转换为此格式，类别文本保留，非有限数值转换为 null。`dataset.py` 从参与者的 metadata 入口取数据，各字段的来源及长度均由数据本身决定；有有效事件的参与者只编码一次，供其事件复用。接入其他队列时，dataset 适配器调用同一 `prepare_metadata` 接口，无需修改世界模型。

服务端 `POST /v1/metadata/encode` 接受统一字段格式，使用冻结的 Qwen3-32B 对标准化文本做编码。有效 token 的末层隐藏状态取均值作为语义向量。模型实际输出决定向量宽度，不由客户端规定。启动服务：

```bash
python extra_code/server.py --backend api --metadata-model Qwen/Qwen3-32B
```

该角色使用本地语义模型，与生成和安全审查的 API 模型独立。可将模型名替换为本地 Qwen3-32B 目录。mock 模式仅用于接口验证，返回带 test_backend 标记的测试向量；真实服务失败时不会替换成测试结果。

编码后的数据接口是：

```json
{"fields":[{"name":"clinical.HbA1c","value":6.1,"unit":"%"}],
 "embedding":[0.1,0.2,0.3],"backend_id":"Qwen/Qwen3-32B","test_backend":false}
```

示例向量仅用于说明格式。向量必须非空且有限。已编码数据可直接使用，不会重复请求服务。训练、候选推理和偏好对齐使用相同接口。生成和安全审查仍能读取原始字段，缺失情况从字段值判断，不从 embedding 判断。

对齐层使用 LazyLinear 自动推断输入宽度；checkpoint 保存投影权重、编码器标识和测试标记。恢复时自动还原宽度，拒绝混用不同编码器标识或不同向量宽度。单域世界模型训练更新对齐投影，融合与偏好对齐期间冻结，Qwen3 始终冻结。

原数值 metadata checkpoint 与该语义路径不兼容，需要重新训练。数字动作模式也使用新的 metadata 语义编码，不再走固定长度原始数值编码。
