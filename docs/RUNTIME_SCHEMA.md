# Server schema

启动 `python extra_code/server.py --schema extra_code/schema.json`。客户端通过同一服务端口的 `POST /v1/schema`（请求 `{}`）读取 schema，地址沿用 `XLIFE_LLM_URL`。不提供本地 schema 回退。修改文件后重启服务。

schema 包含 `supervision` 和 `retrieval`。仓库内的通用消费逻辑不指定辅助模态名称、字段或维度；随服务提供的 schema 默认保留 CGM 监督，并用完整状态、需求和约束进行向量检索。CGM 历史、动作及时间窗口仍沿用现有世界模型主干和数据适配器，本次统一的是辅助监督与知识检索策略。

## 监督

每个目标声明 `name`、`path`、`shape`、`readout`、`loss`、`weight`。路径是字典键和数组索引组成的列表，可从 `patient`、`row` 或适配后的 `sample` 取值。`mask` 可指定观测掩码路径；时间掩码会广播到特征维度。不存在的目标、null 和非有限值不参与损失；整个 batch 没有有效目标则报错。

`readout: sequence` 使用逐步轨迹，shape 的第一维必须等于预测步数；`pooled` 使用轨迹均值。未指定 `prediction` 时，为每个目标、每个动力学分支建立对应大小的线性 decoder。`prediction` 可引用已有模型输出。损失支持 `mse` 和 `l1`，各目标在有效元素上取均值后按权重求和。`scale` 可为正数或已有模型缩放量名称。

例如在 server 的 supervision 列表中增加：

```json
{"name":"assay","path":["row","measurements"],"mask":["row","observed"],"shape":[36,5],"readout":"sequence","loss":"mse","weight":0.2}
```

语义注释监督可声明编码角色。下面的源字段须为 metadata 接口接受的标准字段列表；目标宽度须与服务模型实际输出一致：

```json
{"name":"annotation","path":["patient","annotation_fields"],"shape":[5120],"readout":"pooled","loss":"mse","weight":0.1,"encode":{"role":"metadata/encode","input":"fields","output":"embedding"}}
```

5120 仅为示例，客户端不固定该宽度。编码服务输出的测试标记会写入训练 checkpoint。监督 schema 随 model_config 保存；恢复或阶段切换时与 server 核对，不一致则要求重新训练。旧的无 schema checkpoint 仍可加载推理或对齐，但不能直接继续新的监督训练。

## 检索

`retrieval.generation` 和 `retrieval.safety` 各声明 `strategy`、`paths` 和 `limit`。策略为 `vector`、`exact` 或 `substring`；向量策略另指定 `dimension`。路径从 `context`、`requirement`、`constraints` 取值，安全阶段还可读取 `candidate` 和 `simulation`。

向量查询会序列化所选字段，并实际调用 embed 服务；精确或子串策略把所选字段的标量值作为检索词。客户端不预先选择疾病、食材或其他业务字段。检索 schema 和嵌入服务 provenance 写入推理结果，随机嵌入会使结果标记为 test_only。知识库为空或选定字段均缺失时返回空知识。

验证：在模块目录运行 `python -m unittest test_runtime_schema test_numeric test_metadata test_action_encoder test_alignment`，服务回归在 extra_code 目录运行 `python -m unittest test_server test_pipeline`。
