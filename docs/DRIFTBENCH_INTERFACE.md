# DriftBench验证接口

本仓库预留通用Python与JSONL接口，供DriftBench评测程序接入。不假定作者存在某个未公开的提交API；获取作者实际字段规范后只需转换字段，不改模型或阈值。

## 1. 从新闻图文及材料运行

每行一条输入，字段如下。示例为虚构新闻，不含benchmark样本。

```json
{"case_id":"demo-001","news":"A demonstration event took place in Example City.","image_path":"/path/to/news.jpg","sources":[{"source_id":"e1","text":"A supplied report about the event."}],"reference_images":[{"image_path":"/path/to/reference.jpg"}]}
```

`reference_images`可为空。它们只进入CLIP/RADC，不提供给Reader。`sources`是原始合法检索记录列表，RUPR重新排序且保留全部记录；来源记录原有字段如需保留，应与原评测输入一致，不能事后改变Reader载荷。模型输入不包含标签、条件分组或正确与否。图片路径可以是本地绝对路径；仓库不代为联网搜索材料。

```bash
rupr-radc predict --input inputs.jsonl --output outputs/predictions.jsonl \
  --reader-path /models/Qwen3.5-9B \
  --reranker-path /models/Qwen3-VL-Reranker-8B \
  --reranker-helper /models/Qwen3-VL-Reranker-8B/scripts/qwen3_vl_reranker.py \
  --clip-path /models/ViT-L-14.pt
```

输出包含`case_id`、`label`、连续`score`、`raw_score`和`evidence_order`。每题完成后写盘；遇到失败明确报错，不跳题、不截断材料、不自动换文字模型。完整模型需显存充足的CUDA环境；设置`CUDA_VISIBLE_DEVICES`控制可用卡。

## 2. 已有冻结特征的批量验证

```bash
rupr-radc predict-features --features frozen_features.npz \
  --output outputs/predictions.jsonl --device cuda:0
```

NPZ仅包含以下数组，无标签：

|字段|形状/含义|
|---|---|
|case_id|N个唯一字符串|
|base_score|N，原Reader真实减虚假对数概率差，float64|
|answer|N×4096，最后回答位置隐藏状态|
|nodes|N×L×4096，图片/新闻/逐条有序证据的片段平均隐藏状态|
|node_valid|N×L，bool有效掩码|
|node_roles|N×L，int64；图片0、新闻1、材料2|
|image,text,reference_image,reference_text|各N×768，CLIP单位向量|
|clip_present|N×4，bool；对应前面四类CLIP向量|
|signals|N×6，float32；原Reader分数、图文余弦、新闻图与最佳参考图余弦、新闻与最佳/平均材料余弦、材料数/12|

旧最高模型的CLIP参考文字取最高相似材料；参考图片取相似度最高的至多四张的单位平均。缺失项向量置零并通过掩码排除。片段与实际输入token的绑定见`core/evidence_interaction.py`。不能随意换特征池化或CLIP预处理后声称复现原recipe。

## 3. 外部评测器的Python钩子

```python
from rupr_radc import Calibrator, DriftBenchAdapter

def feature_provider(records):
    # 接评测器自己的合法输入转换与冻结特征提取。
    return case_ids, feature_arrays

adapter = DriftBenchAdapter(Calibrator(device="cuda:0"), feature_provider)
predictions = adapter.predict(records_without_labels)
```

`Calibrator.predict_features`也可直接调用。标签只交给评分器：

```bash
rupr-radc score --predictions outputs/predictions.jsonl \
  --labels labels.jsonl --output outputs/metrics.json
```

标签每行`{"case_id":"...","label":"real"}`或`fake`。ID必须全部匹配且唯一。预测与评分分离，零分判fake；统计Macro-F1时固定包含两个类别。
