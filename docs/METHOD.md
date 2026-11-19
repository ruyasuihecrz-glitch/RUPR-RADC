# 固定方法与训练配方

RUPR先用训练后的Qwen3-VL-Reranker-8B为新闻图文与各条文本材料打分，按降序排列全部材料。冻结Qwen3.5-9B读入新闻图、新闻文字和该顺序的原材料；只改变排序，不增删证据。真实性分数在相同JSON回答前缀后计算real/fake单token对数概率差。

RADC保留该原分数，组合两条校正通路：Reader回答表示与图文匹配信号的直接读出，以及新闻/图文/材料表示经跨来源注意力交互后的门控关系修正。后者使用128维、两层四头注意力、三个查询池化；查询数量等为固定超参数。Reader、CLIP与上游重排器在校准训练中冻结。

```text
s_final = s_reader + direct_readout(answer_representation, image_text_matching)
                   + gate × signed_relation_correction
```

发布固定版本的计算与学习权重。当前预处理文件保留训练时PCA/标准化参数，属于直接读出的实现；不能将此发布版本描述为训练或推理均无PCA。单题校准不依赖同批其他新闻。

## 两个训练阶段

- RUPR实际目标为绝对Plackett–Luce排列偏好间隔与条件KL，并非仅因历史目录含DPO就称标准DPO。训练来源为VisualNews/NewsCLIPpings新闻及对应CCN材料，3207训练输入对应2823原始新闻、2107修复比较；其余保持原排序分布。外部98题用于排序阶段选择。配方与损失见`configs/locked_v1.json`、`core/native_preserving_dpo.py`、`core/rank_policy.py`。
- 校准初始读出在34852条外部原图/AI重绘输入上拟合并固定；再在加入事实修改后的50926条外部输入上训练关系分支。该阶段不是所有参数端到端共同更新。按来源计权并平衡真假总权重，分类损失加原本正确的保持项和残差幅度正则，系数均.01；训练参数见锁定配置。

新闻与证据、生成版本按原始来源追踪，训练来源与评测隔离核查属于原数据准备协议。原始新闻、网页缓存、benchmark图像和受限制的第三方材料不随代码或权重再分发。

发布权重、模型版本和校验值见`configs/locked_v1.json`；运行方式见验证接口文档。
