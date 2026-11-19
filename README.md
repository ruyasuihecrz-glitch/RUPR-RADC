# RUPR-RADC

证据重排与冻结Reader决策校准。

**[下载权重](https://huggingface.co/DavideChen/RUPR-RADC) · [验证接口](docs/DRIFTBENCH_INTERFACE.md) · [固定版本](https://github.com/ruyasuihecrz-glitch/RUPR-RADC/releases/tag/v1.0.0)**

## 快速开始

```bash
git clone https://github.com/ruyasuihecrz-glitch/RUPR-RADC.git
cd RUPR-RADC
pip install -e .
pip install huggingface_hub
python scripts/download_weights.py
```

## 运行验证

```bash
rupr-radc predict-features --features frozen_features.npz \
  --output outputs/predictions.jsonl --device cuda:0
```

新闻图文直接推理、Python调用与独立评分，见[DriftBench验证接口](docs/DRIFTBENCH_INTERFACE.md)。基础Reader、reranker、CLIP与评测输入需另行准备。代码提供推理、验证入口、目标函数与配方；完整数据挖掘和训练调度CLI不在此版本中。

[Apache 2.0](LICENSE) · [方法与配置](docs/METHOD.md)
