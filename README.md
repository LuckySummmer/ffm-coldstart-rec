# FFM-ColdStart-Rec

基于 Field-aware Factorization Machine (FFM) 的冷启动用户推荐系统。本项目使用 Flickr 等真实数据集，比较传统 CTR 模型（如 LR、FM、FFM）在冷启动推荐场景下的表现，使用 Precision、Recall、NDCG、Hit Rate 等排序指标进行评估。

---

## 项目结构

```plaintext
ffm-coldstart-rec/
├── data/
│   ├── raw/                # 原始数据（未加入版本控制）
│   └── processed/          # 预处理后的数据（未加入版本控制）
├── src/
│   ├── external/           # 第三方库（如 libffm 的源码）
│   ├── data_preprocessing.py
│   ├── run.py              # 主运行入口
│   ├── evaluate_tune.py    # 调参阶段评估
│   └── evaluate_run.py     # 冷启动阶段评估
├── utils/
│   ├── eval_metrics.py     # 推荐评估指标实现
│   ├── ffm_format_data2.py # 数据转换成 .ffm 格式
│   └── ffm_result_cal2.py  # 模型结果计算工具
├── results/
│   ├── models/
│   ├── outputs/
│   ├── results_tune/
│   └── results_run/
├── config/
│   └── params_flickr.txt   # 模型超参数配置文件
└── README.md
```

## 数据预处理
```plaintext
python -m src.data_preprocessing -d flickr -t
```

## 运行
```plaintext
python -m src.run -d flickr -t
```
