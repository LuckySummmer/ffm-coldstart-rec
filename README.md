## 项目结构

```plaintext
ffm-coldstart-rec/
├── data/
│   ├── raw/                # 原始数据
│   └── processed/          # 预处理后数据
├── src/
│   ├── external/           # 第三方库（如libffm）
│   ├── data_preprocessing.py
│   ├── run.py
│   ├── evaluate_tune.py
│   └── evaluate_run.py
├── utils/
│   ├── eval_metrics.py
│   ├── ffm_format_data2.py
│   └── ffm_result_cal2.py
├── results/
│   ├── models/
│   ├── outputs/
│   ├── results_tune/
│   └── results_run/
├── config/
│   ├── params.json