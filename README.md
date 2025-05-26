## 项目结构

```plaintext
ffm-coldstart-rec/
├── data/
│   ├── raw/                # 原始数据
│   └── processed/          # 预处理后数据
├── src/
│   ├── external/           # 第三方库（如libffm）
│   ├── data_preprocessing.py
│   ├── evaluate.py
│   └── run.py              # 项目统一运行入口
├── utils/
│   ├── eval_metrics.py
│   ├── ffm_format_data2.py
├── results/
│   ├── models/
│   ├── outputs/