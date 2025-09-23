# GBDT + LR 可解释性机器学习系统

这是一个完整的机器学习系统，结合了梯度提升决策树（GBDT）和逻辑回归（LR） ，并提供 Flask API 服务用于在线预测和结果解释。该架构旨在提高预测精度的同时，提供强大的模型可解释性 ，让业务方理解“为什么”模型会做出特定预测 。

## 📁 项目结构

```
gbdt_lr_predict/
├── config/
│   └── features.csv          # 字段定义配置文件（核心）
├── data/
│   ├── train.csv             # 训练数据（使用 config/features.csv 中的真实字段名）
│   └── test.csv              # 测试数据
├── output/                   # 自动生成目录，API 服务依赖此目录
│   ├── gbdt_model.pkl        # 训练好的 LightGBM 模型
│   ├── lr_model.pkl          # 训练好的逻辑回归模型
│   ├── continuous_features.csv # 连续特征列表
│   ├── category_features.csv  # 类别特征列表
│   ├── train_feature_names.csv # 训练时所有特征名（含 One-Hot 后）
│   └── ...                   # 其他报告文件（如 SHAP 图、重要性 CSV 等）
├── train.py                  # 训练脚本（从 config/features.csv 读取字段）
└── app.py                    # Flask API 服务（自动加载 output/ 下的配置）
```

## 📄 配置文件: `config/features.csv`

你只需修改此文件即可适配不同业务场景。文件格式为 CSV，包含两列：`feature_name` 和 `feature_type`。

**示例内容:**
```csv
feature_name,feature_type
age,continuous
income,continuous
click_rate,continuous
user_level,category
device_type,category
region,category
```

## ✅ 第一部分：训练脚本 (`train.py`)

**核心功能:**
1.  从 `data/train.csv` 和 `data/test.csv` 读取数据。
2.  根据 `config/features.csv` 识别连续特征和类别特征。
3.  填充缺失值（连续特征填 -1）。
4.  对类别特征进行 One-Hot 编码。
5.  使用 LightGBM 训练 GBDT 分类器 。
6.  用训练好的 GBDT 预测训练集和测试集样本落在每棵树的哪个叶子节点。
7.  对叶子节点索引进行 One-Hot 编码，生成新的特征向量 。
8.  使用这些新的特征向量训练逻辑回归（LR）模型 。
9.  评估 LR 模型性能（如 LogLoss）。
10. 保存模型 (`gbdt_model.pkl`, `lr_model.pkl`) 和元数据 (`continuous_features.csv`, `category_features.csv`, `train_feature_names.csv`) 到 `output/` 目录。
11. 生成可解释性报告：
    *   GBDT 特征重要性 (CSV)
    *   LR 叶子节点系数 (CSV)
    *   SHAP 全局特征重要性图
    *   SHAP 单样本瀑布图示例
    *   控制台输出高权重叶子节点的原始决策路径

**技术亮点:**
*   **GBDT + LR 架构**: GBDT 自动进行特征组合和非线性关系捕捉 ，LR 在叶子节点上进行线性加权，兼具预测能力和可解释性 。
*   **增强可解释性**: 不仅输出预测概率，还输出驱动预测的特征（SHAP）和决策路径规则。

## ✅ 第二部分：Flask API 服务 (`app.py`)

**核心功能:**
接收单个样本的 JSON 输入，返回预测概率及详细的可解释性信息。

**`/predict` 接口:**
*   **输入**: JSON 格式单样本数据 (示例见下方)。
*   **处理流程**:
    1.  加载 `output/` 目录下的模型和配置。
    2.  预处理输入样本 (`preprocess_single_sample`):
        *   填充缺失连续特征为 -1。
        *   对类别特征做 One-Hot 编码。
        *   确保特征顺序和维度与训练时一致。
    3.  使用 GBDT 预测样本在每棵树上的叶子节点索引。
    4.  对叶子索引做 One-Hot 编码。
    5.  将 One-Hot 后的特征输入 LR，得到预测概率。
    6.  使用 `shap.TreeExplainer` 计算 SHAP 值，找出最重要特征。
    7.  使用 `get_leaf_path_enhanced` 从 GBDT 树中提取与重要特征相关的、人类可读的决策路径规则 (如 `I5 <= 3.1416`, `C2 == '男'`)。
    8.  生成 SHAP 瀑布图并转换为 Base64 编码。
*   **输出**: JSON 响应，包含:
    *   `probability`: 预测概率 (0~1)。
    *   `explanation`:
        *   `important_features`: 基于 SHAP 值排序的前 N 个重要特征及其 SHAP 值。
        *   `shap_plot_base64`: SHAP 瀑布图的 Base64 编码 (PNG)。
        *   `top_rules`: 前几棵树的决策路径规则（去重）。
        *   `feature_based_rules`: 与重要特征相关的决策规则，附带 SHAP 值。

**`/health` 接口:**
*   健康检查，确认模型加载成功，并显示特征数量。

**技术亮点:**
*   **规则解析**: `get_leaf_path_enhanced` 能解析 LightGBM 模型中指定树和叶子节点的完整决策路径，并支持将 One-Hot 特征还原为原始类别比较 (如 `C1_abc` → `C1 == 'abc'`)，极大提升业务可读性。
*   **SHAP 集成**: 提供全局和局部（单样本）解释 。

## 🚀 使用流程

1.  **准备数据**: 将训练和测试数据放入 `data/` 目录，并确保列名与 `config/features.csv` 中的 `feature_name` 一致。
2.  **运行训练**: 执行 `python train.py`。成功后会在 `output/` 目录生成所有必需文件和报告。
3.  **启动 API**: 执行 `python app.py` 启动 Flask 服务（默认端口 5000）。
4.  **调用预测**:
    ```bash
    curl -X POST http://localhost:5000/predict \
         -H "Content-Type: application/json" \
         -d '{
               "I1": 1.0, "I2": 0, "I3": 1.0, "I4": 227.0, "I5": 1.0, "I6": 173.0,
               "I7": 18.0, "I8": 50.0, "I9": 1.0, "I10": 7.0, "I11": 1.0, "I12": 0,
               "I13": 0, "C1": "75ac2fe6", "C2": "1cfdf714", "C3": "713fbe7c",
               "C4": "aa65a61e", "C5": "25c83c98", "C6": "3bf701e7", "C7": "7195046d",
               "C8": "a73ee510", "C9": "9e5006cd", "C10": "4d8549da", "C11": "a48afad2",
               "C12": "51b97b8f", "C13": "b28479f6", "C14": "d345b1a0", "C15": "3fa658c5",
               "C16": "3486227d", "C17": "e88ffc9d", "C18": "c393dc22", "C19": "b1252a9d",
               "C20": "57c90cd9", "C21": "", "C22": "bcdee96c", "C23": "4d19a3eb",
               "C24": "cb079c2d", "C25": "456c12a0", "C26": ""
             }'
    ```

## 🖼️ API 响应示例

```json
{
  "probability": 0.8743,
  "explanation": {
    "important_features": [
      {"feature": "I5", "shap_value": 0.2134},
      {"feature": "C2_男", "shap_value": 0.1892},
      {"feature": "I7", "shap_value": -0.1567}
    ],
    "shap_plot_base64": "image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQ...",
    "top_rules": [
      "I5 <= 3.1416",
      "C2 == '男'",
      "I7 > 100.0000"
    ],
    "feature_based_rules": [
      "I5 <= 3.1416 (SHAP: +0.2134)",
      "C2 == '男' (SHAP: +0.1892)",
      "I7 > 100.0000 (SHAP: -0.1567)"
    ]
  }
}
```

## 💡 适用场景

*   风控评分卡（需要概率 + 可解释规则）
*   推荐系统排序（经典 GBDT+LR 架构 ）
*   医疗/金融预测（监管要求模型可解释 ）
*   AB 实验分析（理解特征如何影响预测）

## ⚠️ 注意事项

*   **依赖**: 需安装 `flask`, `lightgbm`, `shap`, `matplotlib`, `joblib`, `pandas`, `numpy`, `scikit-learn`。
*   **SHAP 性能**: SHAP 图生成可能较慢，生产环境可考虑异步处理或缓存。
*   **特征一致性**: 输入 API 的类别特征值必须与训练时出现的值一致，否则 One-Hot 编码会出错。
*   **模型兼容性**: 树结构解析依赖 LightGBM 的 `dump_model()`，需注意版本兼容性。
*   **文件依赖**: API 启动前必须存在 `output/` 目录下的所有 `.pkl` 和 `.csv` 文件，否则会报错。
*   **开发环境**: Python 3.10+。

## ✅ 总结

本系统提供了一套从离线训练到在线预测与解释的完整解决方案。它利用 GBDT+LR 架构平衡了预测性能和可解释性 ，通过 SHAP 值和人类可读的决策规则，让模型决策过程透明化，极大提升模型在业务场景（如金融、医疗、广告）中的信任度和合规性。
