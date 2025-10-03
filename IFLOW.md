# GBDT + LR 可解释性机器学习系统 - iFlow 上下文

## 项目概述

这是一个**企业级、端到端**的机器学习系统，结合梯度提升决策树（GBDT）和逻辑回归（LR），提供 **Flask API 服务** 和 **本地批量预测脚本**。核心目标：**高精度预测 + 双维度可解释性**（SHAP 特征贡献 + 人类可读决策规则），让业务、风控、审核人员真正理解"为什么"。

- **逻辑完全一致**：API 服务与本地脚本共用同一套预测引擎 (`predictor.py`)
- **配置驱动**：通过 `config/features.csv` 定义特征类型，适配新业务只需修改配置
- **双解释输出**：`top_3_features`（量化贡献） + `top_3_rules`（决策路径）
- **业务友好**：批量预测结果保留原始 ID，概率列置顶，Excel 直接打开
- **增强可解释性**：SHAP 全局特征重要性图 + 单样本瀑布图 + LR 叶子节点系数分析

## 项目结构

```
gbdt_lr_predict/
├── assets/                    # 存放项目相关图片资源
│   └── 1594867406872.png     # SHAP 可解释性示意图
├── config/                    # 配置文件目录
│   └── features.csv          # 字段定义（业务人员只需改这里！）
├── data/                     # 数据目录
│   ├── data.csv              # 原始数据（由 train.py 生成）
│   ├── train.csv             # 训练数据（含 Label 列）
│   ├── test.csv              # 测试数据（无 Label 列）
│   └── predicted_test.csv    # 本地预测结果示例
├── output/                   # 模型输出目录（训练生成，API/本地脚本依赖）
│   ├── actual_n_estimators.csv # 实际训练树数量
│   ├── category_features.csv   # 类别特征列表
│   ├── continuous_features.csv # 连续特征列表
│   ├── gbdt_feature_importance.csv # GBDT 特征重要性
│   ├── gbdt_model.pkl          # GBDT 模型
│   ├── lr_leaf_coefficients.csv # LR 叶子节点系数
│   ├── lr_model.pkl            # LR 模型
│   ├── roc_curve.png           # ROC曲线图
│   ├── shap_summary_plot.png   # SHAP 汇总图
│   ├── shap_waterfall_sample_0.png # SHAP 瀑布图示例
│   ├── submission_gbdt_lr.csv  # 提交文件示例
│   └── train_feature_names.csv # 训练特征名称列表
├── app.py                    # Flask API 服务（调用 predictor.py）
├── local_batch_predict.py    # 本地批量预测（调用 predictor.py，无需启动服务）
├── predictor.py              # 核心！共享预测逻辑（API + 本地脚本共用）
├── README.md
├── requirements.txt
└── train.py                  # 训练脚本（包含 GBDT+LR 核心逻辑）
```

## 核心组件

### 1. 配置文件 (`config/features.csv`)

定义特征类型，业务人员只需修改此文件即可适配新业务。

**格式**（CSV，两列）：

| feature_name | feature_type |
|--------------|--------------|
| 字段名        | `continuous` 或 `category` |

### 2. 训练脚本 (`train.py`)

**功能**：
1. 读取 `data/` 下数据 + `config/features.csv` 配置
2. 自动处理缺失值（连续填 `-1`，类别填 `"-1"`）
3. 类别特征 → One-Hot 编码
4. 训练 GBDT → 提取叶子索引 → 训练 LR
5. 保存模型到 `output/`
6. 生成可解释性报告（SHAP 图、特征重要性、决策路径、LR 叶子系数、ROC曲线）

**技术亮点**：
- **GBDT+LR 架构**：GBDT 捕捉非线性，LR 在叶子上加权 → 高精度 + 强可解释
- **人类可读规则**：自动将 `C1_75ac2fe6` 还原为 `C1 == '75ac2fe6'`
- **增强可解释性**：
  - SHAP 全局特征重要性图 + 单样本瀑布图
  - GBDT 特征重要性分析
  - LR 叶子节点系数分析（哪些决策路径最重要）
  - 决策路径解析（将机器学习模型的决策过程翻译为人类可读规则）
  - ROC曲线可视化
- **动态树数量**：自动检测实际训练的树数量，确保预测时的一致性

### 3. 预测核心 (`predictor.py`)

封装了所有重复逻辑（模型加载、预处理、预测、解释生成），确保 API 与本地脚本行为 100% 一致。

**核心函数**：
- `load_models()`: 加载训练好的模型和元数据
- `predict_core()`: 核心预测函数，支持单样本和批量预测，生成 SHAP 解释

### 4. 预测方式

#### 模式 1：本地批量预测 (`local_batch_predict.py`)

**使用方式**：
```bash
python local_batch_predict.py data/test.csv
```

**输出**：与 API 批量预测格式完全一致的 CSV 文件

#### 模式 2：Flask API 服务 (`app.py`)

**启动服务**：
```bash
python app.py  # 默认 http://localhost:5000
```

**接口**：
- `/predict` (POST): 单样本预测，返回概率 + SHAP 图（base64）+ 规则（JSON）
- `/predict_batch_csv` (POST): 批量预测，上传 CSV 文件，下载带解释的 CSV
- `/health` (GET): 健康检查，返回服务状态和模型信息

## 可解释性

### `top_3_features` —— 量化贡献

- **来源**：SHAP 算法
- **格式**：`特征名(SHAP值)`，如 `I6(+0.312)` 或 `C1_75ac2fe6(+0.201)`
- **含义**：`+` 推高概率，`-` 拉低概率，数值越大影响越强

### `top_3_rules` —— 决策路径

- **来源**：GBDT 树结构解析
- **格式**：人类可读条件，如 `I6 > 173.0000` 或 `C1 == '75ac2fe6'`
- **含义**：模型实际走的判断路径，可直接作为业务规则

## 使用流程

1. **准备数据**：准备 `train.csv` 和 `test.csv` 放到 `data/` 目录，编辑 `config/features.csv`
2. **训练模型**：`python train.py`
3. **选择预测方式**：
   - 离线批量：`python local_batch_predict.py data/test.csv`
   - 在线 API：`python app.py`
4. **查看结果和解释**：预测结果包含概率和双维度解释，`output/` 目录中有可解释性图表

## 构建和运行

**安装依赖**：
```bash
pip install -r requirements.txt
```

**训练模型**：
```bash
python train.py
```

**本地批量预测**：
```bash
python local_batch_predict.py data/test.csv
```

**启动 API 服务**：
```bash
python app.py
```

## 开发约定

- **编码**：系统自动兼容 UTF-8 和 GBK 编码的 CSV 文件
- **错误处理**：完善的错误处理和日志记录
- **配置驱动**：通过 `config/features.csv` 灵活配置特征类型
- **逻辑复用**：核心预测逻辑集中在 `predictor.py`，避免重复代码