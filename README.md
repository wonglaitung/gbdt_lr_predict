# 🧠 GBDT + LR 可解释性机器学习系统

这是一个**企业级、端到端**的机器学习系统，结合梯度提升决策树（GBDT）和逻辑回归（LR），提供 **Flask API 服务** 和 **本地批量预测脚本**。核心目标：**高精度预测 + 双维度可解释性**（SHAP 特征贡献 + 人类可读决策规则），让业务、风控、审核人员真正理解“为什么”。

> ✅ **逻辑完全一致**：API 服务与本地脚本共用同一套预测引擎  
> ✅ **配置驱动**：改 `config/features.csv` 5 分钟适配新业务  
> ✅ **双解释输出**：`top_3_features`（量化贡献） + `top_3_rules`（决策路径）  
> ✅ **业务友好**：批量预测结果保留原始 ID，概率列置顶，Excel 直接打开  
> ✅ **零重复代码**：核心逻辑抽取至 `predictor.py`，维护成本降低 90%

---

## 📁 项目结构

```
gbdt_lr_system/
├── assets/
│   └── 1594867406872.png     # SHAP 可解释性示意图
├── config/
│   └── features.csv          # ✅ 字段定义（业务人员只需改这里！）
├── data/
│   ├── data.csv              # 原始数据（用于训练）
│   ├── train.csv             # 训练数据
│   └── test.csv              # 测试数据
├── output/                   # 模型目录（训练生成，API/本地脚本依赖）
│   ├── actual_n_estimators.csv # 🆕 树数量校验
│   ├── category_features.csv
│   ├── continuous_features.csv
│   ├── gbdt_feature_importance.csv # 🆕 GBDT 特征重要性
│   ├── gbdt_model.pkl        # GBDT 模型
│   ├── lr_leaf_coefficients.csv # 🆕 LR 叶子节点系数
│   ├── lr_model.pkl          # LR 模型
│   ├── shap_summary_plot.png # 🆕 SHAP 汇总图
│   ├── shap_waterfall_sample_0.png # 🆕 SHAP 瀑布图示例
│   ├── submission_gbdt_lr.csv # 🆕 提交文件示例
│   └── train_feature_names.csv
├── app.py                    # Flask API 服务（调用 predictor.py）
├── gbdt_lr.py                # GBDT+LR 核心训练逻辑
├── local_batch_predict.py    # 🚀 本地批量预测（调用 predictor.py，无需启动服务）
├── predictor.py              # 🔑 **核心！共享预测逻辑（API + 本地脚本共用）**
├── README.md
├── requirements.txt
└── train.py                  # 训练脚本（调用 gbdt_lr.py）
```

> 💡 **关键**：`predictor.py` 封装了**所有重复逻辑**（模型加载、预处理、预测、解释生成），确保 API 与本地脚本行为 100% 一致！

---

## 📄 配置文件: `config/features.csv`

**格式**（CSV，两列）：

| feature_name | feature_type |
|--------------|--------------|
| 字段名        | `continuous` 或 `category` |

**示例**:
```csv
feature_name,feature_type
I1,continuous
I2,continuous
C1,category
C2,category
```

> 📌 **业务人员友好**：只需提供字段名和类型，无需接触代码！

---

## ✅ 第一部分：训练脚本 (`train.py`)

**功能**：
1.  读取 `data/` 下数据 + `config/features.csv` 配置
2.  自动处理缺失值（连续填 `-1`，类别填 `"-1"`）
3.  类别特征 → One-Hot 编码
4.  训练 GBDT → 提取叶子索引 → 训练 LR
5.  保存模型到 `output/`
6.  生成可解释性报告（SHAP 图、特征重要性、决策路径）

**技术亮点**：
- **GBDT+LR 架构**：GBDT 捕捉非线性，LR 在叶子上加权 → 高精度 + 强可解释
- **人类可读规则**：自动将 `device_type_iOS` 还原为 `device_type == 'iOS'`

---

## ✅ 第二部分：预测方式（双模式）

### 🚀 模式 1：本地批量预测（推荐！无需服务）

**适用场景**：离线批量预测、数据敏感环境、高性能需求

**使用方式**：
```bash
# 预测单个 CSV 文件（结果保存在同目录）
python local_batch_predict.py data/test.csv

# 输出：data/predicted_test.csv
```

**输出格式**（与 API 完全一致）：
```csv
prediction_probability,Id,I1,I2,...,C1,C2,...,top_3_features,top_3_rules
0.8921,1001,1.0,0,...,"75ac2fe6","1cfdf714",...,I6(+0.312); C1_75ac2fe6(+0.201),I6 > 173.0000; C1 == '75ac2fe6'
```

> ✅ **优势**：  
> - 无网络开销，速度更快  
> - 不依赖 Flask，部署简单  
> - 与 API 结果 100% 一致（共用 `predictor.py`）

---

### 🌐 模式 2：Flask API 服务 (`app.py`)

**适用场景**：在线实时预测、多系统调用、需要 HTTP 接口

**启动服务**：
```bash
python app.py  # 默认 http://localhost:5000
```

#### 🔹 接口 1：单样本预测 `/predict`
- **输入**：JSON 样本
```json
{
    "I1": 1.0,
    "I2": 0,
    "I3": 1.0,
    "I4": 227.0,
    "I5": 1.0,
    "I6": 173.0,
    "I7": 18.0,
    "I8": 50.0,
    "I9": 1.0,
    "I10": 7.0,
    "I11": 1.0,
    "I12": 0,
    "I13": 0,
    "C1": "75ac2fe6",
    "C2": "1cfdf714",
    "C3": "713fbe7c",
    "C4": "aa65a61e",
    "C5": "25c83c98",
    "C6": "3bf701e7",
    "C7": "7195046d",
    "C8": "a73ee510",
    "C9": "9e5006cd",
    "C10": "4d8549da",
    "C11": "a48afad2",
    "C12": "51b97b8f",
    "C13": "b28479f6",
    "C14": "d345b1a0",
    "C15": "3fa658c5",
    "C16": "3486227d",
    "C17": "e88ffc9d",
    "C18": "c393dc22",
    "C19": "b1252a9d",
    "C20": "57c90cd9",
    "C21": "",
    "C22": "bcdee96c",
    "C23": "4d19a3eb",
    "C24": "cb079c2d",
    "C25": "456c12a0",
    "C26": ""
}
```
- **输出**：概率 + SHAP 图（base64） + 规则（JSON）

#### 🔹 接口 2：批量预测 `/predict_batch_csv`
- **输入**：上传 CSV 文件
- **输出**：下载带解释的 CSV（格式同 `local_batch_predict.py`）

> 💡 **关键设计**：  
> - API 与本地脚本**共用 `predict_core` 逻辑**（来自 `predictor.py`）  
> - 批量预测**不生成 SHAP 图**（`generate_plot=False`），提升性能

---

## 🔍 解释字段详解（双维度）

### 📊 `top_3_features` —— **量化贡献**
- **来源**：SHAP 算法
- **格式**：`特征名(SHAP值)`，如 `I6(+0.312)` 或 `C1_75ac2fe6(+0.201)`
- **含义**：  
  - `+`：推高预测概率  
  - `-`：拉低预测概率  
  - 数值越大，影响越强

### 🧭 `top_3_rules` —— **决策路径**
- **来源**：GBDT 树结构解析
- **格式**：人类可读条件，如 `I6 > 173.0000` 或 `C1 == '75ac2fe6'`
- **含义**：模型实际走的判断路径，可直接作为业务规则

### 🆚 对比总结
| 维度         | `top_3_features`               | `top_3_rules`                 |
|--------------|--------------------------------|-------------------------------|
| **本质**     | 特征贡献分数                   | 模型决策路径                  |
| **使用者**   | 数据科学家（模型优化）         | 业务/风控（规则制定）         |
| **示例**     | `I6(+0.312)`                   | `I6 > 173.0000`               |

> ✅ **两者结合 = 完整解释**：既知“谁贡献大”，又知“模型怎么想”

---

## 🚀 使用流程

1. **准备数据**  
   - 放 `data.csv` 到 `data/` 目录（原始数据）
   - 编辑 `config/features.csv` 定义特征

2. **训练模型**  
   ```bash
   python train.py  # 生成 output/ 目录
   ```

3. **选择预测方式**  
   - **离线批量**（推荐）：  
     ```bash
     python local_batch_predict.py data/test.csv
     ```
   - **在线 API**：  
     ```bash
     python app.py
     curl -F "file=@data/test.csv" http://localhost:5000/predict_batch_csv -o result.csv
     ```

---

## 💡 适用场景

- 🏦 **金融风控**：评分卡 + 可解释拒绝原因  
- 🛒 **推荐系统**：CTR 预估 + 特征归因  
- 🏥 **医疗诊断**：辅助决策 + 医生可理解规则  
- 📊 **业务分析**：上传客户名单 → 下载预测 + 解释  

---

## ⚠️ 注意事项

- **依赖安装**：
  ```bash
  pip install flask lightgbm shap matplotlib scikit-learn pandas joblib
  ```
- **特征一致性**：预测时 CSV 列名/类别值必须与训练一致
- **模型依赖**：`output/` 目录必须包含所有 `.pkl` 和 `.csv` 文件
- **SHAP 性能**：本地脚本默认生成解释，如需极致性能可关闭（修改 `predict_core` 调用参数）

---

## 🎉 为什么选择本系统？

> **这不是玩具项目，而是生产就绪的可解释 ML 解决方案！**

- **业务零门槛**：配置驱动 + CSV 输入输出  
- **解释双保险**：SHAP 量化 + 规则白盒  
- **工程高可靠**：  
  - 逻辑解耦（`predictor.py`）  
  - 临时文件自动清理  
  - 中文兼容（`utf-8-sig`）  
  - 错误处理完善  
- **部署灵活**：  
  - 本地脚本（离线）  
  - Flask API（在线）  

---

✅ **现在，你可以自信地将它交给业务团队使用了！**

> 📘 **附：生产部署建议**  
> - API 服务：用 `gunicorn` 替代 Flask 内置服务器  
> - 高频场景：缓存 SHAP 解释结果  
> - 安全加固：添加 API 认证和限流  

--- 

**让每一次预测都有据可依，让机器学习不再是个黑盒！** 🚀
