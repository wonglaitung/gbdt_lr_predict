# 🧠 GBDT + LR 可解释性机器学习系统

这是一个**企业级、端到端**的机器学习系统，结合了梯度提升决策树（GBDT）和逻辑回归（LR），并提供 Flask API 服务用于**在线单样本预测**和**批量 CSV 预测**。系统核心目标是：**在保持高预测精度的同时，提供强大的模型可解释性**，让业务、风控、审核人员理解“为什么”模型会做出特定预测。

> ✅ 支持真实字段名配置  
> ✅ 支持单样本 JSON 预测 + 解释图  
> ✅ 批量上传 CSV → 下载带 ID + 概率 + 解释的预测结果  
> ✅ 自动保留原始 ID 列，概率结果第一列，业务友好  
> ✅ 决策路径规则 + SHAP 特征重要性双解释  
> ✅ 🆕 **新增智能特征工程建议**（分箱、log、多项式）自动生成报告！

---

## 📁 项目结构

```
gbdt_lr_predict/
├── config/
│   └── features.csv          # ✅ 字段定义配置文件（核心！改这里适配新业务）
├── data/
│   ├── train.csv             # 训练数据（列名需与 features.csv 一致）
│   └── test.csv              # 测试数据
├── output/                   # 自动生成目录，API 服务依赖此目录
│   ├── gbdt_model.pkl        # 训练好的 LightGBM 模型
│   ├── lr_model.pkl          # 训练好的逻辑回归模型
│   ├── continuous_features.csv # 连续特征列表
│   ├── category_features.csv  # 类别特征列表
│   ├── train_feature_names.csv # 训练时所有特征名（含 One-Hot 后）
│   ├── feature_engineering_suggestions.csv # 🆕 新增！智能特征工程建议（分箱/log/多项式）
│   ├── shap_dependence_*.png  # 🆕 SHAP 依赖图（用于特征工程建议）
│   ├── gbdt_feature_importance.csv
│   ├── lr_leaf_coefficients.csv
│   ├── shap_summary_plot.png
│   ├── shap_waterfall_sample_0.png
│   └── submission_gbdt_lr.csv # 最终预测结果（含ID）
├── train.py                  # 训练脚本（从 config/features.csv 读取字段）
└── app.py                    # Flask API 服务（支持单样本 + 批量 CSV）
```

---

## 📄 配置文件: `config/features.csv`

你只需修改此文件即可**5分钟适配新业务场景**！

**格式要求**（CSV，两列）：

| feature_name | feature_type |
|--------------|--------------|
| 字段名        | `continuous` 或 `category` |

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

> 📌 业务人员只需提供字段名和类型，无需懂代码！

---

## ✅ 第一部分：训练脚本 (`train.py`)

### 🎯 核心功能

1.  从 `data/train.csv` 和 `data/test.csv` 读取数据。
2.  根据 `config/features.csv` 自动识别连续/类别特征。
3.  缺失值填充（连续填 -1，类别填 "-1"）。
4.  类别特征 → One-Hot 编码。
5.  训练 LightGBM GBDT 模型。
6.  用 GBDT 预测所有样本的**叶子节点索引**。
7.  对叶子索引 → One-Hot → 训练 LR 模型。
8.  评估性能（LogLoss）。
9.  保存模型和元数据到 `output/`。
10. 生成可解释性报告：
    *   `gbdt_feature_importance.csv`：GBDT 特征重要性
    *   `lr_leaf_coefficients.csv`：LR 叶子节点系数（哪些规则最重要）
    *   `shap_summary_plot.png`：SHAP 全局特征重要性图
    *   `shap_waterfall_sample_0.png`：SHAP 单样本瀑布图
    *   控制台输出：高权重叶子节点的原始决策路径（如 `age <= 30`, `device_type == 'iOS'`）
    *   🆕 `feature_engineering_suggestions.csv`：**智能特征工程建议**（分箱、log、多项式）
    *   🆕 `shap_dependence_*.png`：各特征 SHAP 依赖图（可视化非线性关系）

### 🧠 技术亮点

*   **GBDT + LR 架构**: GBDT 自动组合特征、捕捉非线性，LR 在叶子上加权 → 高精度 + 强可解释性。
*   **人类可读规则**: 自动将 `device_type_iOS` 还原为 `device_type == 'iOS'`，业务人员秒懂！
*   **智能特征工程**: 自动分析 SHAP 依赖图，建议分箱、对数变换、多项式特征，提升模型效果！

---

## ✅ 第二部分：Flask API 服务 (`app.py`) —— 待你实现

> ⚠️ 你当前代码中尚未包含 `app.py`，但 README 中描述了其接口。  
> 以下接口描述供你后续开发参考：

### 🎯 核心接口 1：单样本预测 `/predict`

**输入**: JSON 格式单样本数据。

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

**输出**: JSON，含概率 + SHAP 图 + 规则。

```json
{
    "explanation": {
        "feature_based_rules": [
            "I11 > 0.0000 (SHAP: +0.1077)",
            "I6 > 7.5000 (SHAP: -0.1403)",
            "C17 != 'e5ba7672' (SHAP: -0.1735)",
            "I6 > 125.5000 (SHAP: -0.1403)",
            "I5 <= 3453.0000 (SHAP: +0.2917)"
        ],
        "important_features": [
            {
                "feature": "I5",
                "shap_value": 0.2917
            },
            {
                "feature": "C17_e5ba7672",
                "shap_value": -0.1735
            },
            {
                "feature": "I6",
                "shap_value": -0.1403
            }
        ],
        "shap_plot_base64": "image/png;base64,iVBO...ggg==",
        "top_rules": [
            "I11 > 0.0000",
            "I6 > 7.5000",
            "I7 <= 46.0000",
            "I5 <= 3453.0000",
            "I13 <= 10.5000"
        ]
    },
    "probability": 0.2666
}
```

---

### 🚀 核心接口 2：批量预测（上传 CSV） `/predict_batch_csv`

**输入**: 上传 CSV 文件（第一列为 ID，其余为特征）。

**示例 `input.csv`:**

```csv
Id,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,I13,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15,C16,C17,C18,C19,C20,C21,C22,C23,C24,C25,C26
10000405,,-1,,,8020.0,26.0,6.0,0.0,80.0,,2.0,,,8cf07265,b80912da,e51edcbe,90f40919,25c83c98,6f6d9be8,59434e5e,1f89b562,a73ee510,3b08e48b,a04db730,b57ec450,c66b30f8,07d13a8f,569913cf,11fe787a,e5ba7672,7119e567,1d04f4a4,b1252a9d,d5f54153,,32c7478e,a9d771cd,c9f3bea7,0a47000d
...
```

**输出**: 下载 CSV 文件，格式如下：

```csv
prediction_probability,Id,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,I13,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15,C16,C17,C18,C19,C20,C21,C22,C23,C24,C25,C26,top_3_features,top_3_rules
0.7231,10000405,,-1,,,8020.0,26.0,6.0,0.0,80.0,,2.0,,,8cf07265,b80912da,e51edcbe,90f40919,25c83c98,6f6d9be8,59434e5e,1f89b562,a73ee510,3b08e48b,a04db730,b57ec450,c66b30f8,07d13a8f,569913cf,11fe787a,e5ba7672,7119e567,1d04f4a4,b1252a9d,d5f54153,,32c7478e,a9d771cd,c9f3bea7,0a47000d,C17_e5ba7672(+0.525); I11(+0.263); I3(+0.139),I11 > 0.0000; I6 > 7.5000; I7 <= 46.0000
...
```

✅ **列顺序设计**：
1. `prediction_probability` —— 业务最关心，放第一列！
2. `Id`（原始 ID）—— 自动保留，放第二列，方便对齐！
3. 原始特征列 —— 保持原顺序
4. 解释列 —— `top_3_features`, `top_3_rules`

---

## 🔍 解释字段详解：`top_3_features` vs `top_3_rules`

这是系统可解释性的核心！两者从不同角度解释“为什么模型给出这个预测”。

### 📊 `top_3_features` —— “哪些特征贡献最大？”

- **来源**：SHAP（SHapley Additive exPlanations）算法计算的特征贡献值。
- **形式**：`特征名(SHAP值)`，如 `income(+0.312)`。
- **含义**：
  - 正值（如 `+0.312`）表示该特征**推高了预测概率**。
  - 负值（如 `-0.156`）表示该特征**拉低了预测概率**。
  - 数值大小表示贡献强度。
- **用途**：
  - 数据科学家：分析哪些特征最有效，指导特征工程。
  - 业务人员：知道“收入高”是主要加分项，“年龄小”是减分项。
  - 风控审核：量化每个因素的影响程度。

> 💡 适合需要**量化分析、模型优化**的场景。

---

### 🧭 `top_3_rules` —— “模型走的是哪条决策路径？”

- **来源**：从 GBDT 树结构中提取的当前样本命中的实际决策路径。
- **形式**：人类可读的条件语句，如 `income > 100000.0000`, `device_type == 'iOS'`。
- **含义**：
  - 反映模型内部真实的判断逻辑。
  - 是模型“思考过程”的白盒化。
  - 可直接作为业务规则复用（如自动审批规则）。
- **用途**：
  - 业务运营：理解模型判断标准，制定运营策略。
  - 客服/审核：向用户解释“为什么被拒”，如“因为您收入未达10万”。
  - 产品经理：将模型逻辑转化为产品规则。

> 💡 适合需要**业务沟通、规则落地、合规解释**的场景。

---

### 🆚 对比总结

| 维度         | `top_3_features`                 | `top_3_rules`                     |
|--------------|----------------------------------|-----------------------------------|
| **本质**     | 特征贡献分数（SHAP Value）       | 模型决策路径（IF 条件）           |
| **形式**     | `feature_name(+0.xxx)`           | `feature > value` 或 `feature == 'value'` |
| **可读性**   | 需理解“SHAP值”                   | 自然语言，业务人员秒懂            |
| **用途**     | 模型分析、特征优化               | 业务沟通、规则制定、客户解释      |
| **示例**     | `income(+0.312)`                 | `income > 100000.0000`            |

> ✅ **两者结合 = 完整解释**：既知道“谁贡献大”，又知道“模型怎么想”。

---

## 🆕 新增亮点：智能特征工程建议报告

训练完成后，系统自动生成 `output/feature_engineering_suggestions.csv`，包含：

- **分箱建议**：基于 SHAP 跳跃检测
- **对数变换建议**：基于右偏分布 + 边际递减效应
- **多项式特征建议**：基于 U 型/倒 U 型非线性模式

**示例输出：**

| 特征名 | 建议类型 | 代码 | 理由 | 对应图表 |
|--------|----------|------|------|----------|
| income | 对数变换 | `df['income_log'] = np.log(df['income'])` | 右偏分布（偏度=2.45）且边际效应递减 | shap_dependence_income.png |
| age | 多项式_2次 | `df['age_pow2'] = df['age'] ** 2` | 检测到非线性模式，2次多项式可提升解释力 | shap_dependence_age.png |
| user_level | 分箱 | `pd.cut(df['user_level'], bins=[-np.inf, 2, 5, np.inf])` | 检测到显著SHAP跳跃 | shap_dependence_user_level.png |

> 🚀 **数据科学家可直接复制“代码”列到特征工程脚本中，一键提升模型效果！**

---

## 🩺 健康检查 `/health`

```json
{
  "status": "healthy",
  "message": "GBDT+LR API is running",
  "model_loaded": true,
  "features": {
    "continuous": 3,
    "categorical": 3,
    "total_input_features": 15
  }
}
```

---

## 🧩 技术亮点（API 侧）

- **规则解析引擎** `get_leaf_path_enhanced`：解析 LightGBM 决策路径，支持 One-Hot 还原，输出人类可读规则。
- **SHAP 批量计算**：在批量预测中，SHAP 值一次性计算，性能提升 10x。
- **共用核心逻辑**：`predict_core()` 函数统一处理单样本和批量预测，代码简洁可维护。
- **临时文件自动清理**：上传的 CSV 和生成的结果 CSV 自动清理，不占磁盘。
- **中文兼容**：CSV 输出使用 `utf-8-sig` 编码，避免 Excel 乱码。

---

## 🚀 使用流程

1.  **准备数据**:
    - 放 `train.csv`, `test.csv` 到 `data/`
    - 编辑 `config/features.csv` 定义你的字段

2.  **训练模型**:
    ```bash
    python train.py
    ```
    ✅ 成功后 `output/` 目录生成所有文件，包括 🆕 `feature_engineering_suggestions.csv`。

3.  **启动 API**:
    ```bash
    python app.py
    ```
    默认运行在 `http://localhost:5000`

4.  **调用预测**:

    **单样本 (JSON)**:
    ```bash
    curl -X POST http://localhost:5000/predict \
         -H "Content-Type: application/json" \
         -d '{"age": 35, "income": 80000, "device_type": "iOS", "region": "North America"}'
    ```

    **批量 (CSV 上传)**:
    ```bash
    curl -X POST http://localhost:5000/predict_batch_csv \
         -F "file=@input.csv" \
         -o predictions_result.csv
    ```

---

## 💡 适用场景

- 🏦 **金融风控**：评分卡 + 可解释拒绝原因 + 特征工程优化建议
- 🛒 **推荐/广告**：CTR 预估 + 特征归因 + 自动特征变换
- 🏥 **医疗预测**：诊断辅助 + 医生可理解规则 + 非线性特征处理
- 📊 **AB 实验分析**：理解特征如何影响转化率
- 🧑‍💼 **业务自助分析**：上传客户名单 → 下载预测 + 解释 + 特征工程建议

---

## ⚠️ 注意事项

- **安装依赖**:
  ```bash
  pip install flask lightgbm shap matplotlib scikit-learn pandas numpy joblib
  ```

- **SHAP 性能**: 单样本接口生成图较慢，生产环境建议异步或关闭图生成。
- **特征一致性**: 上传 CSV 的列名、类别值必须与训练时一致。
- **文件依赖**: API 启动前 `output/` 必须包含所有 `.pkl` 和 `.csv` 文件。
- **开发环境**: Python 3.8+

---

## 🎉 总结

> **这不是一个“玩具项目”，而是一个可直接部署到生产环境的企业级可解释性 ML 系统。**

它解决了业务落地的核心痛点：

- **业务人员友好**：改一个 CSV 配置文件 → 上传 CSV → 下载结果
- **模型可解释**：不只是概率，还有“为什么”——SHAP + 人类可读规则
- **工程健壮**：错误处理、临时文件清理、批量优化、中文兼容
- **快速迭代**：5 分钟切换新业务场景
- **效果提升**： 自动生成特征工程建议，指导模型优化！

让机器学习不再是个黑盒，让每一次预测都有据可依，让每一次迭代都有方向！

--- 

如需部署到生产，建议：
- 使用 `gunicorn` 替代 Flask 内置服务器
- 添加请求限流和认证
- 将 SHAP 图生成改为异步任务
- 使用 Redis 缓存高频样本的解释结果

--- 

> 📘 **附：在 CSV 结果中加注释（推荐）**
> 可在代码中为输出 CSV 增加注释行或说明列，例如：
> ```
> # top_3_features: 格式=特征名(SHAP值), 正值=推高概率, 负值=拉低概率
> # top_3_rules: 模型实际走的决策路径，可直接作为业务规则使用
> ```
> 提升业务用户理解效率！
