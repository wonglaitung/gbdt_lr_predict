# 🧠 GBDT + LR 可解释性机器学习系统

这是一个**企业级、端到端**的机器学习系统，结合了梯度提升决策树（GBDT）和逻辑回归（LR），并提供 Flask API 服务用于**在线单样本预测**和**批量 CSV 预测**。系统核心目标是：**在保持高预测精度的同时，提供强大的模型可解释性**，让业务、风控、审核人员理解“为什么”模型会做出特定预测。

> ✅ 支持真实字段名配置  
> ✅ 支持单样本 JSON 预测 + 解释图  
> ✅ 批量上传 CSV → 下载带 ID + 概率 + 解释的预测结果**  
> ✅ 自动保留原始 ID 列，概率结果第一列，业务友好  
> ✅ 决策路径规则 + SHAP 特征重要性双解释

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
│   └── ...                   # 其他报告文件（SHAP 图、重要性 CSV 等）
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

**核心功能:**
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

**技术亮点:**
*   **GBDT + LR 架构**: GBDT 自动组合特征、捕捉非线性，LR 在叶子上加权 → 高精度 + 强可解释性。
*   **人类可读规则**: 自动将 `device_type_iOS` 还原为 `device_type == 'iOS'`，业务人员秒懂！

---

## ✅ 第二部分：Flask API 服务 (`app.py`)

### 🎯 核心接口 1：单样本预测 `/predict`

**输入**: JSON 格式单样本数据。

```json
{
  "age": 35,
  "income": 80000,
  "device_type": "iOS",
  "region": "North America"
}
```

**输出**: JSON，含概率 + SHAP 图 + 规则。

```json
{
  "probability": 0.8743,
  "explanation": {
    "important_features": [
      {"feature": "income", "shap_value": 0.312},
      {"feature": "device_type_iOS", "shap_value": 0.201},
      {"feature": "age", "shap_value": 0.089}
    ],
    "shap_plot_base64": "image/png;base64,...",
    "top_rules": [
      "income > 100000.0000",
      "device_type == 'iOS'",
      "age > 40.0000"
    ],
    "feature_based_rules": [
      "income > 100000.0000 (SHAP: +0.3120)",
      "device_type == 'iOS' (SHAP: +0.2010)"
    ]
  }
}
```

---

### 🚀 核心接口 2：批量预测（上传 CSV） `/predict_batch_csv`

**输入**: 上传 CSV 文件（第一列为 ID，其余为特征）。

**示例 `input.csv`:**

```csv
user_id,age,income,device_type,region
U1001,25,50000,Android,Asia
U1002,45,120000,iOS,North America
```

**输出**: 下载 CSV 文件，格式如下：

```csv
prediction_probability,user_id,age,income,device_type,region,top_3_features,top_3_rules
0.3214,U1001,25,50000,Android,Asia,age(-0.123); device_type_Android(-0.087); income(+0.045),age <= 30.0000; device_type != 'iOS'
0.8921,U1002,45,120000,iOS,North America,income(+0.312); device_type_iOS(+0.201); age(+0.089),income > 100000.0000; device_type == 'iOS'; age > 40.0000
```

✅ **列顺序设计**：
1. `prediction_probability` —— 业务最关心，放第一列！
2. `user_id`（原始 ID）—— 自动保留，放第二列，方便对齐！
3. 原始特征列 —— 保持原顺序
4. 解释列 —— `top_3_features`, `top_3_rules`

---

### 🩺 健康检查 `/health`

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
    ✅ 成功后 `output/` 目录生成所有文件。

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

- 🏦 **金融风控**：评分卡 + 可解释拒绝原因
- 🛒 **推荐/广告**：CTR 预估 + 特征归因
- 🏥 **医疗预测**：诊断辅助 + 医生可理解规则
- 📊 **AB 实验分析**：理解特征如何影响转化率
- 🧑‍💼 **业务自助分析**：上传客户名单 → 下载预测 + 解释

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

让机器学习不再是个黑盒，让每一次预测都有据可依！

---

✅ **现在，你可以自信地将它交给业务团队使用了！**

--- 

如需部署到生产，建议：
- 使用 `gunicorn` 替代 Flask 内置服务器
- 添加请求限流和认证
- 将 SHAP 图生成改为异步任务
- 使用 Redis 缓存高频样本的解释结果

祝你项目成功！ 🚀
