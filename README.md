这段代码实际上由两个独立但高度相关的 Python 脚本组成，共同实现了一个 GBDT + LR（梯度提升决策树 + 逻辑回归）的可解释性机器学习系统，并配套提供了一个 Flask API 服务用于在线预测和解释。

🧩 整体架构概览
训练脚本（下半部分）
读取数据 → 特征工程 → 训练 GBDT → 提取叶子节点 → One-Hot 编码 → 训练 LR → 保存模型和元数据 → 生成可解释性报告（SHAP、规则路径、系数）
API 服务（上半部分）
加载模型 → 接收单样本预测请求 → 预处理 → GBDT 叶子预测 → LR 概率预测 → 生成 SHAP 解释 + 决策路径规则 → 返回 JSON 响应（含图表 Base64）

✅ 第一部分：Flask API 服务（用于在线预测与解释）
🎯 核心功能：
接收一个 JSON 格式的单样本输入，返回：

预测概率（0~1）
重要特征（基于 SHAP 值排序前3）
SHAP 瀑布图（Base64 编码 PNG，前端可直接展示）
原始路径规则（前3棵树的决策路径，去重后取前5条）
特征关联规则（与前5个重要特征相关的决策规则，附带 SHAP 值）
🧠 技术亮点：
1. get_leaf_path_enhanced
解析 LightGBM 模型中指定树和叶子节点的完整决策路径。
支持类别特征的 One-Hot 还原（如 C1_abc → C1 == 'abc'）。
输出人类可读的规则，如：
I5 <= 3.1416
C2 == '男'
I7 > 100.0000
2. preprocess_single_sample
将单样本字典预处理成与训练时一致的特征向量。
自动填充缺失连续特征为 -1。
对类别特征做 One-Hot，并补齐训练时存在的所有虚拟列。
最终对齐 train_feature_names 顺序和维度。
3. /predict 接口流程：
GBDT 预测 → 得到每棵树的叶子索引。
对叶子索引做 One-Hot → 输入给 LR。
LR 输出概率。
用 shap.TreeExplainer 计算 SHAP 值 → 找出最重要特征。
从 GBDT 树中提取与这些特征相关的决策规则。
生成 SHAP 瀑布图 → 转为 Base64。
返回结构化 JSON。
4. /health 接口：
健康检查，确认模型加载成功，显示特征数量。
✅ 第二部分：训练脚本（用于离线训练模型）
🎯 核心功能：
读取 train.csv 和 test.csv。
合并数据，填充缺失值为 -1。
对类别特征（C1~C26）做 One-Hot。
训练 LightGBM 分类器。
用 GBDT 预测训练集和测试集的叶子节点索引。
对叶子索引再做 One-Hot → 输入给逻辑回归。
训练 LR 并评估 LogLoss。
保存模型 + 特征名 + 类别/连续特征列表 → 供 API 使用。
生成丰富的可解释性报告：
GBDT 特征重要性 CSV
LR 叶子节点系数 CSV
SHAP 全局重要性图
SHAP 单样本瀑布图
控制台输出高权重叶子节点的原始决策路径
🧠 技术亮点：
1. GBDT + LR 架构
GBDT 自动组合特征、处理非线性。
LR 在叶子节点上做线性加权 → 更好控制、可解释、适合在线系统。
2. 增强可解释性
不仅输出概率，还输出“为什么”：
哪些特征影响最大（SHAP）。
模型走的是哪条决策路径（规则）。
哪些叶子节点对 LR 贡献最大（系数）。
3. 叶子节点规则解析
示例：解析 gbdt_leaf_5_22 → 输出该叶子在第5棵树上的完整路径。
结合类别特征还原，极大提升业务可读性。
4. SHAP 集成
全局特征重要性图 → 理解模型整体偏好。
单样本瀑布图 → 理解某次预测的驱动因素。
📁 文件依赖（API 启动前必须存在）
output/
├── gbdt_model.pkl          # LightGBM 模型
├── lr_model.pkl            # 逻辑回归模型
├── train_feature_names.csv # 训练时所有特征名（含One-Hot后）
├── category_features.csv   # 原始类别特征名（如C1, C2...）
└── continuous_features.csv # 原始连续特征名（如I1, I2...）
❗ 如果缺失，API 启动时报错，提示“请先运行训练脚本”。 

🖼️ 输出示例（API 响应）
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
🚀 使用流程
准备数据：data/train.csv, data/test.csv
运行训练脚本 → 生成 output/ 下所有文件 + 可视化报告。
启动 API 服务：
调用预测：

curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"I1": 1.2, "C1": "A", "I5": 3.0, ...}'
💡 适用场景
风控评分卡：需要概率 + 可解释规则。
推荐系统排序：GBDT+LR 经典架构。
医疗/金融预测：监管要求模型可解释。
AB 实验分析：理解特征如何影响预测。
⚠️ 注意事项
需安装依赖：flask, lightgbm, shap, matplotlib, joblib, pandas, numpy, scikit-learn
SHAP 图生成可能较慢，生产环境可异步或缓存。
类别特征必须与训练时一致，否则 One-Hot 对不齐。
树结构解析依赖 LightGBM 的 dump_model()，版本兼容性需注意。
✅ 总结
这是一套完整的“可解释性机器学习系统”：从离线训练 → 模型保存 → 在线预测 → 规则+SHAP双解释 → 图表可视化。 

它不仅预测准，更重要的是——让业务方、风控、审核人员能看懂模型为什么这么判，极大提升模型落地信任度和合规性。

非常适合金融、医疗、广告、推荐等对可解释性有强需求的场景。



CSV 配置文件读取字段定义完整改造后的代码，包含：

✅ 训练脚本：从 config/features.csv 读取真实字段名
✅ API 服务：无需修改，自动加载训练时保存的字段配置
✅ 配置文件示例（config/features.csv）
✅ 项目结构说明
📁 项目结构


gbdt_lr_system/
├── config/
│   └── features.csv          ← 字段定义（核心配置）
├── data/
│   ├── train.csv             ← 你的训练数据（含真实字段名）
│   └── test.csv              ← 你的测试数据
├── output/                   ← 自动生成，API 依赖此目录
│   ├── gbdt_model.pkl
│   ├── lr_model.pkl
│   ├── continuous_features.csv
│   ├── category_features.csv
│   ├── train_feature_names.csv
│   └── ...（其他报告文件）
├── train.py                  ← 改造后的训练脚本
└── app.py                    ← API 服务（无需修改）
📄 1. 配置文件：config/features.csv
feature_name,feature_type
age,continuous
income,continuous
click_rate,continuous
user_level,category
device_type,category
region,category
✅ 你只需修改这个文件，即可适配任何业务场景！


POSTMAN post jason data:

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
