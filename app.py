import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import base64
import io
import os
import uuid
import tempfile
import logging

from flask import Flask, request, jsonify, send_file
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import shap

from werkzeug.utils import secure_filename

# ========== 设置日志 ==========
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ========== 全局变量：加载模型和元数据 ==========
MODEL_DIR = 'output'

required_files = [
    'gbdt_model.pkl',
    'lr_model.pkl',
    'train_feature_names.csv',
    'category_features.csv',
    'continuous_features.csv',
    'actual_n_estimators.csv'  # 🆕 新增：确保树数量一致
]

for f in required_files:
    if not os.path.exists(os.path.join(MODEL_DIR, f)):
        raise FileNotFoundError(f"❌ 找不到必需文件: {f}。请先运行训练脚本生成模型。")

gbdt_model = joblib.load(os.path.join(MODEL_DIR, 'gbdt_model.pkl'))
lr_model = joblib.load(os.path.join(MODEL_DIR, 'lr_model.pkl'))

train_feature_names = pd.read_csv(os.path.join(MODEL_DIR, 'train_feature_names.csv'))['feature'].tolist()
category_features = pd.read_csv(os.path.join(MODEL_DIR, 'category_features.csv'))['feature'].tolist()
continuous_features = pd.read_csv(os.path.join(MODEL_DIR, 'continuous_features.csv'))['feature'].tolist()

# 🆕 加载实际树数量
actual_n_estimators = pd.read_csv(os.path.join(MODEL_DIR, 'actual_n_estimators.csv'))['n_estimators'].iloc[0]
print(f"✅ 实际树数量: {actual_n_estimators}")

category_prefixes = [col + "_" for col in category_features]

print("✅ 模型和元数据加载完成")


# ========== 工具函数：解析叶子节点路径 ==========
def get_leaf_path_enhanced(booster, tree_index, leaf_index, feature_names, category_prefixes):
    try:
        model_dump = booster.dump_model()
        if tree_index >= len(model_dump['tree_info']):
            return None
        tree_info = model_dump['tree_info'][tree_index]['tree_structure']
    except Exception as e:
        print(f"解析树结构失败: {e}")
        return None

    node_stack = [(tree_info, [])]

    while node_stack:
        node, current_path = node_stack.pop()

        if 'leaf_index' in node and node['leaf_index'] == leaf_index:
            return current_path

        if 'split_feature' in node:
            feat_idx = node['split_feature']
            if feat_idx >= len(feature_names):
                feat_name = f"Feature_{feat_idx}"
            else:
                feat_name = feature_names[feat_idx]

            threshold = node.get('threshold', 0.0)
            decision_type = node.get('decision_type', '<=')

            is_category = False
            original_col = None
            category_value = None

            for prefix in category_prefixes:
                if feat_name.startswith(prefix):
                    is_category = True
                    original_col = prefix.rstrip('_')
                    category_value = feat_name[len(prefix):]
                    break

            if is_category:
                right_rule = f"{original_col} == '{category_value}'"
                left_rule = f"{original_col} != '{category_value}'"
            else:
                if decision_type == '<=' or decision_type == 'no_greater':
                    right_rule = f"{feat_name} > {threshold:.4f}"
                    left_rule = f"{feat_name} <= {threshold:.4f}"
                else:
                    right_rule = f"{feat_name} {decision_type} {threshold:.4f}"
                    left_rule = f"{feat_name} not {decision_type} {threshold:.4f}"

            if 'right_child' in node:
                node_stack.append((node['right_child'], current_path + [right_rule]))
            if 'left_child' in node:
                node_stack.append((node['left_child'], current_path + [left_rule]))

    return None


# ========== 工具函数：预处理单样本 ==========
def preprocess_single_sample(sample_dict):
    sample_df = pd.DataFrame([sample_dict])

    for col in continuous_features:
        if col not in sample_df.columns:
            sample_df[col] = -1
        else:
            sample_df[col] = pd.to_numeric(sample_df[col], errors='coerce').fillna(-1)

    all_dummies_list = []

    for col in category_features:
        if col in sample_df.columns:
            val = sample_df[col].iloc[0]
            if pd.isna(val) or val == "":
                sample_df[col] = "-1"
            else:
                sample_df[col] = str(val)
        else:
            sample_df[col] = "-1"

        sample_df[col] = sample_df[col].astype('category')
        dummies = pd.get_dummies(sample_df[col], prefix=col)

        missing_cols = [
            train_col for train_col in train_feature_names
            if train_col.startswith(col + "_") and train_col not in dummies.columns
        ]
        if missing_cols:
            missing_df = pd.DataFrame(0, index=dummies.index, columns=missing_cols)
            dummies = pd.concat([dummies, missing_df], axis=1)

        all_dummies_list.append(dummies)

    if all_dummies_list:
        dummies_combined = pd.concat(all_dummies_list, axis=1)
        sample_df = pd.concat([sample_df.drop(columns=category_features), dummies_combined], axis=1)
    else:
        sample_df = sample_df.drop(columns=category_features)

    missing_final_cols = set(train_feature_names) - set(sample_df.columns)
    if missing_final_cols:
        missing_final_df = pd.DataFrame(0, index=sample_df.index, columns=list(missing_final_cols))
        sample_df = pd.concat([sample_df, missing_final_df], axis=1)

    sample_df = sample_df.reindex(columns=train_feature_names, fill_value=0)
    return sample_df


# ========== 核心预测函数（供 /predict 和 /predict_batch_csv 共用） ==========
def predict_core(sample_df_list, return_explanation=True, generate_plot=False):
    """
    对一组样本进行预测（可批量）
    :param sample_df_list: List of DataFrames, each with 1 row (preprocessed)
    :param return_explanation: 是否返回特征/规则解释
    :param generate_plot: 是否生成 SHAP 图（仅第一个样本，批量时不建议）
    :return: List of result dicts
    """
    if not sample_df_list:
        return []

    # 合并为大 DataFrame
    batch_df = pd.concat(sample_df_list, ignore_index=True)

    # Step 1: GBDT 叶子索引 —— 🆕 使用 actual_n_estimators
    leaf_indices_batch = gbdt_model.booster_.predict(batch_df.values, pred_leaf=True)
    n_trees = actual_n_estimators  # ✅ 关键修改：使用训练时保存的实际树数量

    # Step 2: 叶子 One-Hot
    leaf_dummies_list = []
    for i in range(n_trees):
        leaf_col_name = f"gbdt_leaf_{i}"
        leaf_series = pd.Series(leaf_indices_batch[:, i], name=leaf_col_name)
        dummies = pd.get_dummies(leaf_series, prefix=leaf_col_name)
        leaf_dummies_list.append(dummies)

    leaf_dummies_combined = pd.concat(leaf_dummies_list, axis=1) if leaf_dummies_list else pd.DataFrame()

    lr_feature_names = getattr(lr_model, 'feature_names_in_', [f"feature_{i}" for i in range(len(lr_model.coef_[0]))])
    missing_leaf_cols = set(lr_feature_names) - set(leaf_dummies_combined.columns)
    if missing_leaf_cols:
        missing_leaf_df = pd.DataFrame(0, index=leaf_dummies_combined.index, columns=list(missing_leaf_cols))
        leaf_dummies_combined = pd.concat([leaf_dummies_combined, missing_leaf_df], axis=1)

    leaf_dummies_combined = leaf_dummies_combined.reindex(columns=lr_feature_names, fill_value=0)

    # Step 3: LR 概率
    probabilities = lr_model.predict_proba(leaf_dummies_combined)[:, 1]

    results = []

    if not return_explanation:
        for i in range(len(sample_df_list)):
            results.append({
                "probability": round(float(probabilities[i]), 4),
                "explanation": None
            })
        return results

    # Step 4: 批量 SHAP（只算一次）
    explainer = shap.TreeExplainer(gbdt_model.booster_)
    shap_values_batch = explainer.shap_values(batch_df)
    if isinstance(shap_values_batch, list):
        shap_values_batch = shap_values_batch[1]  # 正类

    # 为每个样本生成解释
    for idx in range(len(sample_df_list)):
        shap_vals = shap_values_batch[idx]
        feature_imp = [(train_feature_names[i], float(shap_vals[i])) for i in range(len(shap_vals))]
        feature_imp.sort(key=lambda x: abs(x[1]), reverse=True)

        important_features = [
            {"feature": feat, "shap_value": round(val, 4)}
            for feat, val in feature_imp[:3]
        ]

        top_shap_features = [feat for feat, val in feature_imp[:5]]
        leaf_indices = leaf_indices_batch[idx]

        # 原始路径规则
        path_rules = []
        for tree_idx in range(min(3, len(leaf_indices))):
            leaf_idx = leaf_indices[tree_idx]
            rule = get_leaf_path_enhanced(
                gbdt_model.booster_,
                tree_index=tree_idx,
                leaf_index=leaf_idx,
                feature_names=train_feature_names,
                category_prefixes=category_prefixes
            )
            if rule:
                path_rules.extend(rule[:3])

        seen = set()
        unique_path_rules = []
        for r in path_rules:
            if r not in seen:
                seen.add(r)
                unique_path_rules.append(r)
        top_rules = unique_path_rules[:5]

        # 特征关联规则
        feature_rules = []
        for tree_idx in range(min(10, len(leaf_indices))):
            leaf_idx = leaf_indices[tree_idx]
            rule = get_leaf_path_enhanced(
                gbdt_model.booster_,
                tree_index=tree_idx,
                leaf_index=leaf_idx,
                feature_names=train_feature_names,
                category_prefixes=category_prefixes
            )
            if rule:
                for r in rule:
                    for feat in top_shap_features:
                        if feat in r or (feat.split('_')[0] + " " in r) or (feat.split('_')[0] + " ==" in r):
                            shap_val = next((val for f, val in feature_imp if f == feat), 0)
                            rule_with_shap = f"{r} (SHAP: {shap_val:+.4f})"
                            if rule_with_shap not in feature_rules:
                                feature_rules.append(rule_with_shap)
                            break
                    if len(feature_rules) >= 5:
                        break
            if len(feature_rules) >= 5:
                break

        # 生成图（仅第一个样本，且仅当 generate_plot=True）
        shap_plot_b64 = ""
        if generate_plot and idx == 0:
            try:
                plt.figure(figsize=(10, 6))
                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_vals,
                        base_values=explainer.expected_value,
                        data=batch_df.iloc[idx],
                        feature_names=batch_df.columns.tolist()
                    ),
                    show=False
                )
                plt.title("SHAP Explanation for Prediction", fontsize=14)
                plt.tight_layout()

                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                plt.close()
                buf.seek(0)
                shap_plot_b64 = "image/png;base64," + base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
            except Exception as e:
                print("⚠️ SHAP plot generation failed:", e)

        explanation = {
            "important_features": important_features,
            "shap_plot_base64": shap_plot_b64,
            "top_rules": top_rules,
            "feature_based_rules": feature_rules[:5]
        }

        results.append({
            "probability": round(float(probabilities[idx]), 4),
            "explanation": explanation
        })

    return results


# ========== API 路由：单样本预测 ==========
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        if not isinstance(data, dict):
            return jsonify({"error": "JSON body must be an object"}), 400

        sample_df = preprocess_single_sample(data)
        results = predict_core([sample_df], return_explanation=True, generate_plot=True)

        if not results:
            return jsonify({"error": "Prediction failed"}), 500

        return jsonify(results[0])  # 单样本，取第一个

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ========== API 路由：批量预测（上传 CSV）==========
@app.route('/predict_batch_csv', methods=['POST'])
def predict_batch_csv():
    """
    批量预测接口：接收上传的 CSV 文件，返回带预测结果的 CSV
    """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in request"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if not file.filename.endswith('.csv'):
            return jsonify({"error": "Only CSV files are allowed"}), 400

        # 创建临时文件保存上传内容
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        # 读取上传的 CSV
        input_df = pd.read_csv(tmp_path)
        os.unlink(tmp_path)  # 删除临时文件

        if input_df.empty:
            return jsonify({"error": "Uploaded CSV is empty"}), 400

        print(f"📥 收到 {len(input_df)} 行数据进行批量预测")

        # 预处理所有样本
        processed_rows = []
        for idx, row in input_df.iterrows():
            sample_dict = row.to_dict()
            processed_sample = preprocess_single_sample(sample_dict)
            processed_rows.append(processed_sample)

        # 核心预测（不生成图，节省时间）
        results = predict_core(processed_rows, return_explanation=True, generate_plot=False)

        # 构造简洁结果用于 CSV
        csv_results = []
        for r in results:
            exp = r["explanation"]
            top_features = "; ".join([
                f"{feat['feature']}({feat['shap_value']:+.3f})"
                for feat in exp["important_features"]
            ]) if exp else ""
            top_rules = "; ".join(exp["top_rules"][:3]) if exp else ""

            csv_results.append({
                "probability": r["probability"],
                "top_features": top_features,
                "top_rules": top_rules
            })

        # ========== 生成结果 DataFrame ==========
        # 自动识别原始第一列作为 ID 列
        original_id_col_name = input_df.columns[0]
        original_id_series = input_df.iloc[:, 0].copy()

        result_df = pd.DataFrame()
        result_df['prediction_probability'] = [r['probability'] for r in csv_results]  # 第一列：预测概率
        result_df[original_id_col_name] = original_id_series.values  # 第二列：原始ID

        # 插入其余原始列（从第2列开始）
        for col in input_df.columns[1:]:
            result_df[col] = input_df[col].values

        # 最后追加解释列
        result_df['top_3_features'] = [r['top_features'] for r in csv_results]
        result_df['top_3_rules'] = [r['top_rules'] for r in csv_results]

        # ========== 输出 CSV ==========
        output_filename = f"predictions_{uuid.uuid4().hex[:8]}.csv"
        output_path = os.path.join(tempfile.gettempdir(), output_filename)

        result_df.to_csv(output_path, index=False, encoding='utf-8-sig')  # utf-8-sig 支持中文

        response = send_file(
            output_path,
            mimetype='text/csv',
            as_attachment=True,
            download_name=output_filename
        )

        # 请求结束后删除临时文件
        @response.call_on_close
        def remove_file():
            try:
                os.remove(output_path)
                logger.info(f"✅ 临时文件已删除: {output_path}")
            except Exception as e:
                logger.error(f"❌ 临时文件删除失败: {e}")

        return response

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"批量预测失败: {str(e)}"}), 500


# ========== 健康检查 ==========
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "message": "GBDT+LR API is running",
        "model_loaded": True,
        "features": {
            "continuous": len(continuous_features),
            "categorical": len(category_features),
            "total_input_features": len(train_feature_names)
        },
        "model_info": {
            "actual_n_estimators": int(actual_n_estimators)  # 🆕 新增
        }
    })


if __name__ == '__main__':
    print("🚀 启动 Flask API 服务...")
    print("   访问健康检查: http://localhost:5000/health")
    print("   单样本预测: POST http://localhost:5000/predict")
    print("   批量预测CSV: POST http://localhost:5000/predict_batch_csv (上传 file)")
    app.run(host='0.0.0.0', port=5000, debug=True)
