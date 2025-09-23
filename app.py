import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import base64
import io
from flask import Flask, request, jsonify
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import shap
import os

app = Flask(__name__)

# ========== 全局变量：加载模型和元数据 ==========
MODEL_DIR = 'output'

required_files = [
    'gbdt_model.pkl',
    'lr_model.pkl',
    'train_feature_names.csv',
    'category_features.csv',
    'continuous_features.csv'
]

for f in required_files:
    if not os.path.exists(os.path.join(MODEL_DIR, f)):
        raise FileNotFoundError(f"❌ 找不到必需文件: {f}。请先运行训练脚本生成模型。")

gbdt_model = joblib.load(os.path.join(MODEL_DIR, 'gbdt_model.pkl'))
lr_model = joblib.load(os.path.join(MODEL_DIR, 'lr_model.pkl'))

train_feature_names = pd.read_csv(os.path.join(MODEL_DIR, 'train_feature_names.csv'))['feature'].tolist()
category_features = pd.read_csv(os.path.join(MODEL_DIR, 'category_features.csv'))['feature'].tolist()
continuous_features = pd.read_csv(os.path.join(MODEL_DIR, 'continuous_features.csv'))['feature'].tolist()

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


# ========== API 路由 ==========
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        if not isinstance(data, dict):
            return jsonify({"error": "JSON body must be an object"}), 400

        sample_df = preprocess_single_sample(data)
        
        # Step 1: GBDT 预测叶子索引
        leaf_indices = gbdt_model.booster_.predict(sample_df.values, pred_leaf=True)[0]
        
        # Step 2: 构造叶子 one-hot 特征
        n_trees = len(leaf_indices)
        leaf_df = pd.DataFrame([leaf_indices], columns=[f"gbdt_leaf_{i}" for i in range(n_trees)])
        
        leaf_dummies_list = []
        for i in range(n_trees):
            col_name = f"gbdt_leaf_{i}"
            dummies = pd.get_dummies(leaf_df[col_name], prefix=col_name)
            leaf_dummies_list.append(dummies)
        
        if leaf_dummies_list:
            leaf_dummies_combined = pd.concat(leaf_dummies_list, axis=1)
            leaf_df = pd.concat([
                leaf_df.drop(columns=[f"gbdt_leaf_{i}" for i in range(n_trees)]),
                leaf_dummies_combined
            ], axis=1)
        else:
            leaf_df = leaf_df.drop(columns=[f"gbdt_leaf_{i}" for i in range(n_trees)])
        
        lr_feature_names = getattr(lr_model, 'feature_names_in_', [f"feature_{i}" for i in range(len(lr_model.coef_[0]))])
        missing_leaf_cols = set(lr_feature_names) - set(leaf_df.columns)
        if missing_leaf_cols:
            missing_leaf_df = pd.DataFrame(0, index=leaf_df.index, columns=list(missing_leaf_cols))
            leaf_df = pd.concat([leaf_df, missing_leaf_df], axis=1)
        leaf_df = leaf_df.reindex(columns=lr_feature_names, fill_value=0)
        
        # Step 3: LR 预测概率
        prob = lr_model.predict_proba(leaf_df)[0, 1]
        
        # Step 4: 生成 SHAP 值（用于重要特征和规则关联）
        explainer = shap.TreeExplainer(gbdt_model.booster_)
        shap_values = explainer.shap_values(sample_df)
        shap_vals = shap_values[0]
        
        # 重要特征（SHAP 值前5）
        feature_imp = [(train_feature_names[i], float(shap_vals[i])) for i in range(len(shap_vals))]
        feature_imp.sort(key=lambda x: abs(x[1]), reverse=True)
        important_features = [
            {"feature": feat, "shap_value": round(val, 4)}
            for feat, val in feature_imp[:3]
        ]
        
        top_shap_features = [feat for feat, val in feature_imp[:5]]  # 前5重要特征名
        
        # Step 5: 生成两种规则解释
        ## 5.1 原始路径规则（前3棵树）
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
        
        ## 5.2 🆕 特征关联规则（SHAP 重要特征对应的路径规则）
        feature_rules = []
        for tree_idx in range(min(10, len(leaf_indices))):  # 搜索前10棵树
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
                    # 检查规则是否涉及 top_shap_features
                    for feat in top_shap_features:
                        # 匹配特征名（支持 C1_abc 和 I5 等格式）
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
        
        # Step 6: 生成 SHAP 图
        shap_plot_b64 = ""
        try:
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_vals,
                    base_values=explainer.expected_value,
                    data=sample_df.iloc[0],
                    feature_names=sample_df.columns.tolist()
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

        # 构造响应
        response = {
            "probability": round(float(prob), 4),
            "explanation": {
                "important_features": important_features,
                "shap_plot_base64": shap_plot_b64,
                "top_rules": top_rules,  # 原始路径规则
                "feature_based_rules": feature_rules[:5]  # 🆕 新增：SHAP特征关联规则
            }
        }

        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


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
        }
    })


if __name__ == '__main__':
    print("🚀 启动 Flask API 服务...")
    print("   访问健康检查: http://localhost:5000/health")
    print("   预测接口: POST http://localhost:5000/predict")
    app.run(host='0.0.0.0', port=5000, debug=True)
