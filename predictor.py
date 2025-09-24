import pandas as pd
import joblib
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_DIR = 'output'

# ========== 加载模型和元数据 ==========
def load_models():
    required_files = [
        'gbdt_model.pkl',
        'lr_model.pkl',
        'train_feature_names.csv',
        'category_features.csv',
        'continuous_features.csv',
        'actual_n_estimators.csv'
    ]
    model_dir = Path(MODEL_DIR)
    for f in required_files:
        if not (model_dir / f).exists():
            raise FileNotFoundError(f"❌ 找不到必需文件: {model_dir / f}")

    gbdt_model = joblib.load(model_dir / 'gbdt_model.pkl')
    lr_model = joblib.load(model_dir / 'lr_model.pkl')
    train_feature_names = pd.read_csv(model_dir / 'train_feature_names.csv')['feature'].tolist()
    category_features = pd.read_csv(model_dir / 'category_features.csv')['feature'].tolist()
    continuous_features = pd.read_csv(model_dir / 'continuous_features.csv')['feature'].tolist()
    actual_n_estimators = pd.read_csv(model_dir / 'actual_n_estimators.csv')['n_estimators'].iloc[0]
    category_prefixes = [col + "_" for col in category_features]

    logger.info(f"✅ 模型加载完成，实际树数量: {actual_n_estimators}")
    return {
        'gbdt_model': gbdt_model,
        'lr_model': lr_model,
        'train_feature_names': train_feature_names,
        'category_features': category_features,
        'continuous_features': continuous_features,
        'actual_n_estimators': actual_n_estimators,
        'category_prefixes': category_prefixes
    }


# ========== 工具函数：解析叶子路径 ==========
def get_leaf_path_enhanced(booster, tree_index, leaf_index, feature_names, category_prefixes):
    try:
        model_dump = booster.dump_model()
        if tree_index >= len(model_dump['tree_info']):
            return None
        tree_info = model_dump['tree_info'][tree_index]['tree_structure']
    except Exception as e:
        logger.warning(f"解析树结构失败: {e}")
        return None

    node_stack = [(tree_info, [])]
    while node_stack:
        node, current_path = node_stack.pop()
        if 'leaf_index' in node and node['leaf_index'] == leaf_index:
            return current_path
        if 'split_feature' in node:
            feat_idx = node['split_feature']
            feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"Feature_{feat_idx}"
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
                if decision_type in ('<=', 'no_greater'):
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


# ========== 预处理单样本 ==========
def preprocess_single_sample(sample_dict, continuous_features, category_features, train_feature_names):
    sample_df = pd.DataFrame([sample_dict])
    # 连续特征
    for col in continuous_features:
        if col not in sample_df.columns:
            sample_df[col] = -1
        else:
            sample_df[col] = pd.to_numeric(sample_df[col], errors='coerce').fillna(-1)
    # 分类特征
    all_dummies_list = []
    for col in category_features:
        if col in sample_df.columns:
            val = sample_df[col].iloc[0]
            sample_df[col] = "-1" if pd.isna(val) or val == "" else str(val)
        else:
            sample_df[col] = "-1"
        sample_df[col] = sample_df[col].astype('category')
        dummies = pd.get_dummies(sample_df[col], prefix=col)
        # 补齐训练时的 dummy 列
        missing_cols = [train_col for train_col in train_feature_names if train_col.startswith(col + "_") and train_col not in dummies.columns]
        if missing_cols:
            missing_df = pd.DataFrame(0, index=dummies.index, columns=missing_cols)
            dummies = pd.concat([dummies, missing_df], axis=1)
        all_dummies_list.append(dummies)
    # 合并
    if all_dummies_list:
        dummies_combined = pd.concat(all_dummies_list, axis=1)
        sample_df = pd.concat([sample_df.drop(columns=category_features), dummies_combined], axis=1)
    else:
        sample_df = sample_df.drop(columns=category_features)
    # 补齐所有训练特征
    missing_final_cols = set(train_feature_names) - set(sample_df.columns)
    if missing_final_cols:
        missing_final_df = pd.DataFrame(0, index=sample_df.index, columns=list(missing_final_cols))
        sample_df = pd.concat([sample_df, missing_final_df], axis=1)
    return sample_df.reindex(columns=train_feature_names, fill_value=0)


# ========== 核心预测函数 ==========
def predict_core(sample_df_list, models, return_explanation=True, generate_plot=False):
    """
    与 app.py 中的 predict_core 完全一致
    """
    if not sample_df_list:
        return []

    from sklearn.linear_model import LogisticRegression
    import numpy as np

    batch_df = pd.concat(sample_df_list, ignore_index=True)
    gbdt_model = models['gbdt_model']
    lr_model = models['lr_model']
    train_feature_names = models['train_feature_names']
    actual_n_estimators = models['actual_n_estimators']
    category_prefixes = models['category_prefixes']

    # Step 1: GBDT 叶子索引
    leaf_indices_batch = gbdt_model.booster_.predict(batch_df.values, pred_leaf=True)
    n_trees = actual_n_estimators

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

    if not return_explanation:
        return [{"probability": round(float(p), 4), "explanation": None} for p in probabilities]

    # Step 4: SHAP 解释
    try:
        import shap
        import matplotlib.pyplot as plt
        import base64
        import io
        explainer = shap.TreeExplainer(gbdt_model.booster_)
        shap_values_batch = explainer.shap_values(batch_df)
        if isinstance(shap_values_batch, list):
            shap_values_batch = shap_values_batch[1]
    except Exception as e:
        logger.error(f"SHAP 初始化失败: {e}")
        # 返回空解释
        return [{
            "probability": round(float(probabilities[i]), 4),
            "explanation": {
                "important_features": [],
                "shap_plot_base64": "",
                "top_rules": [],
                "feature_based_rules": []
            }
        } for i in range(len(sample_df_list))]

    results = []
    for idx in range(len(sample_df_list)):
        shap_vals = shap_values_batch[idx]
        feature_imp = [(train_feature_names[i], float(shap_vals[i])) for i in range(len(shap_vals))]
        feature_imp.sort(key=lambda x: abs(x[1]), reverse=True)
        important_features = [{"feature": feat, "shap_value": round(val, 4)} for feat, val in feature_imp[:3]]
        top_shap_features = [feat for feat, val in feature_imp[:5]]
        leaf_indices = leaf_indices_batch[idx]

        # 原始路径规则
        path_rules = []
        for tree_idx in range(min(3, len(leaf_indices))):
            leaf_idx = leaf_indices[tree_idx]
            rule = get_leaf_path_enhanced(gbdt_model.booster_, tree_idx, leaf_idx, train_feature_names, category_prefixes)
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
            rule = get_leaf_path_enhanced(gbdt_model.booster_, tree_idx, leaf_idx, train_feature_names, category_prefixes)
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

        # 生成图（仅第一个样本）
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
                logger.warning(f"SHAP plot generation failed: {e}")

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