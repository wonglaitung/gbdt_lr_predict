# local_batch_predict.py
import pandas as pd
import joblib
import os
import argparse
import logging
from pathlib import Path

# ========== 日志 ==========
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== 模型加载 ==========
MODEL_DIR = 'output'

required_files = [
    'gbdt_model.pkl',
    'lr_model.pkl',
    'train_feature_names.csv',
    'category_features.csv',
    'continuous_features.csv',
    'actual_n_estimators.csv'
]

for f in required_files:
    if not os.path.exists(os.path.join(MODEL_DIR, f)):
        raise FileNotFoundError(f"❌ 找不到必需文件: {f}")

gbdt_model = joblib.load(os.path.join(MODEL_DIR, 'gbdt_model.pkl'))
lr_model = joblib.load(os.path.join(MODEL_DIR, 'lr_model.pkl'))
train_feature_names = pd.read_csv(os.path.join(MODEL_DIR, 'train_feature_names.csv'))['feature'].tolist()
category_features = pd.read_csv(os.path.join(MODEL_DIR, 'category_features.csv'))['feature'].tolist()
continuous_features = pd.read_csv(os.path.join(MODEL_DIR, 'continuous_features.csv'))['feature'].tolist()
actual_n_estimators = pd.read_csv(os.path.join(MODEL_DIR, 'actual_n_estimators.csv'))['n_estimators'].iloc[0]
category_prefixes = [col + "_" for col in category_features]

logger.info(f"✅ 模型加载完成，实际树数量: {actual_n_estimators}")


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


# ========== 预处理单样本 ==========
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


# ========== 核心预测（带解释） ==========
def predict_core_local(sample_df_list):
    """
    与 app.py 中的 predict_core 逻辑一致（return_explanation=True, generate_plot=False）
    """
    if not sample_df_list:
        return []

    batch_df = pd.concat(sample_df_list, ignore_index=True)

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

    # Step 4: SHAP 解释（必须安装 shap）
    try:
        import shap
        explainer = shap.TreeExplainer(gbdt_model.booster_)
        shap_values_batch = explainer.shap_values(batch_df)
        if isinstance(shap_values_batch, list):
            shap_values_batch = shap_values_batch[1]  # 正类
    except Exception as e:
        logger.error(f"SHAP 初始化失败，无法生成解释: {e}")
        # 若无 SHAP，返回空解释（与 API 行为一致：API 也会失败，但这里我们 fallback）
        results = []
        for i in range(len(sample_df_list)):
            results.append({
                "probability": round(float(probabilities[i]), 4),
                "explanation": {
                    "important_features": [],
                    "top_rules": [],
                    "feature_based_rules": []
                }
            })
        return results

    results = []
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

        # 特征关联规则（此处简化：只取 top_rules 用于 CSV）
        # 注意：app.py 的 CSV 只用了 top_rules[:3]，未用 feature_based_rules
        explanation = {
            "important_features": important_features,
            "top_rules": top_rules,
            "feature_based_rules": []  # CSV 中未使用，可忽略
        }

        results.append({
            "probability": round(float(probabilities[idx]), 4),
            "explanation": explanation
        })

    return results


# ========== 主函数 ==========
def main():
    parser = argparse.ArgumentParser(description="本地批量预测（与 app.py /predict_batch_csv 逻辑一致）")
    parser.add_argument("input_csv", help="输入的 CSV 文件路径")
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    if not input_path.exists():
        logger.error(f"❌ 文件不存在: {input_path}")
        return
    if not input_path.suffix.lower() == '.csv':
        logger.error("❌ 仅支持 .csv 文件")
        return

    try:
        input_df = pd.read_csv(input_path)
        if input_df.empty:
            logger.warning("输入 CSV 为空")
            return

        logger.info(f"处理 {len(input_df)} 行数据...")

        # 预处理
        processed_rows = []
        for _, row in input_df.iterrows():
            processed = preprocess_single_sample(row.to_dict())
            processed_rows.append(processed)

        # 预测（带解释）
        results = predict_core_local(processed_rows)

        # ========== 构造 CSV 结果（与 app.py 完全一致） ==========
        csv_results = []
        for r in results:
            exp = r["explanation"]
            top_features = "; ".join([
                f"{feat['feature']}({feat['shap_value']:+.3f})"
                for feat in exp["important_features"]
            ]) if exp["important_features"] else ""
            top_rules = "; ".join(exp["top_rules"][:3]) if exp["top_rules"] else ""

            csv_results.append({
                "probability": r["probability"],
                "top_features": top_features,
                "top_rules": top_rules
            })

        # ========== 生成结果 DataFrame（与 app.py 一致） ==========
        original_id_col_name = input_df.columns[0]
        original_id_series = input_df.iloc[:, 0].copy()

        result_df = pd.DataFrame()
        result_df['prediction_probability'] = [r['probability'] for r in csv_results]  # 第一列
        result_df[original_id_col_name] = original_id_series.values  # 第二列：原始 ID

        # 插入其余原始列（从第2列开始）
        for col in input_df.columns[1:]:
            result_df[col] = input_df[col].values

        # 最后追加解释列
        result_df['top_3_features'] = [r['top_features'] for r in csv_results]
        result_df['top_3_rules'] = [r['top_rules'] for r in csv_results]

        # ========== 保存结果 ==========
        output_path = input_path.parent / f"predicted_{input_path.name}"
        result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"✅ 结果已保存: {output_path}")

    except Exception as e:
        logger.exception(f"预测失败: {e}")


if __name__ == '__main__':
    main()