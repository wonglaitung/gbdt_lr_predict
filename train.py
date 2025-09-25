import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from lightgbm import log_evaluation
import matplotlib.pyplot as plt
import os
import joblib

# ========== 工具函数：解析叶子节点路径（增强版） ==========
def get_leaf_path_enhanced(booster, tree_index, leaf_index, feature_names, category_prefixes):
    """
    解析指定叶子节点的决策路径，支持翻译 one-hot 类别特征
    """
    try:
        model_dump = booster.dump_model()
        if tree_index >= len(model_dump['tree_info']):
            return None
        tree_info = model_dump['tree_info'][tree_index]['tree_structure']
    except Exception as e:
        print(f"获取树结构失败: {e}")
        return None

    node_stack = [(tree_info, [])]  # (当前节点, 路径列表)

    while node_stack:
        node, current_path = node_stack.pop()

        # 如果是目标叶子节点
        if 'leaf_index' in node and node['leaf_index'] == leaf_index:
            return current_path

        # 如果是分裂节点
        if 'split_feature' in node:
            feat_idx = node['split_feature']
            if feat_idx >= len(feature_names):
                feat_name = f"Feature_{feat_idx}"
            else:
                feat_name = feature_names[feat_idx]

            threshold = node.get('threshold', 0.0)
            decision_type = node.get('decision_type', '<=')

            # 检查是否为 one-hot 类别特征
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
                # 类别特征通常用 > 0.5 判断是否激活
                # 假设右子树是“等于该类别”
                right_rule = f"{original_col} == '{category_value}'"
                left_rule = f"{original_col} != '{category_value}'"
            else:
                # 连续特征
                if decision_type == '<=' or decision_type == 'no_greater':
                    right_rule = f"{feat_name} > {threshold:.4f}"
                    left_rule = f"{feat_name} <= {threshold:.4f}"
                else:
                    right_rule = f"{feat_name} {decision_type} {threshold:.4f}"
                    left_rule = f"{feat_name} not {decision_type} {threshold:.4f}"

            # 添加左右子树到栈
            if 'right_child' in node:
                node_stack.append((node['right_child'], current_path + [right_rule]))
            if 'left_child' in node:
                node_stack.append((node['left_child'], current_path + [left_rule]))

    return None  # 未找到路径


# ========== 数据预处理 ==========
def preProcess():
    path = 'data/'
    try:
        df_train = pd.read_csv(path + 'train.csv', encoding='utf-8')
        df_test = pd.read_csv(path + 'test.csv', encoding='utf-8')
    except UnicodeDecodeError:
        print("⚠️ UTF-8 解码失败，尝试使用 GBK 编码...")
        df_train = pd.read_csv(path + 'train.csv', encoding='gbk')
        df_test = pd.read_csv(path + 'test.csv', encoding='gbk')
    
    test_ids = df_test['Id'].copy()
    df_train.drop(['Id'], axis=1, inplace=True)
    df_test.drop(['Id'], axis=1, inplace=True)
    
    df_test['Label'] = -1
    data = pd.concat([df_train, df_test], ignore_index=True)
    data = data.fillna(-1)
    
    data.to_csv('data/data.csv', index=False, encoding='utf-8')
    return data, test_ids


# ========== GBDT + LR 核心训练预测函数 ==========
def gbdt_lr_predict(data, category_feature, continuous_feature, test_ids):
    """
    使用 GBDT + LR，增强可解释性输出
    """
    # 创建输出目录
    os.makedirs('output', exist_ok=True)

    # ========== Step 1: 类别特征 One-Hot 编码 ==========
    for col in category_feature:
        if data[col].dtype == 'object':
            data[col] = data[col].astype('category')
        onehot_feats = pd.get_dummies(data[col], prefix=col)
        data.drop([col], axis=1, inplace=True)
        data = pd.concat([data, onehot_feats], axis=1)

    train = data[data['Label'] != -1].copy()
    target = train.pop('Label')
    test = data[data['Label'] == -1].copy()
    test.drop(['Label'], axis=1, inplace=True)

    # 划分训练/验证集
    x_train, x_val, y_train, y_val = train_test_split(
        train, target, test_size=0.2, random_state=2020, stratify=target
    )

    # ========== Step 2: 训练 GBDT ==========
    n_estimators = 32
    num_leaves = 64

    model = lgb.LGBMClassifier(
        objective='binary',
        boosting_type='gbdt',
        subsample=0.8,
        min_child_weight=0.1,
        min_child_samples=10,
        colsample_bytree=0.7,
        num_leaves=num_leaves,
        learning_rate=0.05,
        n_estimators=n_estimators,
        random_state=2020,
        n_jobs=-1
    )

    model.fit(
        x_train, y_train,
        eval_set=[(x_train, y_train), (x_val, y_val)],
        eval_metric='binary_logloss',
        callbacks=[
            log_evaluation(0),
            lgb.early_stopping(stopping_rounds=5, verbose=False)
        ]
    )

    # ========== 🆕 获取实际训练的树数量 ==========
    actual_n_estimators = model.best_iteration_
    print(f"✅ 实际训练树数量: {actual_n_estimators} (原计划: {n_estimators})")

    # ========== Step 2.5: 输出 GBDT 特征重要性 ==========
    feat_imp = pd.DataFrame({
        'Feature': x_train.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\n" + "="*60)
    print("📊 GBDT Top 20 重要特征:")
    print("="*60)
    print(feat_imp.head(20))
    feat_imp.to_csv('output/gbdt_feature_importance.csv', index=False)
    print("✅ 已保存至 output/gbdt_feature_importance.csv")

    # ========== Step 3: 获取叶子节点索引 ==========
    gbdt_feats_train = model.booster_.predict(train.values, pred_leaf=True)
    gbdt_feats_test = model.booster_.predict(test.values, pred_leaf=True)

    print("\n✅ GBDT 叶子节点索引 shape:", gbdt_feats_train.shape)
    print("✅ 前5个样本叶子索引:\n", gbdt_feats_train[:5])

    # ========== Step 3.5: 示例样本叶子路径 + 规则解析 ==========
    print("\n" + "="*70)
    print("🔍 解析叶子节点示例：gbdt_leaf_5_22")
    print("="*70)
    
    try:
        # 生成类别前缀列表（用于识别 one-hot 列）
        category_prefixes = [col + "_" for col in category_feature]
        
        # 解析第5棵树、第22号叶子
        leaf_rule = get_leaf_path_enhanced(
            model.booster_, 
            tree_index=5, 
            leaf_index=22, 
            feature_names=x_train.columns.tolist(),
            category_prefixes=category_prefixes
        )
        
        if leaf_rule:
            print(f"✅ 叶子节点 gbdt_leaf_5_22 的决策路径：")
            for i, rule in enumerate(leaf_rule, 1):
                print(f"   {i}. {rule}")
        else:
            print("⚠️ 未找到该叶子节点路径（可能索引越界或树结构变化）")
            
    except Exception as e:
        print("⚠️ 解析失败:", e)

    # ========== Step 4: 对叶子节点做 One-Hot 编码 ==========
    # 🆕 使用 actual_n_estimators 替代硬编码 n_estimators
    gbdt_feats_name = ['gbdt_leaf_' + str(i) for i in range(actual_n_estimators)]

    df_train_gbdt_feats = pd.DataFrame(gbdt_feats_train, columns=gbdt_feats_name)
    df_test_gbdt_feats = pd.DataFrame(gbdt_feats_test, columns=gbdt_feats_name)

    train_len = df_train_gbdt_feats.shape[0]
    data_gbdt = pd.concat([df_train_gbdt_feats, df_test_gbdt_feats], ignore_index=True)

    for col in gbdt_feats_name:
        onehot_feats = pd.get_dummies(data_gbdt[col], prefix=col)
        data_gbdt.drop([col], axis=1, inplace=True)
        data_gbdt = pd.concat([data_gbdt, onehot_feats], axis=1)

    train_lr = data_gbdt.iloc[:train_len, :].reset_index(drop=True)
    test_lr = data_gbdt.iloc[train_len:, :].reset_index(drop=True)

    # ========== Step 5: 训练 LR 模型 ==========
    x_train_lr, x_val_lr, y_train_lr, y_val_lr = train_test_split(
        train_lr, target, test_size=0.3, random_state=2018, stratify=target
    )

    lr = LogisticRegression(
        penalty='l2',
        C=0.1,
        solver='liblinear',
        random_state=2018,
        max_iter=1000
    )
    lr.fit(x_train_lr, y_train_lr)

    tr_logloss = log_loss(y_train_lr, lr.predict_proba(x_train_lr)[:, 1])
    val_logloss = log_loss(y_val_lr, lr.predict_proba(x_val_lr)[:, 1])
    print('\n✅ Train LogLoss:', tr_logloss)
    print('✅ Val LogLoss:', val_logloss)

    # ========== Step 5.5: 输出 LR 系数（哪些叶子规则最重要） ==========
    lr_coef = pd.DataFrame({
        'Leaf_Feature': x_train_lr.columns,
        'Coefficient': lr.coef_[0]
    }).sort_values('Coefficient', key=abs, ascending=False)

    print("\n" + "="*60)
    print("📊 LR Top 20 重要叶子特征（按系数绝对值排序）:")
    print("="*60)
    print(lr_coef.head(20))
    lr_coef.to_csv('output/lr_leaf_coefficients.csv', index=False)
    print("✅ 已保存至 output/lr_leaf_coefficients.csv")

    # ========== Step 5.6: 对高权重叶子进行规则解析 ==========
    print("\n" + "="*70)
    print("🧠 解析 LR 中高权重叶子节点对应的原始规则")
    print("="*70)
    
    top_leaves = lr_coef.head(5)  # 解析前5个最重要叶子
    category_prefixes = [col + "_" for col in category_feature]
    
    for idx, row in top_leaves.iterrows():
        leaf_feat = row['Leaf_Feature']
        coef = row['Coefficient']
        
        # 解析叶子名称，如 "gbdt_leaf_5_22"
        if leaf_feat.startswith('gbdt_leaf_'):
            parts = leaf_feat.split('_')
            if len(parts) >= 4:
                tree_idx = int(parts[2])
                leaf_idx = int(parts[3])
                
                print(f"\n🔎 解析 {leaf_feat} (LR系数: {coef:.4f})")
                try:
                    rule = get_leaf_path_enhanced(
                        model.booster_,
                        tree_index=tree_idx,
                        leaf_index=leaf_idx,
                        feature_names=x_train.columns.tolist(),
                        category_prefixes=category_prefixes
                    )
                    if rule:
                        for i, r in enumerate(rule, 1):
                            print(f"   {i}. {r}")
                    else:
                        print("   ⚠️ 路径未找到")
                except Exception as e:
                    print(f"   ⚠️ 解析失败: {e}")

    # ========== Step 6: SHAP 解释（全局 + 单样本） ==========
    print("\n" + "="*60)
    print("🎨 正在生成 SHAP 可解释性图表...")
    print("="*60)
    
    try:
        import shap
        explainer = shap.TreeExplainer(model.booster_)
        
        sample_size = min(100, len(x_val))
        x_val_sample = x_val.iloc[:sample_size]
        shap_values = explainer.shap_values(x_val_sample)

        # 1. 全局特征重要性图
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, x_val_sample, feature_names=x_val_sample.columns.tolist(), show=False)
        plt.title("SHAP Feature Importance (GBDT)", fontsize=16)
        plt.tight_layout()
        plt.savefig("output/shap_summary_plot.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ SHAP 全局特征重要性图已保存: output/shap_summary_plot.png")

        # 2. 单样本瀑布图（第0个样本）
        if len(shap_values) > 0:
            plt.figure(figsize=(12, 8))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=explainer.expected_value,
                    data=x_val_sample.iloc[0],
                    feature_names=x_val_sample.columns.tolist()
                ),
                show=False
            )
            plt.title("SHAP Waterfall Plot - Sample 0", fontsize=16)
            plt.tight_layout()
            plt.savefig("output/shap_waterfall_sample_0.png", dpi=150, bbox_inches='tight')
            plt.close()
            print("✅ SHAP 单样本瀑布图已保存: output/shap_waterfall_sample_0.png")

    except Exception as e:
        print("⚠️ SHAP 解释失败（请确保已安装 shap）:", e)

    # ========== Step 7: 预测测试集并保存带 Id 的结果 ==========
    y_pred = lr.predict_proba(test_lr)[:, 1]

    submission = pd.DataFrame({
        'Id': test_ids,
        'Probability': y_pred
    })

    # ========== Step 8: 保存模型和必要信息用于 API ==========
    joblib.dump(model, 'output/gbdt_model.pkl')
    joblib.dump(lr, 'output/lr_model.pkl')
    
    # 🆕 保存实际树数量，供 API 使用
    pd.Series([actual_n_estimators]).to_csv('output/actual_n_estimators.csv', index=False, header=['n_estimators'])
    
    pd.Series(x_train.columns).to_csv('output/train_feature_names.csv', index=False, header=['feature'])
    pd.Series(category_feature).to_csv('output/category_features.csv', index=False, header=['feature'])
    pd.Series(continuous_feature).to_csv('output/continuous_features.csv', index=False, header=['feature'])
    
    print("✅ 模型和元数据已保存，可用于 API 服务")

    submission.to_csv('output/submission_gbdt_lr.csv', index=False)
    print("\n🎉 预测结果已保存至 output/submission_gbdt_lr.csv（含 Id 列）")

    return y_pred


# ========== 主程序入口 ==========
if __name__ == '__main__':
    print("🚀 开始数据预处理...")
    data, test_ids = preProcess()

    # ========== 从配置文件读取特征定义 ==========
    print("📂 正在加载特征配置...")
    feature_config = pd.read_csv('config/features.csv')
    continuous_feature = feature_config[feature_config['feature_type'] == 'continuous']['feature_name'].tolist()
    category_feature = feature_config[feature_config['feature_type'] == 'category']['feature_name'].tolist()

    print("✅ 连续特征:", continuous_feature)
    print("✅ 类别特征:", category_feature)

    print("🧠 开始训练 GBDT + LR 模型...")
    predictions = gbdt_lr_predict(data, category_feature, continuous_feature, test_ids)

    print("\n✅ 模型训练与预测完成！")
    print("📊 所有可解释性报告已生成在 output/ 目录下：")
    print("   - gbdt_feature_importance.csv")
    print("   - lr_leaf_coefficients.csv")
    print("   - shap_summary_plot.png")
    print("   - shap_waterfall_sample_0.png")
    print("   - submission_gbdt_lr.csv")
    print("   - actual_n_estimators.csv") 
