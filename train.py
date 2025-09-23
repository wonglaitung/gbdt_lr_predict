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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ========== 工具函数：基于 SHAP 依赖关系自动建议分箱边界 ==========
def suggest_bins_from_shap(feature_values, shap_values, feature_name, n_bins=5):
    """
    基于 SHAP 依赖关系自动建议分箱边界
    :param feature_values: 该特征的原始值 (1D array)
    :param shap_values: 对应的 SHAP 值 (1D array)
    :param feature_name: 特征名
    :param n_bins: 尝试检测的分箱数（默认5）
    :return: dict { "should_bin": bool, "suggested_bins": list, "reason": str }
    """
    try:
        # 合并并排序
        df = pd.DataFrame({
            'feature_val': feature_values,
            'shap_val': shap_values
        }).sort_values('feature_val').reset_index(drop=True)

        if len(df) < 10:
            return {"should_bin": False, "suggested_bins": [], "reason": "样本太少"}

        # 按特征值分 n_bins 组（等频）
        df['group'] = pd.qcut(df['feature_val'], q=min(n_bins, len(df)//5), duplicates='drop')
        group_means = df.groupby('group', observed=False)['shap_val'].mean()
        group_bounds = df.groupby('group', observed=False)['feature_val'].agg(['min', 'max'])

        # 检查相邻组的 SHAP 均值差异
        mean_diffs = group_means.diff().abs().dropna()
        if len(mean_diffs) == 0:
            return {"should_bin": False, "suggested_bins": [], "reason": "无显著跳跃"}

        # 如果最大跳跃 > 平均跳跃的 1.5 倍，认为有分箱价值
        avg_diff = mean_diffs.mean()
        max_diff = mean_diffs.max()
        if max_diff < avg_diff * 1.5:
            return {"should_bin": False, "suggested_bins": [], "reason": "SHAP变化平缓，无需分箱"}

        # 建议分箱边界：取每组最大值（最后一组除外）
        suggested_bins = group_bounds['max'].iloc[:-1].tolist()
        # 四舍五入到合理精度（根据数据范围）
        if df['feature_val'].max() > 1000:
            suggested_bins = [round(b, -1) for b in suggested_bins]  # 十位取整
        elif df['feature_val'].max() > 100:
            suggested_bins = [round(b, 0) for b in suggested_bins]   # 个位取整
        else:
            suggested_bins = [round(b, 2) for b in suggested_bins]   # 两位小数

        return {
            "should_bin": True,
            "suggested_bins": sorted(suggested_bins),
            "reason": f"检测到显著SHAP跳跃（最大跳跃={max_diff:.3f} > 平均{avg_diff:.3f}）"
        }

    except Exception as e:
        return {"should_bin": False, "suggested_bins": [], "reason": f"分析失败: {str(e)}"}


# ========== 工具函数：建议是否进行对数变换 ==========
def suggest_log_transform(feature_values, shap_values, feature_name):
    """
    建议是否对特征进行对数变换
    条件：
      - 特征值必须全为正数
      - 分布右偏（偏度 > 1）
      - SHAP 值随特征增长但增速放缓（二阶导为负）
    """
    try:
        df = pd.DataFrame({
            'x': feature_values,
            'shap': shap_values
        }).sort_values('x').reset_index(drop=True)

        # 必须全为正数
        if (df['x'] <= 0).any():
            return {"suggest_log": False, "reason": "包含非正值，无法取对数", "code": ""}

        # 计算偏度
        skewness = df['x'].skew()
        if skewness < 1.0:
            return {"suggest_log": False, "reason": f"偏度太低（{skewness:.2f}），无需对数变换", "code": ""}

        # 检查 SHAP 是否增速放缓：计算一阶差分的差分（近似二阶导）
        if len(df) < 5:
            return {"suggest_log": False, "reason": "样本太少，无法分析趋势", "code": ""}

        df['shap_diff'] = df['shap'].diff()
        df['shap_diff2'] = df['shap_diff'].diff()

        # 取后 80% 数据（避免开头噪声）
        df_tail = df.iloc[int(len(df)*0.2):]
        mean_diff2 = df_tail['shap_diff2'].mean()

        # 如果二阶导为负（增速放缓），建议 log
        if mean_diff2 < -0.001:  # 负值表示凹函数，边际递减
            code = f"df['{feature_name}_log'] = np.log(df['{feature_name}'])"
            return {
                "suggest_log": True,
                "reason": f"右偏分布（偏度={skewness:.2f}）且边际效应递减（二阶导均值={mean_diff2:.4f}）",
                "code": code
            }
        else:
            return {"suggest_log": False, "reason": f"无边际递减效应（二阶导均值={mean_diff2:.4f}）", "code": ""}

    except Exception as e:
        return {"suggest_log": False, "reason": f"分析失败: {str(e)}", "code": ""}


# ========== 工具函数：建议是否添加多项式特征 ==========
def suggest_polynomial(feature_values, shap_values, feature_name, max_degree=3):
    """
    建议是否添加多项式特征（如 x^2, x^3）
    方法：拟合线性、二次、三次模型，看 R² 提升是否显著（>0.05）
    """
    try:
        X = np.array(feature_values).reshape(-1, 1)
        y = np.array(shap_values)

        if len(X) < 10:
            return {"suggest_poly": False, "reason": "样本太少", "degree": None, "code": ""}

        # 拟合线性模型作为 baseline
        lr_linear = LinearRegression()
        lr_linear.fit(X, y)
        r2_linear = r2_score(y, lr_linear.predict(X))

        best_degree = 1
        best_r2 = r2_linear
        best_improvement = 0

        for degree in range(2, max_degree + 1):
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            X_poly = poly.fit_transform(X)
            lr = LinearRegression()
            lr.fit(X_poly, y)
            r2 = r2_score(y, lr.predict(X_poly))
            improvement = r2 - r2_linear

            if improvement > best_improvement and improvement > 0.05:  # 至少提升 0.05 R²
                best_improvement = improvement
                best_degree = degree
                best_r2 = r2

        if best_degree > 1:
            code = f"df['{feature_name}_pow{best_degree}'] = df['{feature_name}'] ** {best_degree}"
            return {
                "suggest_poly": True,
                "degree": best_degree,
                "r2_improvement": best_improvement,
                "reason": f"检测到非线性模式，{best_degree}次多项式可提升解释力（R²提升={best_improvement:.3f}）",
                "code": code
            }
        else:
            return {
                "suggest_poly": False,
                "reason": f"无线性外显著模式（最大R²提升={best_improvement:.3f}）",
                "degree": None,
                "code": ""
            }

    except Exception as e:
        return {"suggest_poly": False, "reason": f"分析失败: {str(e)}", "degree": None, "code": ""}


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
    df_train = pd.read_csv(path + 'train.csv')
    df_test = pd.read_csv(path + 'test.csv')
    
    test_ids = df_test['Id'].copy()
    
    df_train.drop(['Id'], axis=1, inplace=True)
    df_test.drop(['Id'], axis=1, inplace=True)
    
    df_test['Label'] = -1
    data = pd.concat([df_train, df_test], ignore_index=True)
    data = data.fillna(-1)
    
    data.to_csv('data/data.csv', index=False)
    
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

    # ========== 🆕 Step 1.5: 应用智能特征工程建议 ==========
    # 手动添加多项式特征（根据之前建议）
    #data['I5_pow3'] = data['I5'] ** 3
    #data['I11_pow3'] = data['I11'] ** 3
    #data['I13_pow3'] = data['I13'] ** 3
    #data['I6_pow3'] = data['I6'] ** 3
    #data['I3_pow3'] = data['I3'] ** 3
    #data['I8_pow3'] = data['I8'] ** 3
    #data['I2_pow3'] = data['I2'] ** 3
    #pd.cut(data['I5'], bins=[-np.inf, 140.0, 1020.0, 2960.0, 14410.0, np.inf])

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
        min_child_weight=0.5,
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

    # ========== 🆕 获取实际训练的树数量（关键修复！） ==========
    actual_n_estimators = model.best_iteration_
    print(f"✅ 实际训练树数量: {actual_n_estimators} (原计划: {n_estimators})")

    # ========== Step 2.5: 输出 GBDT 特征重要性 ==========
    feat_imp = pd.DataFrame({
        '特征名': x_train.columns,
        '重要性': model.feature_importances_
    }).sort_values('重要性', ascending=False)

    print("\n" + "="*60)
    print("📊 GBDT 前20重要特征:")
    print("="*60)
    print(feat_imp.head(20))
    feat_imp.to_csv('output/gbdt_feature_importance.csv', index=False, encoding='utf-8-sig')
    print("✅ 已保存至 output/gbdt_feature_importance.csv")

    # ========== Step 3: 获取叶子节点索引 ==========
    gbdt_feats_train = model.booster_.predict(train.values, pred_leaf=True)
    gbdt_feats_test = model.booster_.predict(test.values, pred_leaf=True)

    print("\n✅ GBDT 叶子节点索引 shape:", gbdt_feats_train.shape)
    print("✅ 前5个样本叶子索引:\n", gbdt_feats_train[:5])

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
    print('\n✅ 训练集 LogLoss:', tr_logloss)
    print('✅ 验证集 LogLoss:', val_logloss)

    # ========== Step 5.5: 输出 LR 系数（哪些叶子规则最重要） ==========
    lr_coef = pd.DataFrame({
        '叶子特征': x_train_lr.columns,
        '系数': lr.coef_[0]
    }).sort_values('系数', key=abs, ascending=False)

    print("\n" + "="*60)
    print("📊 LR 前20重要叶子特征（按系数绝对值排序）:")
    print("="*60)
    print(lr_coef.head(20))
    lr_coef.to_csv('output/lr_leaf_coefficients.csv', index=False, encoding='utf-8-sig')
    print("✅ 已保存至 output/lr_leaf_coefficients.csv")

    # ========== Step 5.6: 对高权重叶子进行规则解析 ==========
    print("\n" + "="*70)
    print("🧠 解析 LR 中高权重叶子节点对应的原始规则")
    print("="*70)
    
    top_leaves = lr_coef.head(5)  # 解析前5个最重要叶子
    category_prefixes = [col + "_" for col in category_feature]
    
    for idx, row in top_leaves.iterrows():
        leaf_feat = row['叶子特征']
        coef = row['系数']
        
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

        # ========== Step 6.5: 生成 SHAP Dependence Plots + 特征工程建议 ==========
        print("\n" + "="*60)
        print("📈 正在生成 SHAP 依赖图和特征工程建议...")
        print("="*60)

        all_suggestions = []  # 收集所有建议（分箱、log、多项式）

        try:
            # 获取特征重要性排序（用于选择 Top 特征）
            if isinstance(shap_values, list):
                shap_vals_for_imp = shap_values[1]  # 二分类取正类
            else:
                shap_vals_for_imp = shap_values

            # 计算平均绝对 SHAP 值作为重要性
            feature_importance = np.abs(shap_vals_for_imp).mean(axis=0)
            top_feature_indices = np.argsort(feature_importance)[::-1][:10]  # Top 10

            for idx in top_feature_indices:
                feature_name = x_val_sample.columns[idx]
                safe_feature_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in feature_name)

                # ========== 🆕 仅修改以下绘图块 ==========
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(
                    idx,
                    shap_vals_for_imp,
                    x_val_sample,
                    feature_names=x_val_sample.columns.tolist(),
                    show=False
                )
                plt.title(f"SHAP Dependence Plot: {feature_name}", fontsize=14)  # ✅ 改为英文
                plt.xlabel(f"{feature_name} (Value)")                           # ✅ 确保英文
                plt.ylabel("SHAP Value (Contribution to Prediction)")           # ✅ 确保英文
                plt.xticks(rotation=45)
                plt.tight_layout()
                # ========== 修改结束 ==========

                plot_filename = f"shap_dependence_{safe_feature_name}.png"
                plot_path = os.path.join("output", plot_filename)
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"\n✅ 已保存依赖图: {plot_filename}")

                feat_vals = x_val_sample.iloc[:, idx].values
                shap_vals = shap_vals_for_imp[:, idx]

                # ========== 🆕 分箱建议 ==========
                bin_suggestion = suggest_bins_from_shap(feat_vals, shap_vals, feature_name)
                if bin_suggestion["should_bin"] and len(bin_suggestion["suggested_bins"]) >= 2:
                    bins_str = ", ".join(map(str, bin_suggestion["suggested_bins"]))
                    code = f"pd.cut(df['{feature_name}'], bins=[-np.inf, {bins_str}, np.inf])"
                    print(f"    🧠 [分箱建议] {code}")
                    print(f"        理由: {bin_suggestion['reason']}")
                    all_suggestions.append({
                        "特征名": feature_name,
                        "建议类型": "分箱",
                        "代码": code,
                        "理由": bin_suggestion["reason"],
                        "对应图表": plot_filename
                    })

                # ========== 🆕 Log 变换建议 ==========
                log_suggestion = suggest_log_transform(feat_vals, shap_vals, feature_name)
                if log_suggestion["suggest_log"]:
                    print(f"    📊 [对数变换建议] {log_suggestion['code']}")
                    print(f"        理由: {log_suggestion['reason']}")
                    all_suggestions.append({
                        "特征名": feature_name,
                        "建议类型": "对数变换",
                        "代码": log_suggestion["code"],
                        "理由": log_suggestion["reason"],
                        "对应图表": plot_filename
                    })

                # ========== 🆕 多项式特征建议 ==========
                poly_suggestion = suggest_polynomial(feat_vals, shap_vals, feature_name)
                if poly_suggestion["suggest_poly"]:
                    print(f"    📈 [多项式建议] {poly_suggestion['code']}")
                    print(f"        理由: {poly_suggestion['reason']}")
                    all_suggestions.append({
                        "特征名": feature_name,
                        "建议类型": f"多项式_{poly_suggestion['degree']}次",
                        "代码": poly_suggestion["code"],
                        "理由": poly_suggestion["reason"],
                        "对应图表": plot_filename
                    })

            # 保存所有建议到 CSV
            if all_suggestions:
                suggestion_df = pd.DataFrame(all_suggestions)
                suggestion_df.to_csv("output/feature_engineering_suggestions.csv", index=False, encoding='utf-8-sig')
                print(f"\n✅ 所有特征工程建议已保存至: output/feature_engineering_suggestions.csv")

            # 打印使用指南（中文保留，不影响绘图）
            print("\n📘 SHAP 依赖图解读与建议指南:")
            print("  - ↗ 直线上升 → 线性正相关 → 无需变换")
            print("  - ↘ 直线下降 → 线性负相关 → 无需变换")
            print("  - ⌣ U型曲线 → 非线性 → 可尝试加平方项")
            print("  - ⌢ 倒U型 → 边际效应递减 → 可尝试 log 或分箱")
            print("  - ▂▃▅▇ 阶梯状 → 分段效应 → 建议分箱")
            print("  - ➰ 右偏分布 + 边际递减 → 建议 log(x)")
            print("  - 💡 上述建议已输出到控制台和 CSV 文件")

        except Exception as e:
            print(f"⚠️ SHAP 依赖图或建议生成失败: {e}")

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
    # 🆕 保存实际树数量
    pd.Series([actual_n_estimators]).to_csv('output/actual_n_estimators.csv', index=False, header=['n_estimators'])
    pd.Series(x_train.columns).to_csv('output/train_feature_names.csv', index=False, header=['特征名'], encoding='utf-8-sig')
    pd.Series(category_feature).to_csv('output/category_features.csv', index=False, header=['特征名'], encoding='utf-8-sig')
    pd.Series(continuous_feature).to_csv('output/continuous_features.csv', index=False, header=['特征名'], encoding='utf-8-sig')
    
    print("✅ 模型和元数据已保存，可用于 API 服务")

    submission.to_csv('output/submission_gbdt_lr.csv', index=False, encoding='utf-8-sig')
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
    print("   - shap_dependence_*.png")
    print("   - feature_engineering_suggestions.csv ← 🆕 新增！特征工程智能建议")
    print("   - submission_gbdt_lr.csv")
    print("   - 控制台输出叶子节点规则解析")
