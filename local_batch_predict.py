# local_batch_predict.py
import pandas as pd
import argparse
import logging
from pathlib import Path
from predictor import load_models, preprocess_single_sample, predict_core

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="本地批量预测（与 app.py 逻辑一致）")
    parser.add_argument("input_csv", help="输入的 CSV 文件路径")
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    if not input_path.exists() or input_path.suffix.lower() != '.csv':
        logger.error("❌ 请输入有效的 .csv 文件路径")
        return

    try:
        # 兼容 UTF-8 和 GBK 编码
        try:
            input_df = pd.read_csv(input_path, encoding='utf-8')
        except UnicodeDecodeError:
            logger.warning("⚠️ UTF-8 解码失败，尝试使用 GBK 编码...")
            input_df = pd.read_csv(input_path, encoding='gbk')
            
        if input_df.empty:
            logger.warning("输入 CSV 为空")
            return

        logger.info(f"加载模型并处理 {len(input_df)} 行...")
        models = load_models()

        # 预处理
        processed_rows = []
        for _, row in input_df.iterrows():
            processed = preprocess_single_sample(
                row.to_dict(),
                models['continuous_features'],
                models['category_features'],
                models['train_feature_names']
            )
            processed_rows.append(processed)

        # 预测（与 app.py 的 /predict_batch_csv 一致：return_explanation=True, generate_plot=False）
        results = predict_core(processed_rows, models, return_explanation=True, generate_plot=False)

        # 构造 CSV 结果（完全复刻 app.py）
        csv_results = []
        for r in results:
            exp = r["explanation"]
            top_features = "; ".join([f"{feat['feature']}({feat['shap_value']:+.3f})" for feat in exp["important_features"]]) if exp["important_features"] else ""
            top_rules = "; ".join(exp["top_rules"][:3]) if exp["top_rules"] else ""
            csv_results.append({"probability": r["probability"], "top_features": top_features, "top_rules": top_rules})

        # 生成结果 DataFrame
        original_id_col_name = input_df.columns[0]
        result_df = pd.DataFrame()
        result_df['prediction_probability'] = [r['probability'] for r in csv_results]
        result_df[original_id_col_name] = input_df.iloc[:, 0].values
        for col in input_df.columns[1:]:
            result_df[col] = input_df[col].values
        result_df['top_3_features'] = [r['top_features'] for r in csv_results]
        result_df['top_3_rules'] = [r['top_rules'] for r in csv_results]

        # 保存
        output_path = input_path.parent / f"predicted_{input_path.name}"
        result_df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"✅ 结果已保存: {output_path}")

    except Exception as e:
        logger.exception(f"预测失败: {e}")

if __name__ == '__main__':
    main()
