# app.py
import pandas as pd
import tempfile
import os
import uuid
import logging
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from predictor import load_models, preprocess_single_sample, predict_core

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
models = load_models()  # 启动时加载一次


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not isinstance(data, dict):
            return jsonify({"error": "JSON body must be an object"}), 400
        sample_df = preprocess_single_sample(
            data,
            models['continuous_features'],
            models['category_features'],
            models['train_feature_names']
        )
        results = predict_core([sample_df], models, return_explanation=True, generate_plot=True)
        return jsonify(results[0])
    except Exception as e:
        logger.exception("单样本预测失败")
        return jsonify({"error": str(e)}), 500


@app.route('/predict_batch_csv', methods=['POST'])
def predict_batch_csv():
    try:
        if 'file' not in request.files or not request.files['file'].filename.endswith('.csv'):
            return jsonify({"error": "Invalid file"}), 400

        file = request.files['file']
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        # 尝试读取 CSV，兼容 UTF-8 和 GBK
        try:
            input_df = pd.read_csv(tmp_path, encoding='utf-8')
        except UnicodeDecodeError:
            logger.warning("UTF-8 解码失败，尝试使用 GBK 编码...")
            input_df = pd.read_csv(tmp_path, encoding='gbk')
        os.unlink(tmp_path)
        if input_df.empty:
            return jsonify({"error": "Empty CSV"}), 400

        processed_rows = []
        for _, row in input_df.iterrows():
            processed = preprocess_single_sample(
                row.to_dict(),
                models['continuous_features'],
                models['category_features'],
                models['train_feature_names']
            )
            processed_rows.append(processed)

        results = predict_core(processed_rows, models, return_explanation=True, generate_plot=False)

        # 构造结果（与 local_batch_predict.py 完全一致）
        csv_results = []
        for r in results:
            exp = r["explanation"]
            top_features = "; ".join([f"{feat['feature']}({feat['shap_value']:+.3f})" for feat in exp["important_features"]]) if exp["important_features"] else ""
            top_rules = "; ".join(exp["top_rules"][:3]) if exp["top_rules"] else ""
            csv_results.append({"probability": r["probability"], "top_features": top_features, "top_rules": top_rules})

        original_id_col_name = input_df.columns[0]
        result_df = pd.DataFrame()
        result_df['prediction_probability'] = [r['probability'] for r in csv_results]
        result_df[original_id_col_name] = input_df.iloc[:, 0].values
        for col in input_df.columns[1:]:
            result_df[col] = input_df[col].values
        result_df['top_3_features'] = [r['top_features'] for r in csv_results]
        result_df['top_3_rules'] = [r['top_rules'] for r in csv_results]

        output_filename = f"predictions_{uuid.uuid4().hex[:8]}.csv"
        output_path = os.path.join(tempfile.gettempdir(), output_filename)
        result_df.to_csv(output_path, index=False, encoding='utf-8-sig')

        response = send_file(output_path, mimetype='text/csv', as_attachment=True, download_name=output_filename)

        @response.call_on_close
        def remove_file():
            try:
                os.remove(output_path)
            except Exception as e:
                logger.error(f"临时文件删除失败: {e}")
        return response

    except Exception as e:
        logger.exception("批量预测失败")
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "features": {
            "continuous": len(models['continuous_features']),
            "categorical": len(models['category_features']),
            "total": len(models['train_feature_names'])
        },
        "n_estimators": int(models['actual_n_estimators'])
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
