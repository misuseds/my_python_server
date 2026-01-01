# 启动服务（可选地加载已有模型）
python delete_cnn.py --model-path models/improved_dxf_entity_cnn_model.pth

# 加载模型
curl "http://localhost:5000/load_model?model_path=models/improved_dxf_entity_cnn_model.pth"

# 训练新模型
curl "http://localhost:5000/train?json_file=output/combined_labels.json&image_dir=output/pictures&model_save_path=models/improved_dxf_entity_cnn_model.pth&num_epochs=50"

# 评估模型
curl "http://localhost:5000/evaluate?json_file=output/combined_labels.json&image_dir=output/pictures"

# 单图预测
curl "http://localhost:5000/predict?image_path=path/to/image.png"

# 批量预测
curl "http://localhost:5000/predict_batch?json_file=output/combined_labels.json&image_dir=output/pictures"

仓库gitee misuseds/cad_ppo