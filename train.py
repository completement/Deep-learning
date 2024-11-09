#coding:utf-8
from ultralytics import YOLO
# 模型配置文件
model_yaml_path = "ultralytics/cfg/models/11/yolo11.yaml"
#数据集配置文件
data_yaml_path = '/root/autodl-tmp/datasets/data.yaml'
#预训练模型
pre_model_name = 'yolo11n.pt'

if __name__ == '__main__':
    #加载预训练模型
    model = YOLO(model_yaml_path).load(pre_model_name)
    #训练模型
    results = model.train(data=data_yaml_path,
                          epochs=200,
                          batch=128,
                          patience=100,
                          name='train_v11')
