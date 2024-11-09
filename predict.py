import os
import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv11 Prediction Script')
    parser.add_argument('-dir', required=True, type=str, help='Directory of input images')
    parser.add_argument('-model', required=True, type=str, help='Path to the YOLO model file')
    parser.add_argument('-out', required=True, type=str, help='Output directory for predictions')
    return parser.parse_args()

def main(args):
    # 创建输出目录
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    # 加载YOLO模型
    model = YOLO(args.model)

    # 进行推理并保存结果
    results = model.predict(source=args.dir, save=True, save_txt=True, conf=0.5, iou=0.45)

    # 遍历推理结果，保存为XML文件
    for idx, result in enumerate(results):
        # 获取输入图片的路径
        image_path = result.path  #直接访问 path 属性
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        xml_file = os.path.join(args.out, f'{image_name}.xml')

        # 创建XML内容
        with open(xml_file, 'w') as f:
            f.write('<annotations>\n')
            for det in result.boxes:  #boxes是包含检测结果的对象
                # 确保 det 提供所需的属性
                class_id = det.cls.item() if det.cls.numel() > 0 else None  # 获取类 ID
                bbox = det.xyxy[0]  # 获取边界框坐标
                if class_id is not None:
                    f.write(f"  <object>\n")
                    f.write(f"    <name>{result.names[class_id]}</name>\n")  # 访问 class 名称
                    f.write(f"    <bndbox>\n")
                    f.write(f"      <xmin>{bbox[0]}</xmin>\n")
                    f.write(f"      <ymin>{bbox[1]}</ymin>\n")
                    f.write(f"      <xmax>{bbox[2]}</xmax>\n")
                    f.write(f"      <ymax>{bbox[3]}</ymax>\n")
                    f.write(f"    </bndbox>\n")
                    f.write(f"  </object>\n")
            f.write('</annotations>\n')

if __name__ == '__main__':
    args = parse_args()
    main(args)
