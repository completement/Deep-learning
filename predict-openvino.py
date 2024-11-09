import os
import cv2
import numpy as np
from openvino.runtime import Core
import argparse
import time

CLASS_NAMES = {
    0: 'Header',
    1: 'Title',
    2: 'Figure',
    3: 'Text',
    4: 'Foot'
}

class YOLO11:
    def __init__(self, model_path, confidence_thres=0.5, iou_thres=0.45):
        self.model_path = model_path
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.classes = CLASS_NAMES
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

        # 初始化 OpenVINO 模型推理会话
        self.ie = Core()
        self.model = self.ie.read_model(model=self.model_path)
        self.compiled_model = self.ie.compile_model(model=self.model, device_name="CPU")
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

        input_shape = self.model.input(0).shape
        self.input_width = input_shape[3]
        self.input_height = input_shape[2]

    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img, ratio, pad = self.letterbox(image, new_shape=(self.input_width, self.input_height))
        
        image_data = np.array(img) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))
        return np.expand_dims(image_data, axis=0).astype(np.float32), ratio, pad

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(np.floor(dh)), int(np.ceil(dh))
        left, right = int(np.floor(dw)), int(np.ceil(dw))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, r, (dw, dh)
 
    def postprocess(self, image, outputs, ratio, pad):
        outputs = np.transpose(np.squeeze(outputs[0]))
        height, width = image.shape[:2]
        boxes, scores, class_ids = [], [], []

        dw, dh = pad
        for row in outputs:
            classes_scores = row[4:]
            max_score = np.amax(classes_scores)
            if max_score >= self.confidence_thres:
                class_id = np.argmax(classes_scores)
                x, y, w, h = row[:4]

                x -= dw
                y -= dh
                x /= ratio
                y /= ratio
                w /= ratio
                h /= ratio
                left = int(x - w / 2)
                top = int(y - h / 2)

                boxes.append([left, top, int(w), int(h)])
                scores.append(max_score)
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
        for i in indices:
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            self.draw_detections(image, box, score, class_id)
        return image

    def draw_detections(self, img, box, score, class_id):
        x1, y1, w, h = box
        color = self.color_palette[class_id]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, thickness=3)
        label = f"{self.classes[class_id]}: {score:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
        cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED)
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)

    def process_images_in_directory(self, img_directory, output_directory):
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        output_pred_dir = os.path.join(output_directory, 'predictions')
        output_xml_dir = os.path.join(output_directory, 'annotations')

        if not os.path.exists(output_pred_dir):
            os.makedirs(output_pred_dir)
        if not os.path.exists(output_xml_dir):
            os.makedirs(output_xml_dir)
            
        img_files = sorted([f for f in os.listdir(img_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

        for img_file in img_files:
            image_path = os.path.join(img_directory, img_file)
            image = cv2.imread(image_path)
            
            img_data, ratio, pad = self.preprocess(image)
            start_time = time.time()
            outputs = self.compiled_model([img_data])[self.output_layer]
            inference_time = (time.time() - start_time) * 1000  

            output_image = self.postprocess(image, outputs, ratio, pad)
            output_filename = os.path.join(output_pred_dir, f'pred_{img_file}')
            cv2.imwrite(output_filename, output_image)

            # 保存 XML 文件
            self.save_results_xml_annotated(image, img_file, outputs, output_xml_dir, ratio, pad)
            print(f"处理 {img_file} 完成，推理时间: {inference_time:.4f} ms")

    def save_results_xml_annotated(self, image, img_file, outputs, output_directory, ratio, pad):
        xml_file_path = os.path.join(output_directory, f'{os.path.splitext(img_file)[0]}.xml')
        outputs = np.transpose(np.squeeze(outputs[0]))

        boxes, scores, class_ids = [], [], []
        for row in outputs:
            classes_scores = row[4:]
            max_score = np.amax(classes_scores)
            if max_score >= self.confidence_thres:
                class_id = np.argmax(classes_scores)
                x, y, w, h = row[:4]

                # 反向转换坐标
                x -= pad[0]
                y -= pad[1]
                x /= ratio
                y /= ratio
                w /= ratio
                h /= ratio

                xmin = (x - w / 2)
                ymin = (y - h / 2)
                xmax = (x + w / 2)
                ymax = (y + h / 2)

                boxes.append([xmin, ymin, xmax, ymax])
                scores.append(max_score)
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        with open(xml_file_path, 'w') as xml_file:
            xml_file.write('<annotations>\n')
            for i in indices:
                box = boxes[i]
                class_id = class_ids[i]
                xmin, ymin, xmax, ymax = box

                xml_file.write(f"  <object>\n")
                xml_file.write(f"    <name>{self.classes[class_id]}</name>\n")
                xml_file.write(f"    <bndbox>\n")
                xml_file.write(f"      <xmin>{xmin}</xmin>\n")
                xml_file.write(f"      <ymin>{ymin}</ymin>\n")
                xml_file.write(f"      <xmax>{xmax}</xmax>\n")
                xml_file.write(f"      <ymax>{ymax}</ymax>\n")
                xml_file.write(f"    </bndbox>\n")
                xml_file.write(f"  </object>\n")
            xml_file.write('</annotations>\n')

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO11 Prediction Script')
    parser.add_argument('-dir', required=True, type=str, help='Directory of input images')
    parser.add_argument('-model', required=True, type=str, help='Path to the OpenVINO model file')
    parser.add_argument('-out', required=True, type=str, help='Output directory for predictions')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    model_path = os.path.join(args.model, 'best.xml')
    detection = YOLO11(model_path)
    detection.process_images_in_directory(args.dir, args.out)
    print(f"处理完成，输出保存在 {args.out} 目录中。")