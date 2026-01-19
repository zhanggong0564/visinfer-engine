'''
@Author       : gongzhang4
@Date         : 2026-01-17 06:47:42
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-17 06:51:24
@FilePath     : LineSqueezeDemo.py
@Description  :
'''

import sys

sys.path.append('../')
import cv2
from services import LineSqueezeRecognition
from pathlib import Path


if __name__ == '__main__':
    onnx_path = './weights/LineSqueeze_v3.onnx'
    ocr_model_dir = './weights/official_models/PP-en_rec_ppocr_v5'
    line_squeeze_recognition = LineSqueezeRecognition(onnx_path, ocr_model_dir)

    all_image_info = Path(
        '/data/zhanggong/workspace/project/move_vsion/LineSequence_identification/src/datas/1023_debug'
    ).glob('*.jpg')
    for image_info in all_image_info:
        image_info = Path(
            '/data/zhanggong/workspace/project/move_vsion/LineSequence_identification/src/demo/17636431248781991489766612680704.jpg'
        )
        inputs_info = {
            "image": cv2.imread(str(image_info)),
            "types": "五路有熔丝盒有磁环",
        }
        print(image_info)
        import time

        start = time.time()
        for i in range(1):
            results = line_squeeze_recognition.verify_line_sequence(**inputs_info)
        end = time.time()
        print(f'cost time: {(end - start) / 10} s')
        # print(results)
        # 可视化结果,status_all写在图像的右上角，scene和state写在框的左上角
        status_all = results['status']
        vis_image = inputs_info['image'].copy()
        h, w, _ = vis_image.shape
        print(status_all)
        cv2.putText(vis_image, f'status: {status_all}', (w - 500, 500), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        for info in results['detailList']:
            scene = info['scene']
            state = info['status']
            coordinate = info['coordinate']
            if len(coordinate) == 0:
                continue
            x1, y1, x2, y2 = (int(coord * dim) for coord, dim in zip(coordinate, [w, h, w, h]))
            cv2.rectangle(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                vis_image, f'{scene}:{state}', (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
        new_image_name = image_info.name.replace('.jpg', '_vis.jpg')
        result_path = image_info.parent.parent / 'vis'
        result_path.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(result_path / new_image_name), vis_image)
        print(f"save vis image to {result_path / new_image_name}")
        break
