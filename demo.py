import os
import json
import cv2
import numpy as np
from totrt import ONNX_build_engine
from trt import YoLov5TRT, warmUpThread
import time
from PIL import Image
import cuda 

def init():
    """Initialize model
    Returns: model
    """
    # onnxpath = "yolov5s.onnx"
    # trtpath  = "yolov5s.trt"
    # ONNX_build_engine(onnxpath, trtpath, write_engine=True, batch_size=1, imgsz=512,inputname="images")
    engine_file_path = "yolov5s_engine.trt"
    model = YoLov5TRT(engine_file_path, num_classes=80)
    try:
        for i in range(10):
            # create a new thread to do warm_up
            thread1 = warmUpThread(model)
            thread1.start()
            thread1.join()
    finally:
        # destroy the instance
        model.destroy()
    return model

def process_image(handle=None, input_image=None, args=None, **kwargs):
    vis = True
    batch_size = 1
    categories = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                  "traffic light",
                  "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                  "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
                  "frisbee",
                  "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
                  "surfboard", "tennis racket", "bottle",
                  "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                  "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
                  "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
                  "oven",
                  "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
                  "toothbrush"]

    fake_result = {}
    fake_result["model_data"] = {"objects": []}
    input_image = np.expand_dims(input_image, axis=0)
    # print(input_image.shape)
    batch_image_raw, infer_time, results_batch = handle.infer(input_image, vis=vis)
    for i in range(batch_size):
        results = results_batch[i]
        if results[0] is None:
            fake_result["model_data"]["objects"].append([])
        else:
            top_label = np.array(results[0][:, 6], dtype='int32').tolist()
            top_conf = list(results[0][:, 4] * results[0][:, 5])
            top_boxes = results[0][:, :4].tolist()
            for j, c in list(enumerate(top_label)):
                box = top_boxes[j]
                left, top, right, bottom = box
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(input_image.shape[1], np.floor(bottom).astype('int32'))
                right = min(input_image.shape[2], np.floor(right).astype('int32'))
                # print(left, top, right, bottom)
                fake_result['model_data']['objects'].append({
                    "xmin": int(left),
                    "ymin": int(top),
                    "xmax": int(right),
                    "ymax": int(bottom),
                    "confidence": top_conf[j],
                    "name": categories[int(c)]
                })
            # if vis:
            #     save_name = [str(n) + '.jpg' for n in range(1, batch_size+1)]
            #     print(batch_image_raw[i].shape)
            #     cv2.imwrite(save_name[i], batch_image_raw[i])
            #     # cv2.imshow("image",batch_image_raw[i])
            #     # cv2.waitKey()
    print("推理时间为{}s".format(infer_time))
    out = json.dumps(fake_result, indent=4)
    return out,batch_image_raw[0],infer_time

#原始代码
# if __name__ == '__main__':
#     # Test API
#     img = cv2.imread('images/zidane.jpg')
#     predictor = init()
#     res,img = process_image(predictor, img)

#     # cv2.imshow("image",img)
#     # cv2.waitKey()
#     print(res)


def trt_yolo(predictor,image):
    img = image
    res,img,time = process_image(predictor, img)
    return img,time

def vedio_detection():
    predictor = init()
    capture = cv2.VideoCapture(0)
    ref, frame = capture.read()
    if not ref:
        raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

    fps = 0.0
    while(True):
        # 读取某一帧
        ref, frame = capture.read()
        if not ref:
            break
        frame,time = trt_yolo(predictor, frame)
        # frame = cv2.resize(frame,(1024,1024))
        fps  = ( fps + (1./time) ) / 2
        print("fps= %.2f ms"%(fps))
        frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("video",frame)
        c= cv2.waitKey(1) & 0xff 
        # if video_save_path!="":
        #     out.write(frame)

        if c==27:
            capture.release()
            break

    print("Video Detection Done!")
    capture.release()
    cv2.destroyAllWindows()

def image_detection():
    predictor = init()
    img = cv2.imread('images/zidane.jpg')
    predictor = init()
    res,img,time = process_image(predictor, img)
    # print(res)
    cv2.imshow("image",img)
    cv2.waitKey()



if __name__ == '__main__':
    vedio_detection()
    # image_detection()