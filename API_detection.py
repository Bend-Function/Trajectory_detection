# 请自行准备模型
# 官方模型地址：https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
# Bibili space：https://space.bilibili.com/275177832
# 视频演示：https://www.bilibili.com/video/av35255202

import numpy as np
import os
import sys
import tensorflow as tf
import cv2
import time
import math

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

start = time.time()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cv2.setUseOptimized(True)  # 加速cv

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# 可能要改的内容
######################################################
PATH_TO_CKPT = 'model\\faster_rcnn_inception_v2_coco.pb'  # 模型及标签地址
PATH_TO_LABELS = 'model\\mscoco_label_map.pbtxt'

video_PATH = "video\\test.mp4"  # 要检测的视频
out_PATH = "out_v\\out_test_vic2222.mp4"  # 输出地址

NUM_CLASSES = 90  # 检测对象个数

confident = 0.5         # 置信度
img = 0                 # 初始帧，现已弃用
fourcc = cv2.VideoWriter_fourcc(*'MPEG')    # 编码格式

yun = 200        # 阈值，大于此像素的两点不画线
######################################################

l1 = []                     # 前一帧的坐标信息
l2 = []                     # 当前帧的坐标



connect = []        # 要连的线段数组（添加的是一维数组）所以是二维的


# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# 读取视频
video_cap = cv2.VideoCapture(video_PATH)
fps = int(video_cap.get(cv2.CAP_PROP_FPS))

width = int(video_cap.get(3))
hight = int(video_cap.get(4))
videoWriter = cv2.VideoWriter(out_PATH, fourcc, fps, (1920, 1080))

def load_to_video(cut):         #写视频
    videoWriter.write(cut)




def con(image):                 # 画线与点
    for i in range(0, len(connect)):
        cv2.line(image, (connect[i][0], connect[i][1]), (connect[i][2], connect[i][3]), (0, 255, 0), 3)
        cv2.circle(image, (connect[i][0], connect[i][1]), 3, (0, 0, 255), 2)

    return image


def cal():
    global img,l1,l2
    green = (0, 255, 0)

    def square(x):          # 自制平方函数
        return x * x

    distance = [([0] * len(l2)) for _ in range(0, len(l1))]         # 建立数组以存放相邻两帧点的距离

    for i in range(0, len(l1)):
        for j in range(0, len(l2)):
            # print("i = ", i)
            # print("j = ", j)
            # print(distance)
            distance[i][j] = int(math.sqrt(square(l1[i][1] - l2[j][1]) + square(l1[i][2] - l2[j][2])))      # 计算距离

    for i in range(0, len(l1)):
        if min(distance[i]) < yun:                                                                                     #200 为最小距离阈值，大于此的不画线
            j2 = distance[i].index(min(distance[i]))                                                                    # 找到最小值对应的位置
            # print("cv i =", i)
            # print("cv j =", j)

            # cv2.line(img, (int(l1[i][1]), int(l1[i][2])), (int(l2[j2][1]), int(l2[j2][2])), green, 3)       # 废弃，画线代码
            connect.append([int(l1[i][1]), int(l1[i][2]), int(l2[j2][1]), int(l2[j2][2])])
            # 将两个点的坐标存入数组 格式(a,b),(c,d) 存入后 [[a,b,c,d]]


config = tf.ConfigProto()
config.gpu_options.allow_growth = True          # 要多少显存给多少
with detection_graph.as_default():
    with tf.Session(graph=detection_graph, config=config) as sess:
        # global width,hight
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        num = 0
        while True:
            ret, frame = video_cap.read()
            if ret == False:  # 没检测到就跳出
                break
            num += 1
            print(num)
            image_np = frame
            if num == 1:
                img = frame


            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=4)

            s_boxes = boxes[scores > confident]
            s_classes = classes[scores > confident]         # 读检测信息
            s_scores = scores[scores > confident]


            for i in range(len(s_classes)):
                if s_classes[i] == 8 or s_classes[i] == 3:
                    # 仅仅检测单一class
                    # l数组预留第0位，是给不同类别用的，我还没实现

                    ymin = s_boxes[i][0] * hight  # ymin
                    xmin = s_boxes[i][1] * width  # xmin
                    ymax = s_boxes[i][2] * hight  # ymax
                    xmax = s_boxes[i][3] * width  # xmax

                    center_y = (ymin + ymax) / 2        # 计算中点坐标
                    center_x = (xmin + xmax) / 2
                    # cv2.circle(img, (int(center_x), int(center_y)), 5, (0, 0, 255), 4)      # 画点，废弃

                    l2.append([1, center_x, center_y])  # 坐标写入l2数组（第0位为类别，预留）

            if num != 1:
                print(cal())    # 第一帧无l1，不处理
            l1 = l2         # 把l1当做上一帧

            out_img = con(image_np)

            kkk = cv2.resize(out_img, (1920,1080))     # resize 可选
            # kkk = out_img
            cv2.imshow("v", kkk)                                    # 显示 可选
            load_to_video(kkk)                                         # 写入视频
            cv2.waitKey(1)
            l2 = []                                                      # 清空l2

videoWriter.release()
cv2.imshow("v", img)
cv2.waitKey(0)
end = time.time()
print("Execution Time: ", end - start)
