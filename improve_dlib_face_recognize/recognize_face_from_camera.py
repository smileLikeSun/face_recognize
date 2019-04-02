
import dlib
import pandas as pd
import numpy as np
import cv2
import json

def read_face_feature_from_csv():
    all_face_feature_csv_path = './data/total.csv'
    # 读取存放所有人脸特征均值的 csv 文件
    # header=None 参数必须加，否则会把第一行当做标题，会减少一个人脸特征均值的计入
    read_csv = pd.read_csv(all_face_feature_csv_path, header=None)
    # 录入人脸特征的数组
    face_feature_csv = []
    # 读取已知人脸的特征数据
    for i in range(read_csv.shape[0]):
        someone_face_feature = []
        for j in range(0, len(read_csv.ix[i, :])):
            someone_face_feature.append(read_csv.ix[i, :][j])
        face_feature_csv.append(someone_face_feature)
    print('已知人脸数据读取完成！')
    return face_feature_csv

# 计算两个 128D 向量间的欧式距离
def compute_128D_distance(face_feature_1, face_feature_2):
    face_feature_1 = np.array(face_feature_1)
    face_feature_2 = np.array(face_feature_2)
    distance = np.sqrt(np.sum(np.square(face_feature_1 - face_feature_2)))
    if distance > 0.4:
        res = 'different'
    else:
        res = 'same'
    return res


def recognize_face_from_camera():
    name_path = './data/name_dict.json'
    recognize_face_tool_path = 'C:/Users/xsmile/AppData/Local/Programs/' \
                               'Python/Python36/Lib/site-packages/dlib/' \
                               'dlib_face_recognition_resnet_model_v1.dat'
    recognize_face = dlib.face_recognition_model_v1(recognize_face_tool_path)
    # 人脸检测器
    detector = dlib.get_frontal_face_detector()
    # 人脸预测器
    predictor_tool_path = 'C:/Users/xsmile/AppData/Local/Programs/' \
                          'Python/Python36/Lib/site-packages/dlib/' \
                          'shape_predictor_68_face_landmarks.dat'
    predictor =dlib.shape_predictor(predictor_tool_path)
    # 所有人脸特征值
    all_face_feature_mean = read_face_feature_from_csv()
    cap = cv2.VideoCapture(0)
    cap.set(3, 600)
    while cap.isOpened():
        flag, img_frame = cap.read()
        # 转灰度
        img_gray = cv2.cvtColor(img_frame,cv2.COLOR_BGR2GRAY)
        # 识别镜头中的人脸数
        faces = detector(img_gray, 0)
        # 标注的字体
        font = cv2.FONT_HERSHEY_COMPLEX
        # 镜头中人脸的坐标
        face_name_position = []
        # 镜头中人脸名称
        face_name = []
        # 读取从键盘中的按键
        press_key = cv2.waitKey(1)
        if press_key != ord('q'):
            if len(faces):
                # 存放捕捉镜头中的人脸特征值
                capture_faces_feature = []
                for i in range(len(faces)):
                    shape = predictor(img_frame, faces[i])
                    capture_faces_feature.append(recognize_face.compute_face_descriptor(img_frame, shape))
                # 遍历捕捉到的人脸, 设置名字的位子坐标
                for i in range(len(faces)):
                    face_name.append('Unknow')
                    face_name_position.append(tuple([faces[i].left(), int(faces[i].bottom() + (faces[i].bottom() - faces[i].top())/4)]))
                    # 针对每张人脸遍历所有特征进行对比
                    for j in range(len(all_face_feature_mean)):
                        # 某个人脸与所有特征均值进行对比
                        compare = compute_128D_distance(capture_faces_feature[i], all_face_feature_mean[j])
                        if compare == 'same':
                            with open(name_path, 'r') as fi:
                                name_dict = json.loads(fi.read())
                                face_name[i] = name_dict[str(j)]
                            # if j == 0:
                            #     face_name[i] = 'person_1'
                            # elif j == 1:
                            #     face_name[i] = 'nanM'
            for k, position in enumerate(faces):
                # 矩形框的颜色
                rectangle_color = (255, 0, 0)
                cv2.rectangle(img_frame, (position.left(), position.top()), (position.right(), position.bottom()),
                              rectangle_color, 2)
            for i in range(len(faces)):
                cv2.putText(img_frame, face_name[i], face_name_position[i], font, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        else:
            print('退出识别！')
            break

        cv2.putText(img_frame, 'Face Recognize', (10, 20), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img_frame, 'Q: Quit', (10, 450), font, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow('face_recognize', img_frame)
    cap.release()
    cv2.destroyAllWindows()

recognize_face_from_camera()
