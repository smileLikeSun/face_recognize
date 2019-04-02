import dlib
import cv2
import os
from skimage import io
import csv
import pandas as pd
import numpy as np
import json

def make_dir_for_img():
    img_path = './data/img_from_camera'
    # 生成需要存储文件的文件夹
    if not os.path.exists(img_path):
        # 使用 mkdir() 会出错，可能是 mkdir() 不能创建新文件夹下面的文件夹
        os.makedirs(img_path)
    your_id = int(get_person_id()) + 1
    your_face_path = img_path + '/person_' + str(your_id)
    os.mkdir(your_face_path)
    print('个人人脸文件夹建立成功！')
    return your_face_path


def get_person_id():
    img_path = 'data/img_from_camera'
    # 是否已有人脸图像，并获取最后一个图像 id 值
    if os.listdir(img_path):
        img_list = os.listdir(img_path)
        img_list.sort()
        person_id = str(img_list[-1]).split('_')[-1]
    else:
        person_id = 0
    return person_id

def use_camera_save_img():
    cap = cv2.VideoCapture(0)
    cap.set(3, 600)
    # 建立个人人脸文件夹标志，防止多次建立
    make_dir_for_img_flag = False
    # 按 N 键，创建个人人脸文件夹
    your_face_path = ''
    # 存储图像个数
    face_count = 0
    # 图像需要标注的字体
    font = cv2.FONT_HERSHEY_COMPLEX
    while cap.isOpened():
        ret, img_frame = cap.read()
        img_gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
        # 人脸数
        faces = detector(img_gray, 1)
        # 标出人脸特征点
        for face in faces:
            shape = predictor(img_frame, face)
            for pt in shape.parts():
                pt_pos = (pt.x, pt.y)
                cv2.circle(img_frame, pt_pos, 2, (0, 255, 0), 1)
        face_pos = {}
        for k, position in enumerate(faces):
            face_pos['top'] = position.top() - 20
            face_pos['bottom'] = position.bottom() +20
            face_pos['left'] = position.left() - 20
            face_pos['right'] = position.right() +20
            # print(face_pos) {'top': 118, 'bottom': 341, 'left': 291, 'right': 513}
            # 计算矩形的大小
            # half_height = int((d.bottom() - d.top()) / 5) # 一定要加 int ，不然下面用的时候会报错
            # half_width = int((d.right() - d.left()) / 5)
            # 矩形框的颜色
            rectangle_color = (255, 0, 0)
            cv2.rectangle(img_frame, (position.left(), position.top()), (position.right(), position.bottom()), rectangle_color, 2)

        press_key = cv2.waitKey(1)
        if press_key == ord('n'):
            if not make_dir_for_img_flag:
                your_face_path = make_dir_for_img()
                make_dir_for_img_flag = True

        if press_key == ord('s'):
            if len(faces) == 0:
                print('未检测到人脸！')
                continue
            if make_dir_for_img_flag:
                if face_count < 10:
                    # for (x, y, w, h) in faces:
                    #     cv2.rectangle(img_gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.imwrite(your_face_path + '/face_{}.jpg'.format(str(face_count)),
                                img_gray[face_pos['top']:face_pos['bottom'], face_pos['left']:face_pos['right']])
                    face_count += 1
                    print('{} 张图像存储成功！'.format(face_count))
                else:
                    print('最多存储 10 张图像！')
            else:
                print('请先按 N 键，建立个人图像文件夹！')
        if press_key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            print('退出成功！')
        cv2.putText(img_frame, 'Face Register', (10, 20), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img_frame, 'N: New face folder', (10, 50), font, 0.6, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img_frame, 'S: Save current face', (10, 420), font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_frame, 'Q: Quit', (10, 450), font, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
        # 设置窗口大小可调，非等比例扩大或缩小，第一个参数要和 imshow 方法的第一个参数相同
        # cv2.namedWindow('registe your face', 0)
        # imshow 方法一定要放在最后，不然字体渲染或者矩形框等无法显示
        cv2.imshow('registe your face', img_frame)

# 正向人脸检测
detector = dlib.get_frontal_face_detector()
# 人脸预测
predictor_tool_path = 'C:/Users/xsmile/AppData/Local/Programs/' \
                          'Python/Python36/Lib/site-packages/dlib/' \
                          'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_tool_path)
# 人脸识别训练模型
recognize_tool_path = 'C:/Users/xsmile/AppData/Local/Programs/' \
                          'Python/Python36/Lib/site-packages/dlib/' \
                          'dlib_face_recognition_resnet_model_v1.dat'
recogniza_face = dlib.face_recognition_model_v1(recognize_tool_path)

# 创建存储 csv 文件的文件夹
def make_csv_dir():
    csv_dir = './data/csv_from_img'
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

# 返回图像特征值
def return_face_feature(img_path):
    img = io.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector(img_gray, 1)
    # 对截取的人脸进行检测，也有可能检测不出人脸了
    if len(faces) != 0:
        shape = predictor(img_gray, faces[0])
        face_descriptor = recogniza_face.compute_face_descriptor(img_gray, shape)
    else:
        face_descriptor = 0
        print('检测人脸失败!')
    return face_descriptor


# 提取照片特征值，写入 csv
# person_face_path: 某人图像文件夹，eg：person_1
# write_csv_from_person_img: 某人图像特征值存储的 csv 文件
def write_feature_into_csv(person_face_path, write_csv_from_person_img):
    person_imgs = os.listdir(person_face_path)
    with open(write_csv_from_person_img, 'w') as csv_file:
        writer = csv.writer(csv_file)
        if person_imgs:
            for i in range(len(person_imgs)):
                face_feature = return_face_feature(person_face_path + '/' + person_imgs[i])
                if face_feature:
                    writer.writerow(face_feature)
                    print('读取 {} 图像特征成功！'.format(person_face_path + '/' + person_imgs[i]))
        else:
            print('{} 文件夹为空！'.format(person_face_path))
            writer.writerow('')


# 读取某人图像，提取特征值写入 csv
def read_persons_imgs_into_csv():
    img_from_camera = './data/img_from_camera/'
    csv_from_img = './data/csv_from_img/'
    persons = os.listdir(img_from_camera)
    persons.sort()
    person = persons[len(persons) - 1]
    with open(csv_from_img + person + '.csv', 'w'):
        pass
    write_feature_into_csv(img_from_camera + person, csv_from_img + person + '.csv')
    print('{} 图像特征写入成功！'.format(person))

# 读取 csv 数据，计算特征值均值
def compute_feature_mean(csv_from_img):
    column_name = []
    for i in range(128):
        column_name.append('feature_' + str(i + 1))
    read_csv = pd.read_csv(csv_from_img, names=column_name)
    # 存放 128 特征均值
    feature_means = []
    if read_csv.size != 0:
        for i in range(128):
            temp_arr = read_csv['feature_' + str(i + 1)]
            temp_arr = np.array(temp_arr)
            temp_mean = np.mean(temp_arr)
            feature_means.append(temp_mean)
    return feature_means

# 所有的特征均值写入到一个 csv 文件
def all_feature_mean_into_csv():
    total_csv = './data/total.csv'
    feature_csv_path = './data/csv_from_img'
    with open(total_csv, 'a') as csv_file:
        writer = csv.writer(csv_file)
        csv_list = os.listdir(feature_csv_path)
        csv_name = csv_list[len(csv_list) - 1]
        feature_mean = compute_feature_mean(feature_csv_path + '/' + csv_name)
        writer.writerow(feature_mean)
        print('{} 写入成功！'.format(feature_csv_path + '/' + csv_name))

def write_name_to_json(nick_name):
    name_path = './data/name_dict.json'
    # if not os.path.exists(name_path):
    #     os.makedirs(name_path)
    if not os.path.exists(name_path):
        with open(name_path, 'w') as fi:
            # dump 将 python 对象写入 json
            json.dump({}, fi)
    with open(name_path, 'r') as fi:
        # loads 将字符串转换为 python 对象
        name_dict = json.loads(fi.read())
        name_dict[str(len(name_dict))] = nick_name
    with open(name_path, 'w') as fi:
        json.dump(name_dict, fi)

use_camera_save_img()
nick_name = input('input your English nick name: ')
write_name_to_json(nick_name)
print('开始计算人脸特征值！')
make_csv_dir()
print('csv 存储文件夹创建成功！')
read_persons_imgs_into_csv()
all_feature_mean_into_csv()
print('人脸特征均值写入成功！')