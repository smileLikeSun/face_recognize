## recognize_face_from_camera.py

### def computer_128_distance(array_1, array_2)

这里直接使用的是欧式距离, 其它也有用余弦相似度来做的,欧式距离公式:

$$\sqrt{\sum_{i=0}^n (x_i - y_i)^2} $$

! [欧式距离公式](https://github.com/smileLikeSun/face_recognize/blob/master/improve_dlib_face_recognize/oushi_distance.png)

余弦相似度:余弦距离也称余弦相似度.空间向量中两个向量夹角的余弦值,作为两个个体间差异的大小的度量,相比欧式距离,更加注重两个向量在方向上的差异!
