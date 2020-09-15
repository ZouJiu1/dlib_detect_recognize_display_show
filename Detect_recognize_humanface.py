import dlib
import numpy as np
from copy import deepcopy
import cv2

#http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat
#http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

img_1 = r'1.jpg'
img_2 = r'2.jpg'
img_3 = r'3.jpg'

class d_r_h(object):
    def __init__(self, img_1, img_2, img_3):
        super(d_r_h, self).__init__()
        self.img_1 = img_1
        self.img_2 = img_2
        self.img_3 = img_3
        self.detector = dlib.get_frontal_face_detector()
        self.img_size = 150
        self.predictor = dlib.shape_predictor(r'shape_predictor_68_face_landmarks.dat')
        self.recognition = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

    def point_draw(self, img, sp, title, save):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for i in range(68):
            cv2.putText(img, str(i), (sp.part(i).x, sp.part(i).y), cv2.FONT_HERSHEY_DUPLEX, 0.3, (0, 0, 255), 1,
                        cv2.LINE_AA)
            # cv2.drawKeypoints(img, (sp.part(i).x, sp.part(i).y),img, [0, 0, 255])
        if save:
            cv2.imwrite(title+str(np.random.randint(100))+'.jpg', img)
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyWindow(title)

    def show_origin(self, img):
        cv2.imshow('origin', img)
        cv2.waitKey(0)
        cv2.destroyWindow('origin')

    def detection(self, img):
        image = dlib.load_rgb_image(img)
        # 人脸对齐、切图
        dets = self.detector(image, 1)
        if len(dets) == 1:
            # faces = dlib.full_object_detections()
            shape = self.predictor(image, dets[0])
            print("Computing descriptor on aligned image ..")
            images = dlib.get_face_chip(image, shape, size=self.img_size)

            self.point_draw(image, shape, 'before image warping', save=True)
            shapeimage = np.array(images).astype(np.uint8)
            dets = self.detector(shapeimage, 1)
            if len(dets) == 1:
                point68 = self.predictor(shapeimage, dets[0])
                self.point_draw(shapeimage, point68, 'after image warping', save=True)

            #计算128维特征向量
            face_descriptor_from_prealigned_image = self.recognition.compute_face_descriptor(images)
        return face_descriptor_from_prealigned_image

    def compare(self):
        vec1 = np.array(self.detection(self.img_1))
        vec2 = np.array(self.detection(self.img_2))
        vec3 = np.array(self.detection(self.img_3))
        same_people = np.sqrt(np.sum((vec2-vec3)*(vec2-vec3)))
        not_same_people12 = np.sqrt(np.sum((vec1-vec2)*(vec1-vec2)))
        not_same_people13 = np.sqrt(np.sum((vec1-vec3)*(vec1-vec3)))
        print('distance between different people12:{:.3f}, different people13:{:.3f}, same people:{:.3f}'.\
              format(not_same_people12, not_same_people13, same_people))

detection_recognition = d_r_h(img_1, img_2, img_3)
detection_recognition.compare()