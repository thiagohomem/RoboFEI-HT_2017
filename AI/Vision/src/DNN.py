import argparse
import time
import cv2

# import the necessary packages
import imutils

import os
#import tarfile
import time
#import zipfile
from classify import *
import os
import cv2
import numpy as np
import sys

from servo import Servo

import Condensation

SERVO_PAN = 19
SERVO_TILT = 20

SERVO_TILT_VALUE = 705 # Posicao central inicial Tilt
SERVO_PAN_VALUE = 512 # Posicao central inicial Tilt










servo = None
servo = Servo(SERVO_PAN_VALUE, SERVO_TILT_VALUE)


class objectDetect():
    CountLostFrame = 0
    Count = 0
    kernel_perto = np.ones((39, 39), np.uint8)
    kernel_perto2 = np.ones((100, 100), np.uint8)
    kernel_medio = np.ones((22, 22), np.uint8)
    kernel_medio2 = np.ones((80, 80), np.uint8)
    kernel_longe = np.ones((12, 12), np.uint8)
    kernel_longe2 = np.ones((40, 40), np.uint8)
    kernel_muito_longe = np.ones((7, 7), np.uint8)
    kernel_muito_longe2 = np.ones((30, 30), np.uint8)
    mean_file = None
    labels = None
    net = None
    transformer = None
    status =1

    def __init__(self, net, transformer, mean_file, labels):
        self.mean_file = mean_file
        self.labels = labels
        self.net = net
        self.transformer = transformer

    def searchball(self, image, visionMask, visionMorph1, visionMorph2, visionMorph3):

        YUV_frame = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        white_mask = cv2.inRange(YUV_frame[:,:,0], 200, 255)

        if visionMask:
            cv2.imshow('Frame Mascara', white_mask)

#        start2 = time.time()
        BallFound = False
        frame, x, y, raio, maskM = Morphology(self,image , white_mask,self.kernel_perto, self.kernel_perto2,1)
        if visionMorph1:
            cv2.imshow('Morfologia 1', maskM)
#        print "Search = ", time.time() - start2 
        if (x==0 and y==0 and raio==0):
            frame, x, y, raio, maskM = Morphology(self,image, white_mask,self.kernel_medio ,self.kernel_medio2,2)
            if visionMorph2:
                cv2.imshow('Morfologia 2', maskM)
            if (x==0 and y==0 and raio==0):
                frame, x, y, raio, maskM = Morphology(self,image, white_mask,self.kernel_longe , self.kernel_longe2,3)
                if visionMorph3:
                    cv2.imshow('Morfologia 3', maskM)
                if (x==0 and y==0 and raio==0):
                    frame, x, y, raio, maskM = Morphology(self,image, white_mask,self.kernel_muito_longe, self.kernel_muito_longe2,4)
                    if (x==0 and y==0 and raio==0):
                        self.CountLostFrame +=1
                        print("@@@@@@@@@@@@@@@@@@@",self.CountLostFrame)
                        if self.CountLostFrame==10:
                            BallFound = False
                            self.CountLostFrame = 0
                            print("----------------------------------------------------------------------")
                            print("----------------------------------------------------------------------")
                            print("----------------------------------------------------------------------")
                            print("--------------------------------------------------------Ball not found")
                            print("----------------------------------------------------------------------")
                            print("----------------------------------------------------------------------")
                            print("----------------------------------------------------------------------")
                            self.status = SearchLostBall(self)

        if (x!=0 and y!=0 and raio!=0):
            BallFound = True
        return frame, x, y, raio, BallFound, self.status

#Varredura
def SearchLostBall(self):

    if self.Count == 0:
        servo.writeWord(SERVO_PAN,30 , SERVO_PAN_VALUE - 350) #olha para a esquerda
        time.sleep(1)
        self.Count +=1
        return 0
    if self.Count == 1:
        servo.writeWord(SERVO_PAN,30, SERVO_PAN_VALUE)#olha para o centro
        time.sleep(1)
        self.Count +=1
        return 1
    if self.Count == 2:
        servo.writeWord(SERVO_PAN,30, SERVO_PAN_VALUE + 350)#olha para a direita 850- 440
        time.sleep(1)
        self.Count = 0
        return 2




def Morphology(self, frame, white_mask, kernel, kernel2, k):

    start3 = time.time()
    contador = 0

#    cv2.imshow('mask',white_mask)
    mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel2,1)
# Se a morfologia de perto k =1, recorta a parte de cima
    if k ==1:
        mask[0:200,:]=0
# Se a morfologia medio k =2, recorta a parte de baixo
    if k ==2:
        mask[650:,:]=0
# Se a morfologia de longe k =3, recorta a parte de baixo
    if k ==3:
        mask[450:,:]=0
# Se a morfologia de muito longe k = 4, recorta a parte de baixo
    if k ==4:
        mask[400:,:]=0


    ret,th1 = cv2.threshold(mask,25,255,cv2.THRESH_BINARY)

    _,contours,_ = cv2.findContours(th1, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        contador = contador + 1
        x,y,w,h = cv2.boundingRect(cnt)
            #Passa para o classificador as imagens recortadas-----------------------
        type_label, results = classify(cv2.cvtColor(frame[y:y+h,x:x+w], cv2.COLOR_BGR2RGB),
                                                           self.net, self.transformer,
                                                           mean_file=self.mean_file, labels=self.labels,
                                                           batch_size=None)
        #-----------------------------------------------------------------------

#            print results, type_label
    #       cv2.imshow('janela',images[0])
        if type_label == 'Ball':
#            print "contador = ", contador
#            print "CONTOURS = ", time.time() - start3
            return frame, x+w/2, y+h/2, (w+h)/4, mask
        #=================================================================================================
#    print "CONTOURS = ", time.time() - start 
    return frame, 0, 0, 0, mask




#if __name__ == '__main__':


#    parser = argparse.ArgumentParser(description='Classification example using an archive - DIGITS')

####    # Positional arguments
#    parser.add_argument('archive', help='Path to a DIGITS model archive')
####    #parser.add_argument('image_file', nargs='+', help='Path[s] to an image')
####    # Optional arguments
#    parser.add_argument('--batch-size', type=int)
#    parser.add_argument('--nogpu', action='store_true', help="Don't use the GPU")

#    parser.add_argument('--ws', '--ws', action='store_true', help="no servo")

#    args = vars(parser.parse_args())
#    args2 = parser.parse_args()

#    tmpdir = unzip_archive(args['archive'])
#    caffemodel = None
#    deploy_file = None
#    mean_file = None
#    labels_file = None
#    for filename in os.listdir(tmpdir):
#        full_path = os.path.join(tmpdir, filename)
#        if filename.endswith('.caffemodel'):
#            caffemodel = full_path
#        elif filename == 'deploy.prototxt':
#            deploy_file = full_path
#        elif filename.endswith('.binaryproto'):
#            mean_file = full_path
#        elif filename == 'labels.txt':
#            labels_file = full_path
#        else:
#            print 'Unknown file:', filename

#    assert caffemodel is not None, 'Caffe model file not found'
#    assert deploy_file is not None, 'Deploy file not found'

####    # Load the model and images
#    net = get_net(caffemodel, deploy_file, use_gpu=False)
#    transformer = get_transformer(deploy_file, mean_file)
#    _, channels, height, width = transformer.inputs['data']
#    labels = read_labels(labels_file)

####    #create index from label to use in decicion action
#    number_label =  dict(zip(labels, range(len(labels))))
#    print number_label

##    detectBall = objectDetect(net, transformer, mean_file, labels)

##    os.system("v4l2-ctl -d /dev/video0 -c focus_auto=0 && v4l2-ctl -d /dev/video0 -c focus_absolute=0")
##    cap = cv2.VideoCapture(0)
##    cap.set(3,1280) #720 1280 1920
##    cap.set(4,720) #480 720 1080

##    while True:

##        script_start_time = time.time()

####        # Capture frame-by-frame
##        ret, frame = cap.read()
##        ret, frame = cap.read()
##        ret, frame = cap.read()
##        ret, frame = cap.read()
##        frame = frame[:,200:1100]

##        #===============================================================================
##        frame, x, y, raio = detectBall.searchball(frame)

##        cv2.circle(frame, (x, y), raio, (0, 255, 0), 4)
###        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
##        cv2.imshow('frame',frame)
##        print "tempo de varredura = ", time.time() - script_start_time
##        #===============================================================================


##        print 'Script took %f seconds.' % (time.time() - script_start_time,)

##        if cv2.waitKey(1) & 0xFF == ord('q'):
##            break

##    # When everything done, release the capture
##    cap.release()
##    cv2.destroyAllWindows()

