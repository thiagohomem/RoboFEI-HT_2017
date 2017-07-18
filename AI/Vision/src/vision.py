# -*- coding: UTF8 -*-

import sys
sys.path.append("./src")
import numpy as np
import os
import cv2
import ctypes
import argparse
import time
from math import log,exp,tan,radians
import thread
import imutils

#from BallVision import *
from DNN import *

import sys

""" Initiate the path to blackboard (Shared Memory)"""
sys.path.append('../../Blackboard/src/')
"""Import the library Shared memory """
from SharedMemory import SharedMemory 
""" Treatment exception: Try to import configparser from python. Write and Read from config.ini file"""
try:
    """There are differences in versions of the config parser
    For versions > 3.0 """
    from ConfigParser import ConfigParser
except ImportError:
    """For versions < 3.0 """
    from ConfigParser import ConfigParser 

""" Instantiate bkb as a shared memory """
bkb = SharedMemory()
""" Config is a new configparser """
config = ConfigParser()
""" Path for the file config.ini:"""
config.read('../../Control/Data/config.ini')
""" Mem_key is for all processes to know where the blackboard is. It is robot number times 100"""
mem_key = int(config.get('Communication', 'no_player_robofei'))*100 
"""Memory constructor in mem_key"""
Mem = bkb.shd_constructor(mem_key)


parser = argparse.ArgumentParser(description='Robot Vision', epilog= 'Responsavel pela deteccao dos objetos em campo / Responsible for detection of Field objects')
parser.add_argument('--visionball', '--vb', action="store_true", help = 'Calibra valor para a visao da bola')
parser.add_argument('--visionMask', '--vm', action="store_true", help = 'Calibra valor para a visao da bola')
parser.add_argument('--visionMorph1', '--vm1', action="store_true", help = 'Mostra a imagem da morfologia perto')
parser.add_argument('--visionMorph2', '--vm2', action="store_true", help = 'Mostra a imagem da morfologia medio')
parser.add_argument('--visionMorph3', '--vm3', action="store_true", help = 'Mostra a imagem da morfologia medio')
parser.add_argument('--withoutservo', '--ws', action="store_true", help = 'Servos desligado')
parser.add_argument('--head', '--he', action="store_true", help = 'Configurando parametros do controle da cabeca')
parser.add_argument('archive', help='Path to a DIGITS model archive')
###    #parser.add_argument('image_file', nargs='+', help='Path[s] to an image')
###    # Optional arguments
parser.add_argument('--batch-size', type=int)
parser.add_argument('--nogpu', action='store_true', help="Don't use the GPU")


#----------------------------------------------------------------------------------------------------------------------------------




class Parser(object):

	config = None
	self.Config = ConfigParser()


	def __readConfig(self):
		# Read file config.ini
		while True:
			if self.Config.read('../Data/config.ini') and 'Vision Ball' in self.Config.sections():
				print 'Leitura do config.ini, Vision Ball'
				SERVO_PAN = 			self.Config.getint('Basic Settings', 'Center_Servo_Pan')
				SERVO_TILT  = 			self.Config.getint('Basic Settings', 'Center_Servo_Tilt')

				SERVO_PAN_LEFT = 		self.Config.getint('Basic Settings', 'Limit_Servo_Pan_Left')
				SERVO_PAN_RIGHT  = 		self.Config.getint('Basic Settings', 'Limit_Servo_Pan_Right')

				SERVO_TILT_VALUE = 		self.Config.getint('Basic Settings', 'PAN_ID')
				SERVO_PAN_VALUE = 		self.Config.getint('Basic Settings', 'TILT_ID')


				kernel_perto = 			self.Config.getint('Kernel Selection', 'Kernel_closest_Erosion')
				kernel_perto2 = 		self.Config.getint('Kernel Selection', 'Kernel_closest_Dilation')
			
				kernel_medio = 			self.Config.getint('Kernel Selection','Kernel_very_close_Erosion')
				kernel_medio2 = 		self.Config.getint('Kernel Selection','Kernel_very_close_Dilation')

				kernel_longe = 			self.Config.getint('Kernel Selection', 'Kernel_close_Erosion')
				kernel_longe2 = 		self.Config.getint('Kernel Selection', 'Kernel_close_Erosion')

				kernel_muito_longe = 	self.Config.getint('Kernel Selection', 'Kernel_far_Erosion')
				kernel_muito_longe2 = 	self.Config.getint('Kernel Selection', 'Kernel_far_Dilation')





				x_esquerdo = 		self.Config.getint('Distance Limits (Pixels)', 'Left_Region_Division')
				x_centro_esquerdo = self.Config.getint('Distance Limits (Pixels)', 'Center_Left_Region_Division')
				x_centro = 			self.Config.getint('Distance Limits (Pixels)', 'Center_Region_Division')
				x_centro_direito = 	self.Config.get('Distance Limits (Pixels)', 'Center_Right_Region_Division')
				x_direito = 		self.Config.getint('Distance Limits (Pixels)', 'Right_Region_Division')


				break

			else:
				print 'Falha na leitura do config.ini, criando arquivo\nVision Ball inexistente, criando valores padrao'
				self.Config = ConfigParser()
				self.Config.write('../Data/config.ini')

				self.Config.add_section('Basic Settings')
				self.Config.set('Basic Settings', 'Center_Servo_Pan', str(512)+'\t\t\t;Center Servo PAN Position')
				self.Config.set('Basic Settings', 'Center_Servo_Tilt', str(705)+'\t;Center Servo TILT Position')

				self.Config.set('Basic Settings', 'Limit_Servo_Pan_Left', str(162)+'\t\t\t;Center Servo PAN Position')
				self.Config.set('Basic Settings', 'Limit_Servo_Pan_Right', str(862)+'\t;Center Servo TILT Position')

				self.Config.set('Basic Settings', 'PAN_ID', str(19)+'\t\t\t;Servo Identification number for PAN')
				self.Config.set('Basic Settings', 'TILT_ID', str(20)+'\t;Servo Identification number for TILT')

				self.Config.add_section('Kernel Selection')
				self.Config.set('Kernel Selection', 'Kernel_closest_Erosion', str(39)+'\t\t\t;Kernel Erosion ball is closest the robot')
				self.Config.set('Kernel Selection', 'Kernel_closest_Dilation', str(100)+'\t;Kernel Dilation ball is closest the robot')
				self.Config.set('Kernel Selection', 'Kernel_very_close_Erosion', str(22)+'\t\t\t;Kernel Erosion ball is very close to the robot')
				self.Config.set('Kernel Selection', 'Kernel_very_close_Dilation', str(80)+'\t;Kernel Dilation ball is very close to the robot')
				self.Config.set('Kernel Selection', 'Kernel_close_Erosion', str(12)+'\t\t\t;Kernel Erosion ball is close to the robot')
				self.Config.set('Kernel Selection', 'Kernel_close_Dilation', str(40)+'\t;Kernel Dilation ball is close to the robot')
				self.Config.set('Kernel Selection', 'Kernel_far_Erosion', str(7)+'\t\t\t;Kernel Erosion ball is far from the robot')
				self.Config.set('Kernel Selection', 'Kernel_far_Dilation', str(30)+'\t;Kernel Dilation ball is far from the robot')

				self.Config.add_section('Distance Limits (Pixels)')
				self.Config.set('Distance Limits (Pixels)', 'Left_Region_Division'         , str(280)+'\t\t\t;X Screen Left division')
				self.Config.set('Distance Limits (Pixels)', 'Center_Left_Region_Division'  , str(320)+'\t\t\t;X Screen Center Left division')
				self.Config.set('Distance Limits (Pixels)', 'Center_Region_Division'       , str(465)+'\t\t\t;X Screen Center division')
				self.Config.set('Distance Limits (Pixels)', 'Center_Right_Region_Division' , str(645)+'\t\t\t;X Screen Center Right division')
				self.Config.set('Distance Limits (Pixels)', 'Right_Region_Division'        , str(703)+'\t\t\t;X Screen Right division')

				self.Config.set('Distance Limits (Pixels)', 'Down_Region_Division'  , str(549)+'\t\t\t;Y Screen Down division')
				self.Config.set('Distance Limits (Pixels)', 'Up_Region_Division'    , str(220)+'\t\t\t;Y Screen Up division')

				with open('../Data/config.ini', 'wb') as configfile:
					self.Config.write(configfile)





x_centro = 465

x_esquerdo = 280
x_centro_esquerdo = 320
x_direito = 703
x_centro_direito = 645

y_chute = 549
y_longe = 220






x = 0
y = 0
raio = 0


def BallStatus(x,y,status):
	print "X = ", x

	if status  == 1:
		#Bola a esquerda
		if (x <= x_esquerdo):
			bkb.write_float(Mem,'VISION_PAN_DEG', 60) # Posição da bola
			print ("Bola a Esquerda")

		#Bola ao centro esquerda
		if (x > x_esquerdo and x < x_centro):
			bkb.write_float(Mem,'VISION_PAN_DEG', 30) # Variavel da telemetria
			print ("Bola ao Centro Esquerda")

		#Bola centro direita
		if (x < x_direito and x > x_centro):
			bkb.write_float(Mem,'VISION_PAN_DEG', -30) # Variavel da telemetria
			print ("Bola ao Centro Direita")

		#Bola a direita
		if (x >= x_direito):
			bkb.write_float(Mem,'VISION_PAN_DEG', -60) # Variavel da telemetria
			print ("Bola a Direita")

	else: 
		if (status ==2):
			bkb.write_float(Mem,'VISION_PAN_DEG', 60) # Posição da bola
			print ("Bola a Esquerda")
		else:
			bkb.write_float(Mem,'VISION_PAN_DEG', -60) # Variavel da telemetria
			print ("Bola a Direita")


#	#CUIDADO AO ALTERAR OS VALORES ABAIXO!! O código abaixo possui inversão de eixos!
#	# O eixo em pixels é de cima para baixo ja as distancias são ao contrario.
#	# Quanto mais alto a bola na tela menor o valor em pixels 
#	# e mais longe estará a bola do robô
	#Bola abaixo
	if (y < y_longe):
		bkb.write_float(Mem,'VISION_TILT_DEG', 70) # Variavel da telemetria
		print ("Bola acima")
	#Bola ao centro
	if (y < y_chute and y > y_longe):
		bkb.write_float(Mem,'VISION_TILT_DEG', 45) # Variavel da telemetria
		print ("Bola Centralizada")
	#Bola acima
	if (y >= y_chute):
		bkb.write_float(Mem,'VISION_TILT_DEG', 0) # Variavel da telemetria
		print ("Bola abaixo")

#	if status  == 1:
#		#Bola a esquerda
#		if (x > x_limit01 and x < x_limit12):
#			bkb.write_float(Mem,'VISION_PAN_DEG', 60) # Posição da bola
#			print ("Bola a Esquerda")

#		#Bola ao centro
#		if (x > x_limit12 and x < x_limit23):
#			bkb.write_float(Mem,'VISION_PAN_DEG', 30) # Variavel da telemetria
#			print ("Bola ao Centro Esquerda")

#		#Bola a direita
#		if (x > x_limit23 and x < x_limit34):
#			bkb.write_float(Mem,'VISION_PAN_DEG', -30) # Variavel da telemetria
#			print ("Bola ao Centro Direita")

#		#Bola a direita
#		if (x > x_limit34 and x < x_limit45):
#			bkb.write_float(Mem,'VISION_PAN_DEG', -60) # Variavel da telemetria
#			print ("Bola a Direita")
#	else: 
#		if (status ==2):
#			bkb.write_float(Mem,'VISION_PAN_DEG', 60) # Posição da bola
#			print ("Bola a Esquerda")
#		else:
#			bkb.write_float(Mem,'VISION_PAN_DEG', -60) # Variavel da telemetria
#			print ("Bola a Direita")


#	#CUIDADO AO ALTERAR OS VALORES ABAIXO!! O código abaixo possui inversão de eixos!
#	# O eixo em pixels é de cima para baixo ja as distancias são ao contrario.
#	# Quanto mais alto a bola na tela menor o valor em pixels 
#	# e mais longe estará a bola do robô

#	#Bola abaixo
#	if (y > 1 and y < 200):#y_limit1):
#		bkb.write_float(Mem,'VISION_TILT_DEG', 70) # Variavel da telemetria
#		print ("Bola acima")
#	#Bola ao centro
#	if (y > y_limit1 and y < y_limit2):
#		bkb.write_float(Mem,'VISION_TILT_DEG', 45) # Variavel da telemetria
#		print ("Bola Centralizada")
#	#Bola acima
#	if (y > y_limit2 and y < 720):
#		bkb.write_float(Mem,'VISION_TILT_DEG', 0) # Variavel da telemetria
#		print ("Bola abaixo")

#A funcao do Vini
def applyMask(frame):
	lower = np.array([23, 0,0])
	upper = np.array([57,255,255])
        kernel = np.ones((5,5),np.uint8)
#        mask = frame
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	
	mask = cv2.inRange(hsv, lower, upper)
	
	## erosion
	mask = cv2.erode(mask,kernel,iterations=2)
	
	## dilation
	mask = cv2.dilate(mask,kernel,iterations=2)
	#mostra = cv2.bitwise_and(frame,frame,mask=mask)
	return mask




#Outra funcao do Vini
def cutFrame(mask_verde):
##	#cima
	cima = -1
	for i in range(0,len(mask_verde),5):
		if np.any(sum(mask_verde[i]))>int(255*500): #minimo pixels
			cima = i
			break

#	#baixo
	baixo = -1
	for i in range(len(mask_verde)-1,-1,-5):
		if np.any(sum(mask_verde[i]))>int(255*500): #minimo pixels
			baixo = i
			break
#	
#	# Girando mascara
	mask_verde=mask_verde.transpose()
#	
#	#esp p/ dir
	esquerda = -1
	for i in range(0,len(mask_verde),5):
		if np.any(sum(mask_verde[i]))>int(255*500): #minimo 4 pixels
			esquerda = i
			break
#	
#	#dir p/ esp
	direita = -1
	for i in range(len(mask_verde)-1,0,-5):
		if np.any(sum(mask_verde[i]))>int(255*500): #minimo 4 pixels
			direita = i
			break
	
	return np.array([esquerda,direita,cima,baixo])


def thread_DNN():
	time.sleep(1)
	while True:
#		script_start_time = time.time()

#		print "FRAME = ", time.time() - script_start_time
		start1 = time.time()
#===============================================================================
		ball = False
		frame_b, x, y, raio, ball, status= detectBall.searchball(frame, args2.visionMask, args2.visionMorph1, args2.visionMorph2, args2.visionMorph3)
		print "tempo de varredura = ", time.time() - start1
		if ball ==False:
			bkb.write_int(Mem,'VISION_LOST', 1)
		else:
			bkb.write_int(Mem,'VISION_LOST', 0)
			BallStatus(x,y,status)
		if args2.visionball:
			cv2.circle(frame_b, (x, y), raio, (0, 255, 0), 4)
			cv2.imshow('frame',frame_b)
#===============================================================================
#		print "tempo de varredura = ", time.time() - start
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()

#frame = 0







#----------------------------------------------------------------------------------------------------------------------------------
#Inicio programa

if __name__ == '__main__':


	args = vars(parser.parse_args())
	args2 = parser.parse_args()

	tmpdir = unzip_archive(args['archive'])
	caffemodel = None
	deploy_file = None
	mean_file = None
	labels_file = None
	for filename in os.listdir(tmpdir):
		full_path = os.path.join(tmpdir, filename)
		if filename.endswith('.caffemodel'):
			caffemodel = full_path
		elif filename == 'deploy.prototxt':
			deploy_file = full_path
		elif filename.endswith('.binaryproto'):
			mean_file = full_path
		elif filename == 'labels.txt':
			labels_file = full_path
		else:
			print 'Unknown file:', filename

	assert caffemodel is not None, 'Caffe model file not found'
	assert deploy_file is not None, 'Deploy file not found'

###    # Load the model and images
	net = get_net(caffemodel, deploy_file, use_gpu=False)
	transformer = get_transformer(deploy_file, mean_file)
	_, channels, height, width = transformer.inputs['data']
	labels = read_labels(labels_file)

###    #create index from label to use in decicion action
	number_label =  dict(zip(labels, range(len(labels))))
	print number_label
###    #
	detectBall = objectDetect(net, transformer, mean_file, labels)


	cap = cv2.VideoCapture(0) #Abrindo camera
        cap.set(3,1280) #720 1280 1920
        cap.set(4,720) #480 720 1080
	os.system("v4l2-ctl -d /dev/video0 -c focus_auto=0 && v4l2-ctl -d /dev/video0 -c focus_absolute=0")
	os.system("v4l2-ctl -d /dev/video0 -c saturation=200")

	try:
            thread.start_new_thread(thread_DNN, ())
	except:
            print "Error Thread"

	while True:

		bkb.write_int(Mem,'VISION_WORKING', 1) # Variavel da telemetria

		#Salva o frame

#		script_start_time = time.time()

                ret, frame = cap.read()
                frame = frame[:,200:1100]
		
		
#		print "FRAME = ", time.time() - script_start_time
#		start = time.time()
#===============================================================================
#		cv2.imshow('Original',frame)
#		mask_verde = applyMask(frame)
#		#if mask_verde is not 0:
#		cut = cutFrame(mask_verde)
#		#frame_campo = mask_verde
#		frame_campo = frame[cut[2]:cut[3], cut[0]:cut[1]]
#		mostra = cv2.bitwise_and(frame,frame,mask=mask_verde)
#		cv2.imshow('Frame Cortado Grama',mostra)
#		frame_b, x, y, raio = detectBall.searchball(frame)
#		BallStatus(x,y)
#		if args2.visionball:
#			cv2.circle(frame, (x, y), raio, (0, 255, 0), 4)
#			cv2.imshow('Frame Deteccao',frame)
#===============================================================================
#		print "tempo de varredura = ", time.time() - start
		#cv2.imshow('frame',frame)
#		if cv2.waitKey(1) & 0xFF == ord('q'):
#			break
		time.sleep(0.01)

#===============================================================================

#		if args2.withoutservo == False:
#			posheadball = head.mov(positionballframe,posheadball,Mem, bkb)
	

#	raw_input("Pressione enter pra continuar")

#	if args2.withoutservo == False:
#		head.finalize()
#	ball.finalize()
#	cv2.destroyAllWindows()
#	cap.release()
