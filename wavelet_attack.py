import os
from traceback import print_tb
from unittest import result
import wave
from cv2 import imread, imwrite
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt
from pip import main
from sklearn.metrics import roc_curve, auc
from scipy.fft import dct, idct
import random
import cv2
import numpy as np
from numpy import cov
import statistics
from statistics import stdev
import pywt
import pywt.data
from detection_weusedlsb import *
from path_selection import *

# o_path = "0024.bmp"
# ori = cv2.imread(o_path,0)
# mark = np.load('1666596647808_weusedlsb.npy')
# mark = np.uint8(np.rint(mark)).reshape(32, 32)
# watd = mixed_embedding(ori, mark)
# w_path = "watermarked24.bmp"
# cv2.imwrite(w_path, watd)

list_of_broke_wavelet = []

random.seed(3)
def awgn(img, std, seed):
  mean = 0.0   # some constant
  #np.random.seed(seed)
  attacked = img + np.random.normal(mean, std, img.shape)
  attacked = np.clip(attacked, 0, 255)
  return attacked

def blur(img, sigma):
  from scipy.ndimage.filters import gaussian_filter
  attacked = gaussian_filter(img, sigma)
  return attacked

def sharpening(img, sigma, alpha):
  import scipy
  from scipy.ndimage import gaussian_filter
  import matplotlib.pyplot as plt

  #print(img/255)
  filter_blurred_f = gaussian_filter(img, sigma)

  attacked = img + alpha * (img - filter_blurred_f)
  return attacked

def median(img, kernel_size):
  from scipy.signal import medfilt
  attacked = medfilt(img, kernel_size)
  return attacked

def resizing(img, scale):
  from skimage.transform import rescale
  x, y = img.shape
  attacked = rescale(img, scale, preserve_range=True)
  attacked = rescale(attacked, 1/scale, preserve_range=True)
  attacked = attacked[:x, :y]
  return attacked

def jpeg_compression(img, QF):

  from PIL import Image
  img = img.astype(np.uint8)
  img = Image.fromarray(img)
  img.save('tmp.jpg',"JPEG", quality=QF)
  attacked = Image.open('tmp.jpg')
  attacked = np.asarray(attacked,dtype=np.uint8)
  os.remove('tmp.jpg')

  return attacked

def awgn_test(quadrant,quad_lv1, quad_lv2,w_quadrants, w_quadrants2, flag):
    result_AWGN = []
    awgn_img = quadrant.copy()
    for i in range (1,20,4):

        attacked_quad = awgn(awgn_img, i, 255)
        if(flag == 1):
            w_quadrants2[quad_lv2] = attacked_quad
            w_coefficient2 = w_quadrants2[0],(w_quadrants2[1],w_quadrants2[2],w_quadrants2[3])

            #rimettiamo ogni quadrante al suo posto nel primo livello
            w_quadrants[quad_lv1] = pywt.idwt2(w_coefficient2, wavelet='haar')

            coefficient = w_quadrants[0],(w_quadrants[1],w_quadrants[2],w_quadrants[3])

        elif(flag == 0):
            w_quadrants[quad_lv1] = attacked_quad
            coefficient = w_quadrants[0],(w_quadrants[1],w_quadrants[2],w_quadrants[3])
        
        else:
            print('boia mega errore')

        attacked = pywt.idwt2(coefficient, wavelet='haar')
        cv2.imwrite('attacked.bmp', attacked)
        broke, quality = detection(o_path_detection, w_path_detection, 'attacked.bmp')
        #print("quality = ", quality, "param = ",i)        
        awgn_img = quadrant.copy()
        data = ['#AWGN : ', 1, quality, i]
        if(broke == 1 and quality > 40):
          result_AWGN.append([1, broke, quality, i]) #risultati divisi per immagine  
        elif(broke == 0 and quality > 35):
          f = open('result.txt', 'a')
          for d in data:
            f.write(str(d))
            f.write(' ')
          f.write(' // ')
          f.write('\n\n')
          print('brokeeee')
          list_of_broke_wavelet.append([1, broke, quality, i])
        if quality < 35:
            break

    return result_AWGN

def blur_test(quadrant,quad_lv1, quad_lv2,w_quadrants, w_quadrants2, flag):
    result_BLUR = []
    blur_image = quadrant.copy()
    for i in np.arange(5, 11, 5):
        for j in np.arange (5, 11, 5):
            attacked_quad = blur(blur_image, [i, j])
            if(flag == 1):
                w_quadrants2[quad_lv2] = attacked_quad
                w_coefficient2 = w_quadrants2[0],(w_quadrants2[1],w_quadrants2[2],w_quadrants2[3])

                #rimettiamo ogni quadrante al suo posto nel primo livello
                w_quadrants[quad_lv1] = pywt.idwt2(w_coefficient2, wavelet='haar')

                coefficient = w_quadrants[0],(w_quadrants[1],w_quadrants[2],w_quadrants[3])

            elif(flag == 0):
                w_quadrants[quad_lv1] = attacked_quad
                coefficient = w_quadrants[0],(w_quadrants[1],w_quadrants[2],w_quadrants[3])
            
            else:
                print('boia mega errore')

            attacked = pywt.idwt2(coefficient, wavelet='haar')
            cv2.imwrite('attacked.bmp', attacked)
            broke, quality = detection(o_path_detection,  w_path_detection, 'attacked.bmp')
            #print("quality = ", quality, "param = ",i,j)            
            blur_image = quadrant.copy()
            data = ['#BLUR : ', 2, quality, i]
            if(broke == 1 and quality > 40):
                result_BLUR.append([2, broke, quality, i , j]) #risultati divisi per immagine  
            elif(broke == 0 and quality > 35):
                f = open('result.txt', 'a')
                for d in data:
                    f.write(str(d))
                    f.write(' ')
                f.write(' // ')
                f.write('\n\n')
                print('brokeeee')
                list_of_broke_wavelet.append([2, broke, quality, i, j])
            if quality < 35:
              break

    return result_BLUR

def sharpening_test(quadrant,quad_lv1, quad_lv2,w_quadrants, w_quadrants2, flag):

    result_SHARPENING = []
    sharp_image = quadrant.copy()
    for i in np.arange(0.5, 2.6, 1):
        for j in np.arange (0.5, 2.6, 1):
            attacked_quad = sharpening(sharp_image, i, j)
            if(flag == 1):
                w_quadrants2[quad_lv2] = attacked_quad
                w_coefficient2 = w_quadrants2[0],(w_quadrants2[1],w_quadrants2[2],w_quadrants2[3])

                #rimettiamo ogni quadrante al suo posto nel primo livello
                w_quadrants[quad_lv1] = pywt.idwt2(w_coefficient2, wavelet='haar')

                coefficient = w_quadrants[0],(w_quadrants[1],w_quadrants[2],w_quadrants[3])

            elif(flag == 0):
                w_quadrants[quad_lv1] = attacked_quad
                coefficient = w_quadrants[0],(w_quadrants[1],w_quadrants[2],w_quadrants[3])
            
            else:
                print('boia mega errore')

            attacked = pywt.idwt2(coefficient, wavelet='haar')
            cv2.imwrite('attacked.bmp', attacked)
            broke, quality = detection(o_path_detection, w_path_detection, 'attacked.bmp')
            #print("quality = ", quality, "param = ",i,j)            
            sharp_image = quadrant.copy()
            data = ['#SHARPENING : ', 3, quality, i]
            if(broke == 1 and quality > 40):
                result_SHARPENING.append([3, broke, quality, i, j]) #risultati divisi per immagine  
            elif(broke == 0 and quality > 35):
                f = open('result.txt', 'a')
                for d in data:
                    f.write(str(d))
                    f.write(' ')
                f.write(' // ')
                f.write('\n\n')
                print('brokeeee')
                list_of_broke_wavelet.append([3, broke, quality, i, j])
            if quality < 35:
              break

    return result_SHARPENING

def median_test(quadrant,quad_lv1, quad_lv2,w_quadrants, w_quadrants2, flag):
    result_MEDIAN = []
    median_image = quadrant.copy()
    for i in range(3,9,2):
        for j in range(3,9,2):
            attacked_quad = median(median_image, [i, j])
            if(flag == 1):
                w_quadrants2[quad_lv2] = attacked_quad
                w_coefficient2 = w_quadrants2[0],(w_quadrants2[1],w_quadrants2[2],w_quadrants2[3])

                #rimettiamo ogni quadrante al suo posto nel primo livello
                w_quadrants[quad_lv1] = pywt.idwt2(w_coefficient2, wavelet='haar')

                coefficient = w_quadrants[0],(w_quadrants[1],w_quadrants[2],w_quadrants[3])

            elif(flag == 0):
                w_quadrants[quad_lv1] = attacked_quad
                coefficient = w_quadrants[0],(w_quadrants[1],w_quadrants[2],w_quadrants[3])
            
            else:
                print('boia mega errore')

            attacked = pywt.idwt2(coefficient, wavelet='haar')

            cv2.imwrite('attacked.bmp', attacked)
            broke, quality = detection(o_path_detection, w_path_detection , 'attacked.bmp')
            #print("quality = ", quality, "param = ",i,j)            
            median_image = quadrant.copy()
            data = ['#MEDIAN : ', 4, quality, i]
            if(broke == 1 and quality > 40):
                result_MEDIAN.append([4, broke, quality, i, j]) #risultati divisi per immagine
            elif (broke == 0 and quality > 35):
                f = open('result.txt', 'a')
                for d in data:
                    f.write(str(d))
                    f.write(' ')
                f.write(' // ')
                f.write('\n\n')
                print('brokeeee')
                list_of_broke_wavelet.append([4, broke, quality, i, j]) 
            if quality < 35:
              break

    return result_MEDIAN

def resizing_test(quadrant,quad_lv1, quad_lv2,w_quadrants, w_quadrants2, flag):
    result_Resizing = []
    resized_img = quadrant.copy()
    for i in np.arange (0.2, 0.8, 0.1):

        attacked_quad = resizing(resized_img, i)
        if(attacked_quad.shape == (128,128)):
            if(flag == 1):
                w_quadrants2[quad_lv2] = attacked_quad
                w_coefficient2 = w_quadrants2[0],(w_quadrants2[1],w_quadrants2[2],w_quadrants2[3])

                #rimettiamo ogni quadrante al suo posto nel primo livello
                w_quadrants[quad_lv1] = pywt.idwt2(w_coefficient2, wavelet='haar')

                coefficient = w_quadrants[0],(w_quadrants[1],w_quadrants[2],w_quadrants[3])

            elif(flag == 0):
                w_quadrants[quad_lv1] = attacked_quad
                coefficient = w_quadrants[0],(w_quadrants[1],w_quadrants[2],w_quadrants[3])
        
            else:
                print('boia mega errore')

            attacked = pywt.idwt2(coefficient, wavelet='haar')

            cv2.imwrite('attacked.bmp', attacked)
            broke, quality = detection(o_path_detection, w_path_detection , 'attacked.bmp')
            #print("quality = ", quality, "param = ",i)            
            resized_img = quadrant.copy()
            data = ['#RESIZING : ', 5, quality, i]
            if(broke == 1 and quality > 40):
                result_Resizing.append([5, broke, quality, i]) #risultati divisi per immagine
            elif (broke == 0 and quality > 35):
                f = open('result.txt', 'a')
                for d in data:
                    f.write(str(d))
                    f.write(' ')
                f.write(' // ')
                f.write('\n\n')
                print('brokeeee')
                list_of_broke_wavelet.append([5, broke, quality, i]) 

    return result_Resizing

def jpeg_test(quadrant,quad_lv1, quad_lv2,w_quadrants, w_quadrants2, flag):

    result_JPEG = []
    jpeg_img = quadrant.copy()
    for i in range (79, 100, 10):
        attacked_quad = jpeg_compression(jpeg_img, i)
        if(flag == 1):
            w_quadrants2[quad_lv2] = attacked_quad
            w_coefficient2 = w_quadrants2[0],(w_quadrants2[1],w_quadrants2[2],w_quadrants2[3])

            #rimettiamo ogni quadrante al suo posto nel primo livello
            w_quadrants[quad_lv1] = pywt.idwt2(w_coefficient2, wavelet='haar')

            coefficient = w_quadrants[0],(w_quadrants[1],w_quadrants[2],w_quadrants[3])

        elif(flag == 0):
            w_quadrants[quad_lv1] = attacked_quad
            coefficient = w_quadrants[0],(w_quadrants[1],w_quadrants[2],w_quadrants[3])
        
        else:
            print('boia mega errore')

        attacked = pywt.idwt2(coefficient, wavelet='haar')

        cv2.imwrite('attacked.bmp', attacked)
        broke, quality = detection(o_path_detection, w_path_detection , 'attacked.bmp')
        #print("quality = ", quality, "param = ",i)
        jpeg_img = quadrant.copy()
        data = ['#JPEG : ', 6, quality, i]
        if(broke == 1 and quality > 40):
          result_JPEG.append([6, broke, quality, i]) #risultati divisi per immagine
        elif (broke == 0 and quality > 35):
          f = open('result.txt', 'a')
          for d in data:
            f.write(str(d))
            f.write(' ')
          f.write(' // ')
          f.write('\n\n')
          print('brokeeee')
          list_of_broke_wavelet.append([6, broke, quality, i]) 

    return result_JPEG

def run_base_test(quadrant,quad_lv1, quad_lv2,w_quadrants, w_quadrants2, flag):
    
    all_test_check = []

    print('----------------- AWGN --------------------')
    result_awg = awgn_test(quadrant,quad_lv1, quad_lv2,w_quadrants, w_quadrants2, flag)
    all_test_check.extend(result_awg)

    print('----------------- BLUR --------------------')
    result_blur = blur_test(quadrant,quad_lv1, quad_lv2,w_quadrants, w_quadrants2, flag)
    all_test_check.extend(result_blur)

    print('----------------- SHARPENING --------------------')
    result_sharpening= sharpening_test(quadrant,quad_lv1, quad_lv2,w_quadrants, w_quadrants2, flag)
    all_test_check.extend(result_sharpening)

    print('----------------- MEDIAN --------------------')
    result_median = median_test(quadrant,quad_lv1, quad_lv2,w_quadrants, w_quadrants2, flag)
    all_test_check.extend(result_median)

    print('----------------- RESIZING --------------------')
    result_resizing = resizing_test(quadrant,quad_lv1, quad_lv2,w_quadrants, w_quadrants2, flag)
    all_test_check.extend(result_resizing)

    print('----------------- JPEG --------------------')
    result_jpeg = jpeg_test(quadrant,quad_lv1, quad_lv2,w_quadrants, w_quadrants2, flag)
    if(len(result_jpeg) > 0):
     all_test_check.extend(result_jpeg.reverse())   

    print(all_test_check)
    return all_test_check

def attack_selection(atck_data, img):
  if atck_data[0] == 1:
    img = awgn(img, atck_data[3], 255)
  elif atck_data[0] == 2:
    img = blur(img, [atck_data[3],atck_data[4]])
  elif atck_data[0] == 3:
    img = sharpening(img, atck_data[3], atck_data[4])
  elif atck_data[0] == 4:
    img = median(img, [atck_data[3],atck_data[4]])
  elif atck_data[0] == 5:
    img = resizing(img, atck_data[3])
  elif atck_data[0] == 6:
    img = jpeg_compression(img, atck_data[3])
  else:
    print('invalid selection')
  
  return img

def write_atck(attack_flag, f):

    if(attack_flag == 1 ):
        f.write('AWGN')
    if(attack_flag == 2 ):
        f.write('BLUR')
    if(attack_flag == 3 ):
        f.write('SHARPENING')
    if(attack_flag == 4 ):
        f.write('MEDIAN')
    if(attack_flag == 5 ):
        f.write('RESIZING')
    if(attack_flag == 6 ):
        f.write('JPEG')

def double_attack(real_img, watd, result_tests, quadrant, quad_lv1, quad_lv2,w_quadrants, w_quadrants2, flag ):

    print('----------------- Double --------------------')
    safe = quadrant.copy()
    if (len(result_tests)>0):
      for i in range(len(result_tests)):
          attacked = attack_selection(result_tests[i], safe)
          saf_att = attacked.copy()
          for k in range(i, len(result_tests)):
            attacked_2 = attack_selection(result_tests[k], saf_att)
            if(flag == 1):
                w_quadrants2[quad_lv2] = attacked_2
                w_coefficient2 = w_quadrants2[0],(w_quadrants2[1],w_quadrants2[2],w_quadrants2[3])

                #rimettiamo ogni quadrante al suo posto nel primo livello
                w_quadrants[quad_lv1] = pywt.idwt2(w_coefficient2, wavelet='haar')

                coefficient = w_quadrants[0],(w_quadrants[1],w_quadrants[2],w_quadrants[3])

            elif(flag == 0):
                w_quadrants[quad_lv1] = attacked_2
                coefficient = w_quadrants[0],(w_quadrants[1],w_quadrants[2],w_quadrants[3])
            
            else:
                print('boia mega errore')

            attacked2 = pywt.idwt2(coefficient, wavelet='haar')
            cv2.imwrite('attacked.bmp', attacked2)
            broke, quality = detection(real_img, watd, 'attacked.bmp')
            if(broke == 0 and quality > 35):

              f = open('result.txt', 'a')
              f.write('#')
              write_atck(result_tests[i][0], f)
              f.write(' + ')
              write_atck(result_tests[k][0], f)
              f.write(' : ')
              print('rottoooo')
              if(len(result_tests[i]) > 4):
                if(len(result_tests[k]) > 4):
                  list_of_broke_wavelet.append([result_tests[i][0], result_tests[i][3], result_tests[i][4] , broke, quality, result_tests[k][0], result_tests[k][3], result_tests[k][4]])
                  data = [result_tests[i][0], result_tests[i][3], result_tests[i][4] , quality, result_tests[k][0], result_tests[k][3], result_tests[k][4]]
                  for d in data:
                    f.write(str(d))
                    f.write(' ')
                  f.write(' // ')
                else:
                  list_of_broke_wavelet.append([result_tests[i][0], result_tests[i][3], result_tests[i][4], broke, quality, result_tests[k][0], result_tests[k][3]])
                  data = [result_tests[i][0], result_tests[i][3], result_tests[i][4], quality, result_tests[k][0], result_tests[k][3]]
                  for d in data:
                    f.write(str(d))
                    f.write(' ')
                  f.write(' // ')
              elif(len(result_tests[k]) > 4):
                list_of_broke_wavelet.append([result_tests[i][0], result_tests[i][3], broke, quality,result_tests[k][0], result_tests[k][3], result_tests[k][4] ])
                data = [result_tests[i][0], result_tests[i][3], quality,result_tests[k][0], result_tests[k][3], result_tests[k][4] ]
                for d in data:
                    f.write(str(d))
                    f.write(' ')
                f.write(' // ')
              else:
                list_of_broke_wavelet.append([result_tests[i][0], result_tests[i][3], broke, quality,result_tests[k][0], result_tests[k][3]])
                data = [result_tests[i][0], result_tests[i][3], quality,result_tests[k][0], result_tests[k][3]]
                
                for d in data:
                    f.write(str(d))
                    f.write(' ')
                f.write(' // ')

              f.write('\n\n')


            saf_att = attacked.copy()
          safe = quadrant.copy()

def attacco_wavelet(quad_lv1, quad_lv2,w_quadrants, w_quadrants2):

    if quad_lv2 != -1:

        result_test = run_base_test(w_quadrants2[quad_lv2], quad_lv1, quad_lv2,w_quadrants, w_quadrants2, flag = 1)

        double_attack(o_path_detection, w_path_detection, result_test, w_quadrants2[quad_lv2], quad_lv1, quad_lv2,w_quadrants, w_quadrants2, flag = 1)

    else:
        
        result_test = run_base_test(w_quadrants2[quad_lv1], quad_lv1, quad_lv2, w_quadrants, w_quadrants2, flag = 0)

        double_attack(o_path_detection, w_path_detection, result_test, w_quadrants2[quad_lv1], quad_lv1, quad_lv2,w_quadrants, w_quadrants2, flag = 0)
