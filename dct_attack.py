from traceback import print_tb
from unittest import result
from cv2 import imread, imwrite
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt
from pip import main
from sklearn.metrics import roc_curve, auc
from scipy.fft import dct, idct
import cv2
from numpy import cov
from statistics import stdev
import pywt
import pywt.data
from path_selection import *
from detection_weusedlsb import *

list_of_broke_dct = []

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



def awgn_test(watermarked, wat_dct, locations, bit_esclusi,n_bit):
    result_AWGN = []
    awgn_wat = watermarked.copy()
    awgn_img = wat_dct.copy()
    for i in np.arange(8, 15, 2):

        attacked_wat = awgn(awgn_wat, i, 255)
        attacked_dct = dct(dct(attacked_wat,axis=0, norm='ortho'),axis=1, norm='ortho').flatten() 

        for ind in locations[bit_esclusi:(n_bit+bit_esclusi)]: 
            awgn_img[ind] = attacked_dct[ind]
        
        attacked = idct(idct(awgn_img.reshape(512,512),axis=1, norm='ortho'),axis=0, norm='ortho')
        
        cv2.imwrite('attacked.bmp', attacked)
        broke, quality = detection(o_path_detection, w_path_detection, 'attacked.bmp')
        print("quality = ", quality, "param = ",i)
        awgn_img = wat_dct.copy()
        awgn_wat = watermarked.copy()
        data = ['#AWGN : ', 1, quality, i]
        if(broke == 1 and quality > 40):
          result_AWGN.append([1, broke, quality, i]) #risultati divisi per immagine  
        elif(broke == 0 and quality > 35):
          f = open('result.txt', 'a')
          f.write('\n\n')
          for d in data:
            f.write(str(d))
            f.write(' ')
          f.write(' // ')
          print('brokeeee')
          list_of_broke_dct.append([1, broke, quality, i])
        if quality < 35:
            break

    return result_AWGN

def blur_test(watermarked, wat_dct, locations, bit_esclusi,n_bit):
    result_BLUR = []
    blur_image = wat_dct.copy()
    for i in np.arange(1, 11, 5):
        for j in np.arange (1, 11, 5):
            attacked_wat = blur(watermarked, [i, j])
            attacked_dct = dct(dct(attacked_wat,axis=0, norm='ortho'),axis=1, norm='ortho').flatten() 

            for ind in locations[bit_esclusi:(n_bit+bit_esclusi)]:
                blur_image[ind] = attacked_dct[ind]

            attacked = idct(idct(blur_image.reshape(512,512),axis=1, norm='ortho'),axis=0, norm='ortho')
            cv2.imwrite('attacked.bmp', attacked)

            broke, quality = detection(o_path_detection, w_path_detection, 'attacked.bmp')
            print("quality = ", quality, "param = ",i,j)
            blur_image = wat_dct.copy()
            data = ['#BLUR : ', 2, quality, i]
            if(broke == 1 and quality > 40):
                result_BLUR.append([2, broke, quality, i , j]) #risultati divisi per immagine  
            elif(broke == 0 and quality > 35):
                f = open('result.txt', 'a')
                f.write('\n\n')
                for d in data:
                    f.write(str(d))
                    f.write(' ')
                f.write(' // ')
                print('brokeeee')
                list_of_broke_dct.append([1, broke, quality, i])
            if quality < 35:
              break

    return result_BLUR

def sharpening_test(watermarked, wat_dct, locations, bit_esclusi,n_bit):

    result_SHARPENING = []
    sharp_image = wat_dct.copy()
    for i in np.arange(.2, .32, 0.1):
        for j in np.arange (1, 5.5, 1.2):
            attacked_wat = sharpening(watermarked, i, j)
            attacked_dct = dct(dct(attacked_wat,axis=0, norm='ortho'),axis=1, norm='ortho').flatten() 

            for ind in locations[bit_esclusi:(n_bit+bit_esclusi)]:
                sharp_image[ind] = attacked_dct[ind]
        
            attacked = idct(idct(sharp_image.reshape(512,512),axis=1, norm='ortho'),axis=0, norm='ortho')
        
            cv2.imwrite('attacked.bmp', attacked)
            broke, quality = detection(o_path_detection, w_path_detection, 'attacked.bmp')
            print("quality = ", quality, "param = ",i,j)
            sharp_image = wat_dct.copy()
            data = ['#SHARPENING : ', 3, quality, i]
            if(broke == 1 and quality > 40):
                result_SHARPENING.append([3, broke, quality, i, j]) #risultati divisi per immagine  
            elif(broke == 0 and quality > 35):
                f = open('result.txt', 'a')
                f.write('\n\n')
                for d in data:
                    f.write(str(d))
                    f.write(' ')
                f.write(' // ')
                print('brokeeee')
                list_of_broke_dct.append([3, broke, quality, i, j])
            if quality < 35:
              break 

    return result_SHARPENING

def median_test(watermarked, wat_dct, locations, bit_esclusi,n_bit):
    result_MEDIAN = []
    median_image = wat_dct.copy()
    for i in range(5,9,2):
        for j in range(3,9,2):
            attacked_wat = median(watermarked, [i, j])
            attacked_dct = dct(dct(attacked_wat,axis=0, norm='ortho'),axis=1, norm='ortho').flatten() 

            for ind in locations[bit_esclusi:(n_bit+bit_esclusi)]: 
                median_image[ind] = attacked_dct[ind]
        
            attacked = idct(idct(median_image.reshape(512,512),axis=1, norm='ortho'),axis=0, norm='ortho')
            cv2.imwrite('attacked.bmp', attacked)
            broke, quality = detection(o_path_detection, w_path_detection, 'attacked.bmp')
            print("quality = ", quality, "param = ",i,j)
            median_image = wat_dct.copy()
            data = ['#MEDIAN : ', 4, quality, i]
            if(broke == 1 and quality > 40):
                result_MEDIAN.append([4, broke, quality, i, j]) #risultati divisi per immagine
            elif (broke == 0 and quality > 35):
                f = open('result.txt', 'a')
                f.write('\n\n')
                for d in data:
                    f.write(str(d))
                    f.write(' ')
                f.write(' // ')
                print('brokeeee')
                list_of_broke_dct.append([4, broke, quality, i, j]) 
            if quality < 35:
              break

    return result_MEDIAN

def resizing_test(watermarked, wat_dct, locations, bit_esclusi,n_bit):
    result_Resizing = []
    resized_img = wat_dct.copy()
    for i in np.arange (0.2, 0.6, 0.1):

        attacked_wat = resizing(watermarked, i)
        if(attacked_wat.shape == (512,512)):
            attacked_dct = dct(dct(attacked_wat,axis=0, norm='ortho'),axis=1, norm='ortho').flatten() 

            for ind in locations[bit_esclusi:(n_bit+bit_esclusi)]: 
                resized_img[ind] = attacked_dct[ind]
        
            attacked = idct(idct(resized_img.reshape(512,512),axis=1, norm='ortho'),axis=0, norm='ortho')
        
            cv2.imwrite('attacked.bmp', attacked)
            broke, quality = detection(o_path_detection, w_path_detection, 'attacked.bmp')
            print("quality = ", quality, "param = ",i)
            resized_img = wat_dct.copy()
            data = ['#RESIZING : ', 5, quality, i]
            if(broke == 1 and quality > 40):
                result_Resizing.append([5, broke, quality, i]) #risultati divisi per immagine
            elif (broke == 0 and quality > 35):
                f = open('result.txt', 'a')
                f.write('\n\n')
                for d in data:
                    f.write(str(d))
                    f.write(' ')
                f.write(' // ')
                print('brokeeee')
                list_of_broke_dct.append([5, broke, quality, i]) 

    return result_Resizing

def jpeg_test(watermarked, wat_dct, locations,bit_esclusi,n_bit):

    result_JPEG = []
    jpeg_img = wat_dct.copy()
    for i in range (1,30,9):
        attacked_wat = jpeg_compression(watermarked, i)
        attacked_dct = dct(dct(attacked_wat,axis=0, norm='ortho'),axis=1, norm='ortho').flatten() 

        for ind in locations[bit_esclusi:(n_bit+bit_esclusi)]: 
            jpeg_img[ind] = attacked_dct[ind]
        
        attacked = idct(idct(jpeg_img.reshape(512,512),axis=1, norm='ortho'),axis=0, norm='ortho')
        
        cv2.imwrite('attacked.bmp', attacked)
        broke, quality = detection(o_path_detection, w_path_detection, 'attacked.bmp')
        print("quality = ", quality, "param = ",i)
        jpeg_img = wat_dct.copy()
        data = ['#JPEG : ', 6, quality, i]
        if(broke == 1 and quality > 40):
          result_JPEG.append([6, broke, quality, i]) #risultati divisi per immagine
        elif (broke == 0 and quality > 35):
          f = open('result.txt', 'a')
          f.write('\n\n')
          for d in data:
            f.write(str(d))
            f.write(' ')
          f.write(' // ')
          print('brokeeee')
          list_of_broke_dct.append([6, broke, quality, i]) 

    return result_JPEG

def run_base_test(watermarked,wat_dct, locations, bit_esclusi,n_bit):
    
    all_test_check = []

    print('----------------- AWGN --------------------')
    result_awg = awgn_test(watermarked, wat_dct, locations, bit_esclusi,n_bit)
    all_test_check.extend(result_awg)

    print('----------------- BLUR --------------------')
    result_blur = blur_test(watermarked, wat_dct, locations, bit_esclusi,n_bit)
    all_test_check.extend(result_blur)

    print('----------------- SHARPENING --------------------')
    result_sharpening = sharpening_test(watermarked, wat_dct, locations, bit_esclusi,n_bit)
    all_test_check.extend(result_sharpening)

    print('----------------- MEDIAN --------------------')
    result_median = median_test(watermarked, wat_dct, locations, bit_esclusi,n_bit)
    all_test_check.extend(result_median)

    print('----------------- RESIZING --------------------')
    result_resizing = resizing_test(watermarked, wat_dct, locations, bit_esclusi,n_bit)
    all_test_check.extend(result_resizing)

    print('----------------- JPEG --------------------')
    result_jpeg = jpeg_test(watermarked, wat_dct, locations, bit_esclusi,n_bit)
    all_test_check.extend(result_jpeg)    

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

def double_attack(result_tests, watermarked,wat_dct, locations, bit_esclusi,n_bit):
    print('----------------- Double --------------------')

    f = open('result.txt', 'a')
    f.write('\n\n')
    safe_wat = watermarked.copy()
    safe = wat_dct.copy()
    if (len(result_tests)>0):
      for i in range(len(result_tests)):
          attacked_wat = attack_selection(result_tests[i], safe_wat )
          saf_att = attacked_wat.copy()
          for k in range(i, len(result_tests)):
            attacked_wat_2 = attack_selection(result_tests[k], saf_att)

            attacked_dct = dct(dct(attacked_wat_2,axis=0, norm='ortho'),axis=1, norm='ortho').flatten() 
            for ind in locations[bit_esclusi:(n_bit+bit_esclusi)]: 
                safe[ind] = attacked_dct[ind]
            
            attacked = idct(idct(safe.reshape(512,512),axis=1, norm='ortho'),axis=0, norm='ortho')
            cv2.imwrite('attacked.bmp', attacked)
            broke, quality = detection(o_path_detection, w_path_detection, 'attacked.bmp')
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
                  list_of_broke_dct.append([result_tests[i][0], result_tests[i][3], result_tests[i][4] , broke, quality, result_tests[k][0], result_tests[k][3], result_tests[k][4]])
                  data = [result_tests[i][0], result_tests[i][3], result_tests[i][4] , quality, result_tests[k][0], result_tests[k][3], result_tests[k][4]]
                  for d in data:
                    f.write(str(d))
                    f.write(' ')
                  f.write(' // ')
                else:
                  list_of_broke_dct.append([result_tests[i][0], result_tests[i][3], result_tests[i][4], broke, quality, result_tests[k][0], result_tests[k][3]])
                  data = [result_tests[i][0], result_tests[i][3], result_tests[i][4], quality, result_tests[k][0], result_tests[k][3]]
                  for d in data:
                    f.write(str(d))
                    f.write(' ')
                  f.write(' // ')
              elif(len(result_tests[k]) > 4):
                list_of_broke_dct.append([result_tests[i][0], result_tests[i][3], broke, quality,result_tests[k][0], result_tests[k][3], result_tests[k][4] ])
                data = [result_tests[i][0], result_tests[i][3], quality,result_tests[k][0], result_tests[k][3], result_tests[k][4] ]
                for d in data:
                    f.write(str(d))
                    f.write(' ')
                f.write(' // ')
              else:
                list_of_broke_dct.append([result_tests[i][0], result_tests[i][3], broke, quality,result_tests[k][0], result_tests[k][3]])
                data = [result_tests[i][0], result_tests[i][3], quality,result_tests[k][0], result_tests[k][3]]
                
                for d in data:
                    f.write(str(d))
                    f.write(' ')
                f.write(' // ')

              f.write('\n\n')


            saf_att = attacked.copy()
          safe = wat_dct.copy()
          safe_wat = watermarked.copy()

def attacco_dct(watermarked,wat_dct,locations,bit_esclusi,n_bit):


    result_test = run_base_test(watermarked, wat_dct, locations, bit_esclusi,n_bit)

    double_attack(result_test, watermarked,wat_dct, locations, bit_esclusi,n_bit)
