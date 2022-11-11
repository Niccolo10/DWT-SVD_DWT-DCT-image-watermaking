import os
from traceback import print_tb
from unittest import result
from cv2 import imread, imwrite
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt
from pip import main
from scipy.fft import dct, idct
import random
import cv2
import numpy as np
from numpy import cov
import pywt
import pywt.data
from detection_weusedlsb import *

seq_0_dct = [0.24408944605607918, 0.1476558114426969, 0.05750293895971759, 0.16609520545175438, 0.03052366059070033, 0.35126985893137375, 0.7071542455281019, 0.04888803145816256]
seq_1_dct = [0.3847442219089656, 0.03400997379723547, 0.7222368081836926, 0.26513711527181616, 0.508999665821366, 0.034590062745215366, 0.357950535449292, 0.3267616515773091]

seq_0_svd = [1.0804572654610185, 0.02786209305450052] 
seq_1_svd = [0.02786209305450052, 1.0804572654610185] #, 0.39922671375956564, 4.188354021251055]

list_of_broke_attacks = []

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

def wpsnr(img1, img2):
  from scipy.signal import convolve2d 
  from math import sqrt
  
  img1 = np.float32(img1)/255.0
  img2 = np.float32(img2)/255.0

  difference = img1-img2
  same = not np.any(difference)
  if same is True:
      return 9999999
  csf = np.genfromtxt('csf.csv', delimiter=',')
  ew = convolve2d(difference, np.rot90(csf,2), mode='valid')
  decibels = 20.0*np.log10(1.0/sqrt(np.mean(np.mean(ew**2))))
  return decibels


def awgn_test(real_img, watd):
    result_AWGN = []
    watermarked = cv2.imread(watd, 0)
    awgn_img0 = watermarked.copy()
    for i in range (2,8):

        attacked = awgn(awgn_img0, i, 255)
        cv2.imwrite('attacked.bmp', attacked)
        broke, quality = detection(real_img, watd, 'attacked.bmp')
        awgn_img = watermarked.copy()
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
          list_of_broke_attacks.append([1, broke, quality, i])
          if quality < 35:
            break

    return result_AWGN

def blur_test(real_img, watd):
    result_BLUR = []
    watermarked = cv2.imread(watd, 0)
    blur_image = watermarked.copy()
    for i in np.arange(1, 10, 2):
        for j in np.arange (1, 10, 2):
            attacked = blur(blur_image, [i, j])
            cv2.imwrite('attacked.bmp', attacked)

            broke, quality = detection(real_img, watd, 'attacked.bmp')
            blur_image = watermarked.copy()
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
                list_of_broke_attacks.append([1, broke, quality, i])
            if quality < 35:
              break

    return result_BLUR

def sharpening_test(real_img, watd):

    result_SHARPENING = []
    watermarked = cv2.imread(watd, 0)
    sharp_image = watermarked.copy()
    for i in np.arange(0.2, 0.35, .1):
        for j in np.arange (1,5.5, 0.5):
            attacked = sharpening(sharp_image, i, j)
            cv2.imwrite('attacked.bmp', attacked)
            broke, quality = detection(real_img, watd, 'attacked.bmp')
            sharp_image = watermarked.copy()
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
                list_of_broke_attacks.append([3, broke, quality, i, j]) 
            if quality < 35:
              break

    return result_SHARPENING

def median_test(real_img, watd):
    result_MEDIAN = []
    watermarked = cv2.imread(watd, 0)
    median_image = watermarked.copy()
    for i in range(3,9,2):
        for j in range(3,9,2):
            attacked = median(median_image, [i, j])
            cv2.imwrite('attacked.bmp', attacked)
            broke, quality = detection(real_img, watd, 'attacked.bmp')
            median_image = watermarked.copy()
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
                list_of_broke_attacks.append([4, broke, quality, i, j]) 
            if quality < 35:
              break

    return result_MEDIAN

def resizing_test(real_img, watd):
    result_Resizing = []
    watermarked = cv2.imread(watd, 0)
    resized_img = watermarked.copy()
    for i in np.arange (0.2, 0.8, 0.1):

        attacked = resizing(resized_img, i)
        if(attacked.shape == (512,512)):
          cv2.imwrite('attacked.bmp', attacked)
          broke, quality = detection(real_img, watd, 'attacked.bmp')
          resized_img = watermarked.copy()
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
              list_of_broke_attacks.append([5, broke, quality, i]) 
          if quality < 35:
            break

    return result_Resizing

def jpeg_test(real_img, watd):

    result_JPEG = []
    watermarked = cv2.imread(watd, 0)
    jpeg_img = watermarked.copy()
    for i in range (20,85,20):
        attacked = jpeg_compression(jpeg_img, i)
        cv2.imwrite('attacked.bmp', attacked)
        broke, quality = detection(real_img, watd, 'attacked.bmp')
        jpeg_img = watermarked.copy()
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
          list_of_broke_attacks.append([6, broke, quality, i]) 

    return result_JPEG

def run_base_test(real_img, watd):
    
    all_test_check = []

    print('----------------- AWGN --------------------')
    result_awg = awgn_test(real_img, watd)
    all_test_check.extend(result_awg)

    print('----------------- BLUR --------------------')
    result_blur = blur_test(real_img, watd)
    all_test_check.extend(result_blur)

    print('----------------- SHARPENING --------------------')
    result_sharpening = sharpening_test(real_img, watd)
    all_test_check.extend(result_sharpening)

    print('----------------- MEDIAN --------------------')
    result_median = median_test(real_img, watd)
    all_test_check.extend(result_median)

    print('----------------- RESIZING --------------------')
    result_resizing = resizing_test(real_img, watd)
    all_test_check.extend(result_resizing)

    print('----------------- JPEG --------------------')
    result_jpeg = jpeg_test(real_img, watd)
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
  
def double_attack(real_img, watd, result_tests ):
    print('----------------- Double --------------------')
    watermarked = cv2.imread(watd, 0)

    safe = watermarked.copy()
    if (len(result_tests)>0):
      for i in range(len(result_tests)):
          attacked = attack_selection(result_tests[i], safe )
          saf_att = attacked.copy()
          for k in range(i, len(result_tests)):
            attacked_2 = attack_selection(result_tests[k], saf_att)
            cv2.imwrite('attacked.bmp', attacked_2)
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
                  list_of_broke_attacks.append([result_tests[i][0], result_tests[i][3], result_tests[i][4] , broke, quality, result_tests[k][0], result_tests[k][3], result_tests[k][4]])
                  data = [result_tests[i][0], result_tests[i][3], result_tests[i][4] , quality, result_tests[k][0], result_tests[k][3], result_tests[k][4]]
                  for d in data:
                    f.write(str(d))
                    f.write(' ')
                  f.write(' // ')
                else:
                  list_of_broke_attacks.append([result_tests[i][0], result_tests[i][3], result_tests[i][4], broke, quality, result_tests[k][0], result_tests[k][3]])
                  data = [result_tests[i][0], result_tests[i][3], result_tests[i][4], quality, result_tests[k][0], result_tests[k][3]]
                  for d in data:
                    f.write(str(d))
                    f.write(' ')
                  f.write(' // ')
              elif(len(result_tests[k]) > 4):
                list_of_broke_attacks.append([result_tests[i][0], result_tests[i][3], broke, quality,result_tests[k][0], result_tests[k][3], result_tests[k][4] ])
                data = [result_tests[i][0], result_tests[i][3], quality,result_tests[k][0], result_tests[k][3], result_tests[k][4] ]
                for d in data:
                    f.write(str(d))
                    f.write(' ')
                f.write(' // ')
              else:
                list_of_broke_attacks.append([result_tests[i][0], result_tests[i][3], broke, quality,result_tests[k][0], result_tests[k][3]])
                data = [result_tests[i][0], result_tests[i][3], quality,result_tests[k][0], result_tests[k][3]]
                
                for d in data:
                    f.write(str(d))
                    f.write(' ')
                f.write(' // ')

              f.write('\n\n')


            saf_att = attacked.copy()
          safe = watermarked.copy()

def normal_attack(images, marked):

    result_tests = run_base_test(images, marked)

    double_attack(images, marked, result_tests)

def main():

    images = '0000.bmp'
    marked = 'watermarked0.bmp'
    
    result_tests = run_base_test(images, marked)

    double_attack(images, marked, result_tests)

    print(list_of_broke_attacks)



if __name__=="__main__":

    main()
