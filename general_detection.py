from traceback import print_tb
from unittest import result
from cv2 import imread, imwrite
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt
from pip import main
from scipy.fft import dct, idct
import cv2
from numpy import cov
from statistics import stdev
import pywt
import pywt.data
from path_selection import *
from wavelet_attack import *
from attacks import *
from dct_attack import *
from fft_attack import *


def attacco_wavelet_detection():

    #cerca i quadranti della wavelet più promettenti, se ce ne sono, -1 se non li trova

    original = cv2.imread(o_path_detection,0) 
    watermarked =  cv2.imread(w_path_detection,0)
     #primo livello di wavelet per original e watermarked

    w_coefficient = pywt.dwt2(watermarked, wavelet='haar')
    w_quadrants = [w_coefficient[0],*w_coefficient[1]]

    o_coefficient = pywt.dwt2(original, wavelet='haar')
    o_quadrants = [o_coefficient[0],*o_coefficient[1]]

    diff = []
    norms = []
    for i in range(4):
        diff.append(w_quadrants[i]-o_quadrants[i])
        norms.append(np.linalg.norm(diff[i]))

    print(norms)
    if np.var(norms)>.5:
        quad_lv1 = np.argmax(norms)
        
        #secondo livello di wavelet differenziato per svd e dct
    
        w_coefficient2 = pywt.dwt2(w_quadrants[quad_lv1], wavelet='haar')
        w_quadrants2 = [w_coefficient2[0],*w_coefficient2[1]]

        o_coefficient2 = pywt.dwt2(o_quadrants[quad_lv1], wavelet='haar')
        o_quadrants2 = [o_coefficient2[0],*o_coefficient2[1]]

        diff2 = []
        norms2 = []
        for i in range(4):
            diff2.append(w_quadrants2[i]-o_quadrants2[i])
            norms2.append(np.linalg.norm(diff2[i]))

        if np.var(norms2)>.5:
            quad_lv2 = np.argmax(norms2)
            print(norms2)
            return quad_lv1, quad_lv2, w_quadrants, w_quadrants2
        else:
            return quad_lv1, -1, w_quadrants, -1

    else:
        return -1,-1,-1,-1

def attacco_dct_detection():

    original = cv2.imread(o_path_detection,0) 
    watermarked =  cv2.imread(w_path_detection,0)

    o_dct = dct(dct(original,axis=0, norm='ortho'),axis=1, norm='ortho').flatten()           
    w_dct = dct(dct(watermarked,axis=0, norm='ortho'),axis=1, norm='ortho').flatten()           

    diff = w_dct-o_dct
    abs_dct = diff.copy()
    abs_dct = abs(abs_dct)
    locations = np.argsort(-abs_dct,axis=None) # - sign is used to get descending order
    values = -np.sort(-abs_dct,axis = None)

    if np.mean(values[0:1024]) > np.mean(values[1024:])+10: 
        #primo check: hanno usato proprio SS?
        print("SS detected")
        return watermarked,w_dct,locations,0,1024

    elif np.mean(values[0:5000]) > np.mean(values[5000:])+2.5:
        #secondo check: niente SS, ma il loro embedding ha comunque a che fare con dct?
        print("dct like emb detected")
        return watermarked,w_dct,locations,0,5000
    
    else:
        return [0],[0],[0],[0],[0]

def attacco_ftt_detection():

    original = cv2.imread(o_path_detection,0) 
    watermarked =  cv2.imread(w_path_detection,0)

    o_fft = np.fft.fftshift(np.fft.fft2(original)).flatten()      
    w_fft = np.fft.fftshift(np.fft.fft2(watermarked)).flatten()        
    diff = w_fft-o_fft
    abs_fft = diff.copy()
    abs_fft= abs(abs_fft)
    locations = np.argsort(-abs_fft,axis=None) # - sign is used to get descending order
    values = -np.sort(-abs_fft,axis = None)

    if np.mean(values[0:1024]) > np.mean(values[1024:])+2000: 
        #primo check: hanno usato proprio SS in DFT?
        print("SS fft detected")
        return watermarked,w_fft,locations,0,1024

    elif np.mean(values[0:5000]) > np.mean(values[5000:])+1200:
        #secondo check: niente SS, ma il loro embedding ha comunque a che fare con FFT?
        print("fft like emb detected")
        return watermarked,w_fft,locations,0,5000
    
    else:
        return [0],[0],[0],[0],[0]

def main():


    quad_lv1, quad_lv2, w_quadrants, w_quadrants2 = attacco_wavelet_detection()
    watermarked, w_dct,locations, bit_esclusi,n_bit = attacco_dct_detection()
    watermarked_ftt, w_ftt,locations_ftt, bit_esclusi_ftt,n_bit_ftt = attacco_ftt_detection()


    if(quad_lv1 != -1):
        
        print('---------wavelet-----------')
        print(quad_lv1, quad_lv2)
        attacco_wavelet(quad_lv1, quad_lv2, w_quadrants, w_quadrants2)
        print(list_of_broke_wavelet)

    if(n_bit != [0]):

        print('---------dct-----------')
        attacco_dct(watermarked, w_dct, locations, bit_esclusi, n_bit)
        print(list_of_broke_dct)

    if(n_bit_ftt != [0]):

        print('---------ftt-----------')
        attacco_fft(watermarked_ftt, w_ftt,locations_ftt, bit_esclusi_ftt,n_bit_ftt)
        print(list_of_broke_fft)

    print('---------normal-----------')
    normal_attack(o_path_detection, w_path_detection)
    print(list_of_broke_attacks)

if __name__=="__main__":

    main()
