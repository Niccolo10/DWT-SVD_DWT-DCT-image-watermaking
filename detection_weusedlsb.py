import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
import pywt.data
from scipy.fft import dct, idct
from PIL import Image


def similarity(X,X_star):
    #Computes the similarity measure between the original and the new watermarks.
    s = np.sum(np.multiply(X, X_star)) / np.sqrt(np.sum(np.multiply(X_star, X_star)))

    return s

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



def mixed_extraction(watermarked, original,  alpha_dct = 10, alpha_svd = 4):

    loc_dct_lv1 = 1 
    loc_dct_lv2 = 1 
    loc_svd_lv1 = 0 
    loc_svd_lv2 = 0

    seq_0_dct = [0.24408944605607918, 0.1476558114426969, 0.05750293895971759, 0.16609520545175438, 0.03052366059070033, 0.35126985893137375, 0.7071542455281019, 0.04888803145816256]
    seq_1_dct = [0.3847442219089656, 0.03400997379723547, 0.7222368081836926, 0.26513711527181616, 0.508999665821366, 0.034590062745215366, 0.357950535449292, 0.3267616515773091]

    seq_0_svd = [1.0804572654610185, 0.02786209305450052] 
    seq_1_svd = [0.02786209305450052, 1.0804572654610185] 

    extra_mark_dct = np.array([0]*1024).reshape(32,32)
    extra_mark_svd = np.array([0]*1024).reshape(32,32)
    
    #primo livello di wavelet per original e watermarked
    w_coefficient = pywt.dwt2(watermarked, wavelet='haar')
    w_quadrants = [w_coefficient[0],*w_coefficient[1]]

    o_coefficient = pywt.dwt2(original, wavelet='haar')
    o_quadrants = [o_coefficient[0],*o_coefficient[1]]
    

    #secondo livello di wavelet differenziato per svd e dct
    w_coefficient2_dct = pywt.dwt2(w_quadrants[loc_dct_lv1], wavelet='haar')
    w_quadrants2_dct = [w_coefficient2_dct[0],*w_coefficient2_dct[1]]
    w_coefficient2_svd = pywt.dwt2(w_quadrants[loc_svd_lv1], wavelet='haar')
    w_quadrants2_svd = [w_coefficient2_svd[0],*w_coefficient2_svd[1]]

    o_coefficient2_dct = pywt.dwt2(o_quadrants[loc_dct_lv1], wavelet='haar')
    o_quadrants2_dct = [o_coefficient2_dct[0],*o_coefficient2_dct[1]]
    o_coefficient2_svd = pywt.dwt2(o_quadrants[loc_svd_lv1], wavelet='haar')
    o_quadrants2_svd = [o_coefficient2_svd[0],*o_coefficient2_svd[1]]

    w_coefficient3_svd = pywt.dwt2(w_quadrants2_svd[loc_svd_lv1], wavelet='haar')
    w_quadrants3_svd = [w_coefficient3_svd[0],*w_coefficient3_svd[1]]
    o_coefficient3_svd = pywt.dwt2(o_quadrants2_svd[loc_svd_lv1], wavelet='haar')
    o_quadrants3_svd = [o_coefficient3_svd[0],*o_coefficient3_svd[1]]

    size = w_quadrants2_dct[0].shape[0]
    size_svd = w_quadrants3_svd[1].shape[0]
    
    #dividiamo in blocchi i quadranti scelti
    w_blocks_dct = w_quadrants2_dct[loc_dct_lv2]
    o_blocks_dct = o_quadrants2_dct[loc_dct_lv2]

    w_blocks_svd = w_quadrants3_svd[loc_svd_lv2]
    o_blocks_svd = o_quadrants3_svd[loc_svd_lv2]
   
    w_blocks_dct = np.hsplit(w_blocks_dct, size//4)
    o_blocks_dct = np.hsplit(o_blocks_dct, size//4)

    w_blocks_svd = np.hsplit(w_blocks_svd, size_svd//2)
    o_blocks_svd = np.hsplit(o_blocks_svd, size_svd//2)

    for k in range(len(w_blocks_dct)):
      w_blocks_dct[k] = np.vsplit(w_blocks_dct[k], size//4)
      w_blocks_svd[k] = np.vsplit(w_blocks_svd[k], size_svd//2)
      o_blocks_dct[k] = np.vsplit(o_blocks_dct[k], size//4)
      o_blocks_svd[k] = np.vsplit(o_blocks_svd[k], size_svd//2)

    #dct & embedding 

    exctd_seq_dct = [0]*8
    exctd_seq_svd = [0]*2
    for i in range(len(w_blocks_dct)):
      for j in range(len(w_blocks_dct)):
          w_blocks_dct[i][j] = (dct(dct(w_blocks_dct[i][j],axis=0, norm='ortho'),axis=1, norm='ortho')).flatten() 
          o_blocks_dct[i][j] = (dct(dct(o_blocks_dct[i][j],axis=0, norm='ortho'),axis=1, norm='ortho')).flatten() 
          _,w_S,_ = np.linalg.svd(w_blocks_svd[i][j])
          _,o_S,_ = np.linalg.svd(o_blocks_svd[i][j])
          abs_dct = o_blocks_dct[i][j].copy() 
          abs_dct = abs(abs_dct)
          locations = np.argsort(-abs_dct,axis=None)

          for t,pos in enumerate(locations[1:9]):
            exctd_seq_dct[t] = (w_blocks_dct[i][j]-o_blocks_dct[i][j])[pos]
        
        #EXTRACTION DALLA DCT
          if np.linalg.norm(exctd_seq_dct) > 1e-8 :
            seq_0_corr_dct = np.corrcoef(exctd_seq_dct,alpha_dct*np.array(seq_0_dct))[0][1]
            seq_1_corr_dct = np.corrcoef(exctd_seq_dct,alpha_dct*np.array(seq_1_dct))[0][1]

            if(seq_0_corr_dct >= seq_1_corr_dct):  
               extra_mark_dct[i][j] = 0
            else:
               extra_mark_dct[i][j] = 1

          else: #caso original=watermarked
            extra_mark_dct[i][j] = random.randint(0,1)

          exctd_seq_svd = w_S - o_S
          if 1e-8 < np.linalg.norm(exctd_seq_svd) < 30:
            diff_svd0 = exctd_seq_svd - alpha_svd*np.array(seq_0_svd)
            diff_svd1=exctd_seq_svd-alpha_svd*np.array(seq_1_svd)
         
            seq_0_corr_svd = np.linalg.norm(diff_svd0)
            seq_1_corr_svd = np.linalg.norm(diff_svd1) 
          
            if(seq_0_corr_svd <= seq_1_corr_svd ): 
               extra_mark_svd[i][j] = 0
            else:
               extra_mark_svd[i][j] = 1
          else: #caso watermarked = original
            extra_mark_svd[i][j] = random.randint(0,1)
            
    error_tr = 30 #tolleranza nel considerare un immagine totalmente bianca o totalmente nera
#se il marchio estratto è quasi tutto bianco(o nero) ritorna un marchio randomico
    random_mark = np.random.uniform(0.0,1.0,1024)
    random_mark = np.uint8(np.rint(random_mark)).reshape(32,32)
    if np.sum(extra_mark_dct) < error_tr or np.sum(extra_mark_dct) > (1024-error_tr):
      extra_mark_dct =  random_mark.copy()
    if np.sum(extra_mark_svd) < error_tr or np.sum(extra_mark_svd) > (1024-error_tr):
      extra_mark_svd =  random_mark.copy() 
    
    return extra_mark_dct, extra_mark_svd



def detection(original, watermarked, attacked):
    original = cv2.imread(original,0)
    watermarked = cv2.imread(watermarked,0)
    attacked = cv2.imread(attacked,0)

    threshold = 12.3

    quality = wpsnr(watermarked, attacked)
    w_ex1, w_ex2 = mixed_extraction(watermarked, original)
    
    a_ex1, a_ex2 = mixed_extraction(attacked, original)
 
    sim1 = similarity(w_ex1, a_ex1)
    sim2 = similarity(w_ex1, a_ex2)
    s = max(sim1,sim2)

    if s >= threshold:
        return 1, quality
    else:
        return 0, quality
