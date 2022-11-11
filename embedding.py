import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
import pywt.data
from scipy.fft import dct, idct
from PIL import Image

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

#modificata!!
def similarity(X,X_star):
    #Computes the similarity measure between the original and the new watermarks.
    s = np.sum(np.multiply(X, X_star)) / (np.sqrt(np.sum(np.multiply(X_star, X_star)))*(np.sqrt(np.sum(np.multiply(X, X)))))

    s1 = np.sum(np.multiply(X, X_star)) / np.sqrt(np.sum(np.multiply(X_star, X_star)))
    return s1

def padding(block_dct,seq): 
  #ritorna la sequenza di encoding del bit inserita nei bit più significativi della
  #dct ad eccezione del primo

  abs_dct = block_dct.copy()
  abs_dct= abs(abs_dct)
  locations = np.argsort(-abs_dct,axis=None) # - sign is used to get descending order
  pad_seq = [0]*16

  for ind,i in enumerate(locations[1:9]): 
    pad_seq[i] = seq[ind]

  return pad_seq
  #return seq[:2]+[0]*2+seq[2:5]+[0]+seq[5:]+[0]*5


def mixed_embedding(original, mark, seq_0_dct, seq_1_dct, seq_0_svd, seq_1_svd, loc_dct_lv1 = 1, loc_dct_lv2 = 1, loc_svd_lv1 = 0, loc_svd_lv2 = 0, alpha_dct = 10, alpha_svd = 4):
    
    #funzione per embedding misto: inserisce due marchi al secondo livello della wavelet
    #uno tramite dct nei quadranti loc_dct_lv1,loc_dct_lv2, l'altro tramite svd in loc_svd_lv1,loc_svd_lv2
    #seq_0_* e seq_1_* sono le sequenze che codificano rispettivamente i bit 0 e 1 per il metodo *
    #alpha rappresenta l'intensità con cui il marchio viene insierito nell'immagine

    #WARNING! scegliere quadrati diversi tra svd e dct per un corretto funzionamento
    
    #dwt 
    coefficient = pywt.dwt2(original, wavelet='haar')
    quadrants = [coefficient[0],*coefficient[1]]

    coefficient2_dct = pywt.dwt2(quadrants[loc_dct_lv1], wavelet='haar')
    quadrants2_dct = [coefficient2_dct[0],*coefficient2_dct[1]]
    coefficient2_svd = pywt.dwt2(quadrants[loc_svd_lv1], wavelet='haar')
    quadrants2_svd = [coefficient2_svd[0],*coefficient2_svd[1]]

#SVD SU TERZO LIVELLO
    coefficient3_svd = pywt.dwt2(quadrants2_svd[loc_svd_lv1], wavelet='haar')
    quadrants3_svd = [coefficient3_svd[0],*coefficient3_svd[1]]

    size = quadrants2_dct[1].shape[0]
    size_svd = quadrants3_svd[1].shape[0]

    #divisione in blocchi dei quadranti scelti
    
    blocks_dct = quadrants2_dct[loc_dct_lv2]
    blocks_dct = np.hsplit(blocks_dct, size//4)
    blocks_svd = quadrants3_svd[loc_svd_lv2]
    blocks_svd = np.hsplit(blocks_svd, size_svd//2)

    for k in range(len(blocks_dct)):
      blocks_dct[k] = np.vsplit(blocks_dct[k], size//4)
      blocks_svd[k] = np.vsplit(blocks_svd[k], size_svd//2)

    # dct, svd; embedding; idct, isvd 

    for i in range(len(blocks_dct)):
      for j in range(len(blocks_dct)):
          blocks_dct[i][j] = dct(dct(blocks_dct[i][j],axis=0, norm='ortho'),axis=1, norm='ortho')           
          U,S,VH = np.linalg.svd(blocks_svd[i][j])
          if(mark[i][j] == 0):  
             blocks_dct[i][j] += alpha_dct*(np.array(padding(blocks_dct[i][j],seq_0_dct)).reshape(4,4))
             S +=  alpha_svd*(np.array(seq_0_svd)) 
          else:
             blocks_dct[i][j] += alpha_dct*(np.array(padding(blocks_dct[i][j],seq_1_dct)).reshape(4,4))
             S +=  alpha_svd*(np.array(seq_1_svd))

          blocks_svd[i][j] = np.dot(U*S,VH)
          blocks_dct[i][j] = idct(idct(blocks_dct[i][j],axis=1, norm='ortho'),axis=0, norm='ortho')
    
    for k in range(len(blocks_dct)):
      blocks_dct[k] = np.array(np.vstack(blocks_dct[k])).reshape(128,4)
      blocks_svd[k] = np.array(np.vstack(blocks_svd[k])).reshape(64,2)
   
    quadrants2_dct[loc_dct_lv2] = np.array(np.hstack(blocks_dct)).reshape(128,128)
    quadrants3_svd[loc_svd_lv2] = np.array(np.hstack(blocks_svd)).reshape(64,64)

    coefficient3_svd = quadrants3_svd[0],(quadrants3_svd[1],quadrants3_svd[2],quadrants3_svd[3])
    quadrants2_svd[loc_svd_lv1] = pywt.idwt2(coefficient3_svd, wavelet='haar')

    coefficient2_dct = quadrants2_dct[0],(quadrants2_dct[1],quadrants2_dct[2],quadrants2_dct[3])
    coefficient2_svd = quadrants2_svd[0],(quadrants2_svd[1],quadrants2_svd[2],quadrants2_svd[3])
    
    #rimettiamo ogni quadrante al suo posto nel primo livello
    quadrants[loc_dct_lv1] = pywt.idwt2(coefficient2_dct, wavelet='haar')
    quadrants[loc_svd_lv1] = pywt.idwt2(coefficient2_svd, wavelet='haar')
    
    coefficient = quadrants[0],(quadrants[1],quadrants[2],quadrants[3])
    final = pywt.idwt2(coefficient, wavelet='haar')

    return np.uint8(np.rint(np.clip(final, 0, 255)))


def main():

    seq_0_dct = [0.24408944605607918, 0.1476558114426969, 0.05750293895971759, 0.16609520545175438, 0.03052366059070033, 0.35126985893137375, 0.7071542455281019, 0.04888803145816256]
    seq_1_dct = [0.3847442219089656, 0.03400997379723547, 0.7222368081836926, 0.26513711527181616, 0.508999665821366, 0.034590062745215366, 0.357950535449292, 0.3267616515773091]

    seq_0_svd = [1.0804572654610185, 0.02786209305450052] 
    seq_1_svd = [0.02786209305450052, 1.0804572654610185]

    images = ['image1.bmp', 'image2.bmp', 'image3.bmp']
    marked = ['image1_wtd.bmp','image2_wtd.bmp','image3_wtd.bmp']
    
    mark = np.load('weusedlsb.npy')
    mark = mark.reshape(32,32)

    for i in range(0,3):
      im = cv2.imread(images[i], 0)
      watd = mixed_embedding(im, mark, seq_0_dct, seq_1_dct, seq_0_svd, seq_1_svd)
      cv2.imwrite(marked[i], watd)

      print(wpsnr(im,watd))


if __name__=="__main__":    

    main()
