import numpy as np
import random
from sklearn.metrics import roc_curve, auc
from detection_weusedlsb import *

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
  attacked = rescale(img, scale, preserve_range = True)
  attacked = rescale(attacked, 1/scale, preserve_range = True)
  attacked = attacked[:x, :y]
  return attacked

def jpeg_compression(img, QF):
  from PIL import Image
  img = Image.fromarray(img)
  img.save('tmp.jpg',"JPEG", quality=QF)
  attacked = Image.open('tmp.jpg')
  attacked = np.asarray(attacked,dtype=np.uint8)
  os.remove('tmp.jpg')

  return attacked

def random_attack(img):
  i = random.randint(1,6)
  attacked = img
  if i==1:
    attacked = awgn(img, 5.0, 123)
  elif i==2:
    #attacked = blur(img, [3, 2])
    attacked = blur(img, [1, 1])
  elif i==3:
    attacked = sharpening(img, 1, 1)
  elif i==4:
    attacked = median(img, [3, 5])
  elif i==5:
    attacked = resizing(img, 0.5)
  elif i==6:
    attacked = jpeg_compression(img, 75)
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

def similarity(X,X_star):
    #Computes the similarity measure between the original and the new watermarks.

    s1 = np.sum(np.multiply(X, X_star)) / np.sqrt(np.sum(np.multiply(X_star, X_star)))
    return s1

mark_size = 1024

ori_mark = np.load('1666596647808_weusedlsb.npy')
ori_mark = np.uint8(np.rint(ori_mark))
ori_mark = ori_mark.reshape(32, 32)

np.random.seed(seed=123)
scores = []
labels = []
ext = 'bmp'
files = ['%04d.%s' % (i, ext) for i in range(100)]

for im in files:
  img = cv2.imread('%s' % im, 0)
  sample = 0
  watd = mixed_embedding(img,ori_mark)
  e_wat1,e_wat2 = mixed_extraction(watd, img)

  while sample<20:
    
    #fakemark is the watermark for H0 
    fakemark = np.random.uniform(0.0, 1.0, mark_size)
    fakemark = np.uint8(np.rint(fakemark))
    #random attack to watermarked image
    res_att = random_attack(watd)

    #extract attacked watermark 
    
    a_wat1,a_wat2 = mixed_extraction(res_att, img)

    sim = max(similarity(e_wat1, a_wat1), similarity(e_wat1, a_wat2))
    if np.isclose(sim,similarity(e_wat1,a_wat2)):
      a_wat = a_wat2
    else:
      a_wat = a_wat1    
    
    scores.append(similarity(ori_mark,a_wat))
    labels.append(1)
    #compute similarity H0
    scores.append(similarity(fakemark.reshape(32,32),a_wat))
    labels.append(0)
    sample += 1

#compute ROC
fpr, tpr, tau = roc_curve(np.asarray(labels), np.asarray(scores), drop_intermediate=False)
#compute AUC
roc_auc = auc(fpr, tpr)
idx_tpr_1 = np.where((fpr-0.05)==min(i for i in (fpr-0.05) if i > 0))
idx_tpr_2 = np.where((fpr-0.1)==min(i for i in (fpr-0.1) if i > 0))
idx_tpr_3 = np.where((fpr-0.037)==min(i for i in (fpr-0.037) if i > 0))

ind = 0
for j in range(10):
    ind += .01
    idx_tpr_3 = np.where((fpr-ind)==min(i for i in (fpr-ind) if i > 0))
    print('For a FPR approximately equals to ', ind, ' corresponds a threshold equals to %0.2f' % tau[idx_tpr_3[0][0]])
plt.figure()
lw = 2

plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='AUC = %0.2f' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()