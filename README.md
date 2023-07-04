# DWT-SVD_DWT-DCT-image-watermaking
Embedding of a watermak using DWT-SVD and a DWT-DCT tranform.

This is a watermarking method that utilise two different mark embedding:

-The first use a two-level wavelet transform in the HL/HL quadrant, where is aapplied a DCT transform. Then the bit are sorted to select the most significant one, and the mark is embedded, using a 8 bit long code to represent a single bit of the mark.

-The second use a three-level wavelet transform in LL/LL/LL, where is applied a SVD. in this case the single bit of the mark correspond to a 2 bit long code.

The embedding method is created to embedd the mark in 3 different images at the same time, and utilize a csv file for the wpsnr function.
The detection method, given the original image, the watermarked image and the image after being attacked, return if the mark is still present in the attacked image, and the wpsnr of this.(this method can be easy modified for detect the presence of the mark in any given image).

The attacks file contain different Brute-force attacks for destroy the mark on the images:

-Base-attacks is a brute-force method on the spatial domain, that attacks with different single method (AWGN;BLUR;SHARPENINg,MEDIAN-FILTER,RESIZING,JPEG) the given image, and the select the best attaks beetween a list of successfull attacks , and try to attack again the image with combined attacks.

-Wavelet_attack act the same as Base_attacks, with the different that in this case the attack is localized in the area of the dwt transform where the mark seems to be, after a comparison beetween the original and the watermarked image.

-Ftt_attack and Dct_attack works the same as the other attacks, with the difference that in this case the attacks is localized on the DCT or FTT most significative bit, where the mark seems to be , after a comparison beetween the original and the watermarked image. 

General_detection is the file that launch all the attacks, after a control on which frequency domain the mark seems to be embedd.

Path_selection simply is a folder where to write the path to the original and the watermarked image, that is accessible to all the other fucntions.
