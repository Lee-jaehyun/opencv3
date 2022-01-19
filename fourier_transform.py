import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


#주석추가
image = cv2.imread("../../../Desktop/cnn/lenna.png")


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
height, width = gray.shape

crow, ccol = int(width/2), int(height/2)

dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

#dft_shift[crow-30 : crow+30, ccol-30 : ccol+30] = 0

out = 20*np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

f_ishift = np.fft.ifftshift(dft_shift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

plt.subplot(121)
plt.imshow(gray, cmap='gray')
plt.subplot(122)
plt.imshow(out, cmap='gray')

plt.imshow(img_back)

plt.show()

#######
#wow###
#######
