
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# #%%
# load image
img_name = '20201121_143402' # 20201121_143330, 20201121_143257, 20201121_143402
img = cv2.imread(img_name+'.jpg')
img = cv2.resize(img, (200, 300))

img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# convert to gray scale
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# create data frame in which each column is a filter and each row is a pixel
df = pd.DataFrame()
# save original image as a vector and insert to df
vec_img2 = img2.reshape(-1)
df['Original Image'] = vec_img2

plt.imshow(img2)
plt.show()

# generate Gabor features
# num = 1  # count numbers in order to give Gabor features labels
# kernels = []  # holds the kernels we will generate
# for theta in range(8):  # define number of thetas (orientation of the filter)
#     theta = theta/8 * np.pi
#     for sigma in (1, 3, 5, 7):  # define number of sigmas (std of the gaussian envelope)
#         for lamda in np.arange(np.pi/4, np.pi, np.pi/4):  # define number of wavelengths
#             for gamma in (0.05, 0.5):  # define number of gammas
#
#                 gabor_label = 'Gabor' + str(num)  # label of column in the data frame
#                 ksize = 10
#                 psi = 0  # phase offset
#                 kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, psi, ktype=cv2.CV_32F)
#                 kernels.append(kernel)
#                 plt.imshow(kernel)
#                 plt.show()
#                 cv2.imwrite('filters/' + gabor_label + '.jpg', kernel)
#
#                 # filter the image
#                 filtered_image = cv2.filter2D(img2, cv2.CV_8UC3, kernel)
#                 # add the filtered image as vector to data frame
#                 vec_filtered_img2 = filtered_image.reshape(-1)
#                 df[gabor_label] = vec_filtered_img2
#
#                 plt.imshow(filtered_image)
#                 plt.show()
#                 cv2.imwrite('gabor_filtered_images/'+img_name+'/'+gabor_label+'.jpg', filtered_image)
#
#                 num += 1
#
#
# df.head()
# df.to_csv(img_name+'.csv', index=False)
#%%
# classify the data frame using Kmeans with k=2
df = pd.read_csv(img_name+'.csv')
kmeans = KMeans(n_clusters=2).fit(df.values)

# creating mask from the classification
mask = np.uint8(kmeans.labels_.reshape(img2.shape))
segmented_img = cv2.bitwise_and(img_RGB,img_RGB,mask=mask)
mask_inverse = np.uint8(np.ones(mask.shape)-mask)  # creating the inverse of the mask
segmented_img_inverse = cv2.bitwise_and(img_RGB,img_RGB,mask=mask_inverse)

# plotting the img and the img segments
fig = plt.figure()
fig.add_subplot(1,3,1)
plt.imshow(img_RGB)
plt.title('Original')
plt.axis(False)

fig.add_subplot(1,3,2)
plt.imshow(segmented_img)
plt.title('Segment 1')
plt.axis(False)
fig.add_subplot(1,3,3)

plt.imshow(segmented_img_inverse)
plt.title('Segment 2')
plt.axis(False)
plt.show()

# cv2.imwrite('segmented_results/'+img_name+'_50_filters.jpg', segmented_img)
fig.savefig('segmented_results/'+img_name+'.jpg')