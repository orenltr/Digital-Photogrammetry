
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from tkinter.filedialog import askopenfilenames

# #%%


def sampleGroundTruth(img, number_of_samples):

    # sampling object and background pixels

    plt.title('sample ',number_of_samples/2, ' points of the object', fontweight="bold")
    plt.imshow(img2)
    object_points = plt.ginput(number_of_samples/2)
    object_points = np.array(object_points, dtype=int)
    print(object_points)
    plt.close()

    plt.title('sample ',number_of_samples/2, 'points of the background', fontweight="bold")
    plt.imshow(img2)
    background_points = plt.ginput(number_of_samples/2)
    background_points = np.array(background_points, dtype=int)
    print(background_points)

    # building a mask of the samples pixels where
    # object pixels = 1 and background pixels = 2
    sampled_pixels = np.zeros(img.shape,dtype=np.uint8)
    sampled_pixels[object_points[:, 1], object_points[:, 0]] = 1
    sampled_pixels[background_points[:, 1], background_points[:, 0]] = 2

    plt.close()

    return sampled_pixels.reshape(-1)


def generateGaborFiltersBank():

    kernels = []  # holds the kernels we will generate
    for theta in range(8):  # define number of thetas (orientation of the filter)
        theta = theta / 8 * np.pi
        for sigma in (1, 3, 5, 7):  # define number of sigmas (std of the gaussian envelope)
            for lamda in np.arange(np.pi / 4, np.pi, np.pi / 4):  # define number of wavelengths
                for gamma in (0.05, 0.5):  # define number of gammas
                    ksize = 9
                    psi = 0  # phase offset
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, psi, ktype=cv2.CV_32F)
                    kernels.append(kernel)
    return kernels


def correctlyClassified(df, sampled_pixels):
    # classify the pixels using means with k=2
    kmeans = KMeans(n_clusters=2).fit(df.values[sampled_pixels > 0])
    classified_correctly1= np.sum(kmeans.labels_ == (sampled_pixels[sampled_pixels > 0]-1))
    classified_correctly2 = np.sum(1-kmeans.labels_ == (sampled_pixels[sampled_pixels > 0]-1))
    return max(classified_correctly1, classified_correctly2)

if __name__ == '__main__':

    # load image
    # img_path = askopenfilenames(title='Select Input File')
    img_name = '20201121_143402'  # 20201121_143330, 20201121_143257, 20201121_143402
    img = cv2.imread(img_name+'.jpg')
    img = cv2.resize(img, (200, 300))  # resize image to ease calculations

    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # convert to gray scale
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#%%
    # create data frame in which each column is a filter and each row is a pixel
    df = pd.DataFrame()

    # save original image as a vector and insert to df
    vec_img2 = img2.reshape(-1)
    df['Original Image'] = vec_img2

    # sample ground truth of the object and the background
    # use to determine if a filter contribute to the segmentation
    # use also the evaluate the accuracy of the resulting segmentation
    number_of_samples = 100
    sampled_pixels = sampleGroundTruth(img2,number_of_samples)

    # generate filter bank
    kernels = generateGaborFiltersBank()

    classified_correctly = 0  # keeps the number of correctly classified pixels (from the sampled pixels)

    # generate Gabor features
    for i, kernel in enumerate(kernels):

        gabor_label = 'Gabor' + str(i)  # label of column in the data frame
        # filter the image
        filtered_image = cv2.filter2D(img2, cv2.CV_8UC3, kernel)
        # add the filtered image as vector to data frame
        vec_filtered_img2 = filtered_image.reshape(-1)
        df[gabor_label] = vec_filtered_img2

        classified_correctly_new = correctlyClassified(df, sampled_pixels)
        if classified_correctly_new > classified_correctly:
            classified_correctly = classified_correctly_new
            accuracy_rate = classified_correctly/number_of_samples *100
            print(accuracy_rate, ' % of pixels been classified correctly')
            # save the filtered img
            cv2.imwrite('gabor_filtered_images/' + img_name + '/' + gabor_label + '.jpg', filtered_image)

        else:
            df.drop(columns=gabor_label, inplace=True)

        # plt.imshow(filtered_image)
        # plt.show()



    df.head()
    df.to_csv(img_name+'.csv', index=False)
#%%

    df = pd.read_csv(img_name+'.csv')
    df['sum'] = np.sum(df.values,axis=1)

    # classify the pixels using Kmeans with k=2
    kmeans = KMeans(n_clusters=2).fit(df.values)

    # creating mask from the classification
    mask = np.uint8(kmeans.labels_.reshape(img2.shape))
    mask = cv2.GaussianBlur(mask, (9, 9), 0)  # blur mask to get rid of "noise" in the segmentation
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

    # cv2.imwrite('segmented_results/'+img_name+'.jpg', segmented_img)
    fig.savefig('segmented_results/'+img_name+'_kernel_9.jpg')