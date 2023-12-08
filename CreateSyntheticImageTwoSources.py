#2023-12-08 Fabian Reuter
# This script generates a 16-bit grayscale synthetic test image for evaluation using WaFE
# It models two shock sources emitting at different times (given by the radii Radius1 und Radius2 and the second exposures RadiusIm2_1 and RadiusIm2_2)
# and using Gaussian intensity profiles of the shock fronts of standard deviation sigma

import numpy as np
import math
import matplotlib.pyplot as plt
import tifffile

SavePath="C:\WaFE\SyntheticImages/"

sigma = 1 #width of wave front ( Gaussian intensity profile)
Radius1=20 #First exposure of first shock front
Radius2=50 #Second exposure of first shock front

RadiusIm2_1=17 #First exposure of second shock front
RadiusIm2_2=47 #Second exposure of second shock front

CenterPoint2_xOffset=0
CenterPoint2_yOffset=20

image = np.ones((250, 400), dtype=np.uint16) * 65535


center_x = image.shape[1] // 2
center_y = image.shape[0] // 2


x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
#Calculate the distance from the center for each pixel

distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
#Create the intensity profile using a Gaussian function


intensity_profile = np.exp(-(distance - Radius1) ** 2 / (2 * sigma ** 2)) + np.exp(-(distance - Radius2) ** 2 / (2 * sigma ** 2))
#Normalize the intensity profile to the range [0, 1]

intensity_profile = (intensity_profile - np.min(intensity_profile)) / (np.max(intensity_profile) - np.min(intensity_profile))
#Apply the intensity profile to the image

image = (image * intensity_profile).astype(np.uint16)

#invert
image=65535-image
#reduce dynamic range
image=image/4
image=image+65400/8

#----------- image2
image2 = np.ones((250, 400), dtype=np.uint16) * 65535


center_x = image2.shape[1] // 2+CenterPoint2_xOffset
center_y = image2.shape[0] // 2+CenterPoint2_yOffset


x, y = np.meshgrid(np.arange(image2.shape[1]), np.arange(image2.shape[0]))
#Calculate the distance from the center for each pixel

distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
#Create the intensity profile using a Gaussian function


intensity_profile = np.exp(-(distance - RadiusIm2_1) ** 2 / (2 * sigma ** 2)) + np.exp(-(distance - RadiusIm2_2) ** 2 / (2 * sigma ** 2))
#Normalize the intensity profile to the range [0, 1]

intensity_profile = (intensity_profile - np.min(intensity_profile)) / (np.max(intensity_profile) - np.min(intensity_profile))
#Apply the intensity profile to the image

image2 = (image2 * intensity_profile).astype(np.uint16)

#invert
image2=65535-image2
#reduce dynamic range - allows to add more noise (noise can be added later in a second step to the image)
image2=image2/4
image2=image2+65400/8


image=np.round(image+image2).astype(np.uint16)


DisplayImage=False
if DisplayImage:
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()
    #plot profile
    #plt.plot(image[140,:])
    

tifffile.imwrite(SavePath+"\ArtificialTwoCircles_R1_"+str(Radius1)+"_R2_"+str(Radius2)+"_ZweiterKreis3pxWeniger_Sigma"+str(sigma)+".tiff", image.astype(np.uint16), dtype=np.uint16)

