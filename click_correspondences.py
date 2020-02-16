'''
  File name: click_correspondences.py
  Author: 
  Date created: 
'''

'''
  File clarification:
    Click correspondences between two images
    - Input im1: target image
    - Input im2: source image
    - Output im1_pts: correspondences coordiantes in the target image
    - Output im2_pts: correspondences coordiantes in the source image
'''
import matplotlib.pyplot as plt
import numpy as np
from cpselect import cpselect
import helpers

def click_correspondences(im1, im2):

  '''
    Tips:
      - use 'matplotlib.pyplot.subplot' to create a figure that shows the source and target image together
      - add arguments in the 'imshow' function for better image view
      - use function 'ginput' and click correspondences in two images in turn
      - please check the 'ginput' function documentation carefully
        + determine the number of correspondences by yourself which is the argument of 'ginput' function
        + when using ginput, left click represents selection, right click represents removing the last click
        + click points in two images in turn and once you finish it, the function is supposed to 
          return a NumPy array contains correspondences position in two images
  '''
  im1, im2 = helpers.imresize(im1, im2)
  im1_pts, im2_pts = cpselect(im1, im2)
  return im1_pts,im2_pts

if __name__ == "__main__":
  im1 = plt.imread('im1.jpg')
  im2 = plt.imread('im2.jpg')
  im1_pts,im2_pts = click_correspondences(im1,im2)
  np.save('im1_pts', im1_pts)
  np.save('im2_pts', im2_pts)