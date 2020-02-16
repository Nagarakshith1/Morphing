'''
  File name: helpers.py
  Author:
  Date created:
'''

'''
  File clarification:
    Helpers file that contributes the project
    You can design any helper function in this file to improve algorithm
'''
import scipy.misc
def imresize(im1,im2):
    y1 = im1.shape[0]
    x1 = im1.shape[1]

    im2 = scipy.misc.imresize(im2, [y1,x1])
    return im1,im2
