'''
  File name: morph_tri.py
  Author:
  Date created:
'''

'''
  File clarification:
    Image morphing via Triangulation
    - Input im1: target image
    - Input im2: source image
    - Input im1_pts: correspondences coordiantes in the target image
    - Input im2_pts: correspondences coordiantes in the source image
    - Input warp_frac: a vector contains warping parameters
    - Input dissolve_frac: a vector contains cross dissolve parameters

    - Output morphed_im: a set of morphed images obtained from different warp and dissolve parameters.
                         The size should be [number of images, image height, image Width, color channel number]
'''

from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import numpy as np
import helpers
import interp
import imageio
import time

def get_abcmat(tri,point_list):
  ntri = tri.shape[0]
  trif = tri.flatten().astype(int)
  tri_all_cor = point_list[trif, :]
  one_matrix = np.ones((tri_all_cor.shape[0], 1))
  tri_all_cor_man = np.concatenate((tri_all_cor, one_matrix), axis=1)
  tri_all_cor_man_t = np.transpose(tri_all_cor_man)
  matrix_list = np.array(np.hsplit(tri_all_cor_man_t, ntri))
  return matrix_list

def get_value(A_mat,t_list,bary_mat,im):
  im_red = np.copy(im[:, :, 0])
  im_green = np.copy(im[:, :, 1])
  im_blue = np.copy(im[:, :, 2])
  nr = im.shape[0]
  nc = im.shape[1]

  cor_im = np.matmul(A_mat[t_list,:,:],bary_mat)
  x_cor_im = cor_im[:,0,:].flatten().reshape(nr,nc)
  y_cor_im = cor_im[:, 1, :].flatten().reshape(nr,nc)

  clist_red = interp.interp2(im_red,x_cor_im,y_cor_im)

  clist_green = interp.interp2(im_green,x_cor_im,y_cor_im)

  clist_blue = interp.interp2(im_blue,x_cor_im,y_cor_im)


  a = np.copy(im)
  a[:,:,0] = clist_red
  a[:, :, 1] = clist_green
  a[:, :, 2] = clist_blue
  return a

def morph_tri(im1, im2, im1_pts, im2_pts, warp_frac, dissolve_frac):
  # TODO: Your code here
  # Tips: use Delaunay() function to get Delaunay triangulation;
  # Tips: use tri.find_simplex(pts) to find the triangulation index that pts locates in.
  im1,im2 = helpers.imresize(im1,im2)
  nr = im1.shape[0]
  nc = im1.shape[1]

  morphed_im = np.zeros((len(warp_frac), nr, nc, 3))
  frame = list(range(len(warp_frac)))

  for warp, dissolve, frame_no in zip(warp_frac, dissolve_frac, frame):

    inter_pts = ((1 - warp) * im1_pts) + (warp * im2_pts)

    #Computing Delaunay triangulation
    Tri = Delaunay(inter_pts)
    tri = Tri.simplices

    #uncomment the below lines to visualize the Delaunay triangles
    # fig,(ax1,ax2,ax3) = plt.subplots(1,3)
    # ax1.triplot(inter_pts[:, 0], inter_pts[:, 1], tri.copy())
    # ax2.triplot(im1_pts[:, 0], im1_pts[:, 1], tri.copy())
    # ax3.triplot(im2_pts[:, 0], im2_pts[:, 1], tri.copy())
    # plt.show()

    #Calculating inverse matrices for all the triangles in intermediate frame
    matrix_list = get_abcmat(tri,inter_pts)
    inverse_list = np.linalg.inv(matrix_list)

    #Calculating triangle correspondenses for all the pixels in intermediate image
    x,y=np.meshgrid(np.arange(nc),np.arange(nr))
    x_flat = np.array(x.flatten())
    y_flat = np.array(y.flatten())
    x_flat_t = x_flat.reshape((-1,1))
    y_flat_t = y_flat.reshape((-1, 1))
    z = np.concatenate((x_flat_t,y_flat_t),axis = 1)#
    t_list = Tri.find_simplex(z)

    #Calculating the bary coordinates for all the pixels in the intermediate image
    x_flat1 = x_flat.reshape(1,x_flat.shape[0])
    y_flat1 = y_flat.reshape(1, y_flat.shape[0])
    z_flat1 = np.array(np.ones((1,y_flat1.shape[1])))
    l = np.concatenate((x_flat1,y_flat1,z_flat1))#
    cor_list = np.array(np.hsplit(l,l.shape[1]))
    np.matmul(inverse_list[[1,2],:,:],cor_list[[1,2],:,:])
    bary_mat = np.matmul(inverse_list[t_list,:,:],cor_list)

    #Calculating the A matrices for the image 1 and image 2
    A_mat_im1 = get_abcmat(tri,im1_pts)
    A_mat_im2 = get_abcmat(tri, im2_pts)

    # Get the warped images of both the images in all channels
    im1_warp = get_value(A_mat_im1,t_list,bary_mat,im1)
    im2_warp = get_value(A_mat_im2,t_list, bary_mat,im2)

    #Cross dissolve the warped images to morph
    morph_im = ((1 - dissolve) * im1_warp) + (dissolve * im2_warp)

    #Store all the morphed frames
    morphed_im[frame_no, :, :, :] = morph_im

  return morphed_im


if __name__ == '__main__':
  start_time = time.time()
  im1 = plt.imread('Musk21.jpg')
  im2 = plt.imread('Tony.jpg')

  im1_pts = np.load('musk21_pts.npy')
  im2_pts = np.load('tony_pts.npy')

  warp_frac = np.linspace(0,1,num = 60, dtype =float)
  dissolve_frac = np.linspace(0,1,num = 60, dtype = float)

  #warp_frac = [.5]
 # dissolve_frac = [.5]

  morphed_im = morph_tri(im1, im2, im1_pts, im2_pts, warp_frac, dissolve_frac)

  # generate gif file
  p = 0
  frames = []
  while p < morphed_im.shape[0]:
    frames.append(morphed_im[p, :, :, :])
    p += 1
  imageio.mimsave('1.gif', frames)

  print("--- %s seconds ---" % (time.time() - start_time))

