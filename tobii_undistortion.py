import numpy as np
from matplotlib import image
from scipy import ndimage
import glob

img_directory = 'images/tobii/*.png'
destination_directory = 'undistorted/'

# Hard-coded Parameters
DIM=(1920, 1080)
K=np.array([[1133.22, 0, 1920/2-23.038],
            [0, 1133.22, 1080/2+1.704],
            [0, 0, 1]])
D=np.array([0.071919304166037062, -0.295408, 0.259981]) #(k1, k2, k3)

def undistortRadialNoCrop(img_path):
    img = image.imread(img_path)
    # # Padding for image
    #x = np.linspace(-199, len(img[0])+200, len(img[0]) + 400)
    #y = np.linspace(-199, len(img)+200, len(img) + 400)
    #img = np.pad(img, [2, 2], mode='constant')
    # No padding for image
    x = np.linspace(1, len(img[0]), len(img[0]))
    y = np.linspace(1, len(img), len(img))
    X, Y = np.meshgrid(x, y)

    to_multiply = np.array([np.transpose(X).flatten(),np.transpose(Y).flatten(),np.ones(np.transpose(Y.flatten()).shape)])
    points = np.matmul(np.linalg.inv(K), np.array(to_multiply))

    points_distort = radialDistortPoints(points[0:2,:], D, K)
    X_distort = np.reshape(points_distort[0], tuple(np.array([X.shape[0],X.shape[1]])), order="F")
    Y_distort = np.reshape(points_distort[1], tuple(np.array([X.shape[0],X.shape[1]])), order="F")

    # Array to keep undistorted image
    im_undist = np.zeros((y.size, x.size, 3))
    for i in range(0, 3):
        im_undist[:,:,i] = np.absolute(np.nan_to_num(ndimage.map_coordinates(img[:,:,i], [Y_distort.ravel(), X_distort.ravel()], order=3).reshape(im_undist[:,:,i].shape)))
        np.place(im_undist[:,:,i], im_undist[:,:,i] >1, 1)

    image.imsave(destination_directory+img_path, im_undist)
    print(img_path+" done")

def radialDistortPoints(points, D, K):
    k1 = D[0]
    k2 = D[1]
    k3 = D[2]

    a = points[0, :]
    b = points[1, :]
    r = np.sqrt(np.multiply(a,a) + np.multiply(b,b))
    r2 = np.multiply(r,r)
    r4 = np.multiply(r2,r2)
    r6 = np.multiply(np.multiply(r2,r2),r2)
    xp = np.multiply((1 + k1 * r2 + k2 * r4 + k3 * r6),a)
    yp = np.multiply((1 + k1 * r2 + k2 * r4 + k3 * r6),b)
    f = K[0][0]
    cx = K[0][2]
    cy = K[1][2]
    u = np.multiply(f,xp) + cx
    v = np.multiply(f,yp) + cy
    uvDistorted = np.array([u,v])
    return uvDistorted

if __name__ == '__main__':
    images = glob.glob(img_directory)
    for p in images:
        undistortRadialNoCrop(p)