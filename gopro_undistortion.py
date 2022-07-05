import cv2
assert cv2.__version__[0] == '4', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np
import glob

img_directory = 'sample_frames_gopro/*.png'
destination_directory = 'undistorted/'

DIM=(1920, 1080)
K=np.array([[955.76, 0, 1920/2-4.8148],
            [0, 955.76, 1080/2-9.83184],
            [0, 0, 1]])
D=np.array([[0.044273025549419101], [0.024841471047530431], [-0.012883284942803224], [0]]) #(k1, k2, k3, k4 missing?)

# Balance = 0.0 only saves the part of the image that has been completed distorted and is in the same aspect ratio
# Balance = 1.0 saves all pixels of the original image
def undistortFisheyeNoCrop(img_path, balance=1.0):
    img = cv2.imread(img_path)

    # This is how K, dim2 and balance are used to determine the final K used to un-distort image
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, DIM, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    # This saves the image to a local directory
    cv2.imwrite(destination_directory+img_path, undistorted_img)
    print(img_path+" done")

if __name__ == '__main__':
    images = glob.glob(img_directory)
    for p in images:
        undistortFisheyeNoCrop(p)