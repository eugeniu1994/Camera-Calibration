import numpy as np
import cv2
import glob

nRows,nCols = 9,6
dimension = 22  # - mm

workingFolder = "./images"
imageType = 'jpg'
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, dimension, 0.001)

#object points, (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((nRows * nCols, 3), np.float32)
objp[:, :2] = np.mgrid[0:nCols, 0:nRows].T.reshape(-1, 2)

#object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# Find the images files
filename = workingFolder + "/*." + imageType
images = glob.glob(filename)

nr_images = 0
imgNotGood = images[1]

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("Reading ", fname)
    ret, corners = cv2.findChessboardCorners(gray, (nCols, nRows), None)
    if ret == True:
        print("ESC to skip or ENTER to accept")
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(img, (nCols, nRows), corners2, ret)
        cv2.imshow('img', img)

        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            print("Skipped")
            imgNotGood = fname
            continue

        print("Accepted")
        nr_images += 1
        objpoints.append(objp)
        imgpoints.append(corners2)
    else:
        imgNotGood = fname

cv2.destroyAllWindows()

if (nr_images > 1):
    print("Got {} images ".format(nr_images))
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Undistort an image
    img = cv2.imread(imgNotGood)
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # undistort
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv2.imwrite(workingFolder + "/calibresult.png", dst)

    np.save('mtx.npy', mtx)
    np.save('dist.npy', dist)
    np.save('rvecs.npy', rvecs)
    np.save('tvecs.npy', tvecs)

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    print("total error: ", mean_error / len(objpoints))
