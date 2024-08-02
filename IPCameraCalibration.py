import cv2
import threading
import numpy as np
import os
from queue import Queue

str1 = "rtsp://admin:sp123456@192.168.1.108/"
str2 = "rtsp://admin:sp123456@192.168.1.109/"

cap1 = cv2.VideoCapture(str1)
cap2 = cv2.VideoCapture(str2)

#important
############################
num_pic = 2

square_size = 6
CHESSBOARD = (7, 5)

# square_size = 2.2 #cm
# square_size = 2.35 #cm
# CHESSBOARD = (9, 6)

############################


maxsize = 1  
q1 = Queue(maxsize=maxsize)
q2 = Queue(maxsize=maxsize)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objpoints = []
imgpoints1 = []
imgpoints2 = []

objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
objp[:, :2] = square_size * np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)

output_dir = "./0802test/"
os.makedirs(output_dir, exist_ok=True)

def syn_time(q):
    latest = min(q1.qsize(), q2.qsize())
    if latest and q.qsize() > latest:
        for _ in range(q.qsize() - latest):
            q.get()

def get_frame(q, cap):
    while True:
        ret, frame = cap.read()
        if ret:
            syn_time(q)
            if q.qsize() < maxsize:
                q.put(frame)

def save_images(frame1, frame2, count):
    imname1 = f"{output_dir}img108_{count}.png"
    imname2 = f"{output_dir}img109_{count}.png"
    cv2.imwrite(imname1, frame1)
    cv2.imwrite(imname2, frame2)
    print(f"儲存影像: {imname1} 和 {imname2}")

def print_info(camera, mtx, dist, rvecs, tvecs):
    print(f"Camera {camera} parameter : \n")
    print("camera matrix 印相機矩陣（內參） : \n", np.array2string(mtx, separator=','))
    print("dist 失真係數 : \n", np.array2string(dist, separator=','))
    print("rvecs 旋轉向量（外參） : \n", rvecs)
    print("tvecs 位移向量（外參） : \n", tvecs)
    print('\n')

def process_frames():
    count = 0
    saved_images = 0

    while True:
        if not q1.empty() and not q2.empty():
            frame1 = q1.get()
            frame2 = q2.get()

            resized_frame1 = cv2.resize(frame1, (0, 0), fx=0.5, fy=0.5)
            resized_frame2 = cv2.resize(frame2, (0, 0), fx=0.5, fy=0.5)

            combined_frame = cv2.hconcat([resized_frame1, resized_frame2])

            cv2.imshow('Synchronized Frames', combined_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):

                gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

                ret1, corners1 = cv2.findChessboardCorners(gray1, CHESSBOARD, None)
                ret2, corners2 = cv2.findChessboardCorners(gray2, CHESSBOARD, None)

                if ret1 and ret2:
                    objpoints.append(objp)

                    corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
                    corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

                    imgpoints1.append(corners1)
                    imgpoints2.append(corners2)
                    print("camera108_cornerPoint = \n", imgpoints1)
                    print("camera109_cornerPoint = \n", imgpoints2)

                    cv2.drawChessboardCorners(frame1, CHESSBOARD, corners1, ret1)
                    cv2.drawChessboardCorners(frame2, CHESSBOARD, corners2, ret2)

                    count += 1
                    saved_images += 1

                    save_images(frame1, frame2, count)

                    cv2.waitKey(500)

                if saved_images >= num_pic:
                    count = 0
                    saved_images = 0

                    ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpoints1, gray1.shape[::-1], None, None)
                    ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, imgpoints2, gray2.shape[::-1], None, None)

                    print_info(108, mtx1, dist1, rvecs1, tvecs1)
                    print_info(109, mtx2, dist2, rvecs2, tvecs2)

                    validate_points(mtx1, dist1, rvecs1, tvecs1, mtx2, dist2, rvecs2, tvecs2, corners1, corners2)

            elif key == ord('q'):  
                break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

def validate_points(mtx1, dist1, rvecs1, tvecs1, mtx2, dist2, rvecs2, tvecs2, corners1, corners2):

    rvecs1_last = np.asarray(rvecs1[-1], dtype=np.float64) if isinstance(rvecs1, tuple) else np.asarray(rvecs1, dtype=np.float64)
    tvecs1_last = np.asarray(tvecs1[-1], dtype=np.float64) if isinstance(tvecs1, tuple) else np.asarray(tvecs1, dtype=np.float64)
    rvecs2_last = np.asarray(rvecs2[-1], dtype=np.float64) if isinstance(rvecs2, tuple) else np.asarray(rvecs2, dtype=np.float64)
    tvecs2_last = np.asarray(tvecs2[-1], dtype=np.float64) if isinstance(tvecs2, tuple) else np.asarray(tvecs2, dtype=np.float64)

    R1, _ = cv2.Rodrigues(rvecs1_last)
    R2, _ = cv2.Rodrigues(rvecs2_last)

    print('val_r1 = ',rvecs1_last)
    print('val_t1 = ',tvecs1_last)
    print('val_r2 = ',rvecs2_last)
    print('val_t2 = ',tvecs2_last)

    P1 = np.hstack((R1, tvecs1_last))
    P1 = np.dot(mtx1, P1)

    P2 = np.hstack((R2, tvecs2_last))
    P2 = np.dot(mtx2, P2)

    # 自動撈取棋盤格上的指定點
    points_indices = [(0, 0), (1, 0), (0, 1)]  # 棋盤格上的(0, 0), (1, 0), (0, 1)三個點
    pts1 = np.array([corners1[CHESSBOARD[0] * y + x].ravel() for x, y in points_indices], dtype='float32')
    pts2 = np.array([corners2[CHESSBOARD[0] * y + x].ravel() for x, y in points_indices], dtype='float32')

    print(f"(0, 0) in camera 1 = {pts1[0]}")
    print(f"(1, 0) in camera 1 = {pts1[1]}")
    print(f"(0, 1) in camera 1 = {pts1[2]}")

    print(f"(0, 0) in camera 2 = {pts2[0]}")
    print(f"(1, 0) in camera 2 = {pts2[1]}")
    print(f"(0, 1) in camera 2 = {pts2[2]}")

    # 使用三角測量獲得三維點
    points_4d_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points_3d = points_4d_hom[:3] / points_4d_hom[3]

    print("三維坐標：", points_3d)

    # 計算三維空間中(0, 0)-(1, 0)和(0, 0)-(0, 1)的距離
    distance_01 = np.linalg.norm(points_3d[:, 0] - points_3d[:, 1])
    distance_02 = np.linalg.norm(points_3d[:, 0] - points_3d[:, 2])

    print(f"距離(0,0)-(1,0): {distance_01} cm (應約等於 {square_size} cm)")
    print(f"距離(0,0)-(0,1): {distance_02} cm (應約等於 {square_size} cm)")

if __name__ == "__main__":
    thread1 = threading.Thread(target=get_frame, args=(q1, cap1)).start()
    thread2 = threading.Thread(target=get_frame, args=(q2, cap2)).start()

    process_frames()

