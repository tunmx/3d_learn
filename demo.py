import cv2
import numpy as np

# 定义立方体的边长
length = 1

# 定义立方体的8个顶点坐标
vertices = np.array([
    [-length / 2, -length / 2, -length / 2],
    [-length / 2, -length / 2, length / 2],
    [-length / 2, length / 2, -length / 2],
    [-length / 2, length / 2, length / 2],
    [length / 2, -length / 2, -length / 2],
    [length / 2, -length / 2, length / 2],
    [length / 2, length / 2, -length / 2],
    [length / 2, length / 2, length / 2]])

# 定义旋转角度（单位：弧度）

x = 0
while 1:
    yaw = np.deg2rad(x)

    # 构造绕yaw轴旋转的旋转矩阵
    R_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    roll = np.deg2rad(x)
    R_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    pitch = np.deg2rad(x)
    R_pitch = np.array([
        [0, np.cos(pitch), -np.sin(pitch)],
        [1, 0, 0],
        [0, np.sin(pitch), np.cos(pitch)]
    ])

    # 相机外参 在什么位置看 正方体
    R = np.matmul(R_pitch, np.matmul(R_yaw, R_roll))
    T = np.array([0, 0, 30])

    vvertices = np.matmul(R, vertices.T).T + T
    print(vvertices.shape)

    # 定义相机内参
    fx = 1600
    fy = 1600
    cx = 300
    cy = 300
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # 投影到2D像素坐标系
    # perspective projection
    vertices_2d = np.matmul(K, vvertices.T).T
    vertices_2d[:, 0] /= vertices_2d[:, 2]
    vertices_2d[:, 1] /= vertices_2d[:, 2]
    vertices_2d = vertices_2d[:, :2].astype(np.int32)

    rvec = np.zeros((3, 1))
    tvec = np.zeros((3, 1))

    retval, rvec, tvec = cv2.solvePnP(vertices.astype(np.float32), vertices_2d.astype(np.float32), K.astype(np.float32),
                                      None, rvec, tvec, False, cv2.SOLVEPNP_ITERATIVE)
    R_solved, _ = cv2.Rodrigues(rvec)
    print("solved", -R_solved)
    print("gt", R)

    print("solved", -tvec)
    print("gt", T)

    # weak perspective projection
    # s = 200
    # vertices_2d = s*vvertices + np.array([300,300,0])
    # vertices_2d = vertices_2d[:, :2].astype(np.int32)

    ##
    # 定义立方体的12条边
    edges = np.array([
        [0, 1], [0, 2], [0, 4],
        [1, 3], [1, 5],
        [2, 3], [2, 6],
        [4, 5], [4, 6],
        [7, 5], [7, 6], [7, 3]])

    # 创建图像
    img = np.zeros((600, 600, 3), dtype=np.uint8)

    # 绘制立方体的边
    for edge in edges:
        pt1 = tuple(vertices_2d[edge[0]])
        pt2 = tuple(vertices_2d[edge[1]])
        cv2.line(img, pt1, pt2, (0, 255, 0), 2)

    # 显示图像
    cv2.imshow("立方体", img)
    cv2.waitKey(30)
    x += 1
