import cv2
import numpy as np


def show_vertices_2d(vertices_2d, win_name='vertices', wait_key=0, canvas_size=(640, 640)):
    img = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.uint8)
    edges = np.array([
        [0, 1], [0, 2], [0, 4],
        [1, 3], [1, 5],
        [2, 3], [2, 6],
        [4, 5], [4, 6],
        [7, 5], [7, 6], [7, 3]])

    for edge in edges:
        pt1 = tuple(vertices_2d[edge[0]])
        pt2 = tuple(vertices_2d[edge[1]])
        cv2.line(img, pt1, pt2, (0, 255, 255), 2)

    cv2.imshow(win_name, img)
    cv2.waitKey(wait_key)


def weak_perspective_projection(vertices, s=200):
    vertices_2d = s * vertices + np.array([300, 300, 0])
    vertices_2d = vertices_2d[:, :2].astype(np.int32)

    return vertices_2d


def perspective_projection(vertices, K):
    """use perspective projection"""
    vertices_2d = np.matmul(K, vertices.T).T
    vertices_2d[:, 0] /= vertices_2d[:, 2]
    vertices_2d[:, 1] /= vertices_2d[:, 2]
    vertices_2d = vertices_2d[:, :2].astype(np.int32)

    return vertices_2d


def camera_extrinsic_matrix_R(yaw_angle, roll_angle, pitch_angle):
    yaw = np.deg2rad(yaw_angle)
    roll = np.deg2rad(roll_angle)
    pitch = np.deg2rad(pitch_angle)
    R_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    R_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    R_pitch = np.array([
        [0, np.cos(pitch), -np.sin(pitch)],
        [1, 0, 0],
        [0, np.sin(pitch), np.cos(pitch)]
    ])
    R = np.matmul(R_pitch, np.matmul(R_yaw, R_roll))

    return R


def sample_a():
    # 定义立方体的边长
    length = 1

    # 定义立方体的8个顶点坐标
    vertices_w = np.array([
        [-length / 2, -length / 2, -length / 2],
        [-length / 2, -length / 2, length / 2],
        [-length / 2, length / 2, -length / 2],
        [-length / 2, length / 2, length / 2],
        [length / 2, -length / 2, -length / 2],
        [length / 2, -length / 2, length / 2],
        [length / 2, length / 2, -length / 2],
        [length / 2, length / 2, length / 2]])

    x = 30
    yaw = np.deg2rad(x)

    # 构造绕yaw轴旋转的旋转矩阵
    R_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    # 相机外参 在什么位置看 正方体
    T = np.array([0, 0, 5])

    vertices_c = np.matmul(R_yaw, vertices_w.T).T + T

    # 定义相机内参
    fx = 1600
    fy = 1600
    cx = 300
    cy = 300
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    vertices_2d = perspective_projection(vertices_c, K)
    # vertices_2d = weak_perspective_projection(vertices_transform, s=200)

    show_vertices_2d(vertices_2d)


if __name__ == '__main__':
    sample_a()
