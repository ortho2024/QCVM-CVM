import json
import numpy as np
from sympy import *
def p2p_distance_counting(p1, p2):  # 计算两点之间的距离
    dis = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
    return dis
def p2line_distance_counting(p, p1, p2):# 计算点到直线的距离，p为直线外的点，p1,p2为直线上的点
    A = p2[1] - p1[1]
    B = p1[0] - p2[0]
    C = p1[1] * (p2[0] - p1[0]) - p1[0] * (p2[1] - p1[1])
    dis = np.abs((A * p[0] + B * p[1] + C) / (A ** 2 + B ** 2) ** 0.5)
    return dis

def measure(points,scaler):
    C2p = points['C2p']
    C2d = points['C2d']
    C2a = points['C2a']
    C3up = points['C3up']
    C3ua = points['C3ua']
    C3lp = points['C3lp']
    C3la = points['C3la']
    C3ld = points['C3ld']
    C4up = points['C4up']
    C4ua = points['C4ua']
    C4lp = points['C4lp']
    C4la = points['C4la']
    C4ld = points['C4ld']
    C2Conc = p2line_distance_counting(C2d,C2p,C2a)*scaler
    C3Conc = p2line_distance_counting(C3ld,C3lp,C3la)*scaler
    C4Conc = p2line_distance_counting(C4ld,C4lp,C4la)*scaler
    C3PAR  = p2p_distance_counting(C3up,C3lp)/p2p_distance_counting(C3ua,C3la)
    C3BAR  = p2p_distance_counting(C3lp,C3la)/p2p_distance_counting(C3ua,C3la)

    C4PAR = p2p_distance_counting(C4up, C4lp) / p2p_distance_counting(C4ua, C4la)
    C4BAR = p2p_distance_counting(C4lp, C4la) / p2p_distance_counting(C4ua, C4la)
    return [C2Conc,C3Conc,C4Conc,C3PAR,C3BAR,C4PAR,C4BAR]

