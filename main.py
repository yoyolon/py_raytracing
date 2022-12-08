import numpy as np
from PIL import Image
import time

WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480
WINDOW_RATIO = WINDOW_WIDTH / WINDOW_HEIGHT
INFTY = float('inf')
EPSILON = 0.0001
ZERO = np.zeros(3)
FOCAL_LENGTH = 1.0
INV_GAMMA = 1 / 1.0
BG_COLOR = np.array((1.0, 1.0, 1.0))
DIFFURE = 1
SPECULAR = 2

def ray_pos(orig, dir, t):
    return orig + t * dir

class Intersection:
    def __init__(self, p=None, t=None, m=None, n=None):
        self.pos = p
        self.t = t
        self.mat = m
        self.normal = n


class Sphere:
    def __init__(self, r, c, m):
        self.radius = r
        self.center = c
        self.mat = m
    
    def intersect(self, orig, dir, isect, t_max=INFTY, t_min=EPSILON):
        temp = orig - self.center
        a = np.dot(dir, dir)
        b_half = np.dot(dir, temp)
        c = np.dot(temp, temp) - self.radius**2
        D = b_half**2 - a*c
        if D < 0: return False
        b = b_half * 2
        d = 2 * np.sqrt(D)
        t = (-b - d) / (2 * a)
        if t < t_min or t > t_max:
            t = (-b + d) / (2 * a)
            if t < t_min or t > t_max:
                return False
        isect.t = t
        isect.pos = ray_pos(orig, dir, isect.t)
        n = isect.pos - self.center
        isect.normal = n / np.linalg.norm(n)
        isect.mat = self.mat
        return True

scene = []
sphere = Sphere(2.0, np.array((0.0,0.0,-5.0)), DIFFURE)
scene.append(sphere)

def raytracing(orig, dir):
    for obj in scene:
        isect = Intersection()
        if obj.intersect(orig, dir, isect):
            return isect.normal * 0.5 + 0.5
    return BG_COLOR

if __name__=="__main__":
    t_start = time.time()
    array = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3))
    for i in range(WINDOW_HEIGHT):
        for j in range(WINDOW_WIDTH):
            u = j / WINDOW_WIDTH * -2 + 1.0
            v = i / WINDOW_HEIGHT * -2 + 1.0
            w = -FOCAL_LENGTH
            orig = ZERO
            dir = np.array((u*WINDOW_RATIO, v, w))
            array[i][j] = raytracing(orig, dir)
    # 画像を保存
    array = 255 * (array ** INV_GAMMA) # ガンマ補正
    array = array.astype(np.uint8)    
    img = Image.fromarray(array)
    img.save("img.png")
    t_end = time.time()
    print("time: ", t_end-t_start)