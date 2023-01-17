import numpy as np
from PIL import Image

# 定数
WINDOW_WIDTH  = 640; 
WINDOW_HEIGHT = 480; 
WINDOW_RATIO  = WINDOW_WIDTH / WINDOW_HEIGHT
INFTY   = float('inf')
EPSILON = 0.0001
RED     = np.array((1.0,0.1,0.1))
GREEN   = np.array((0.1,1.0,0.1))
BLUE    = np.array((0.1,0.1,1.0))
BLACK   = np.array((0.0,0.0,0.0))
WHITE   = np.array((1.0,1.0,1.0))
GRAY    = np.array((0.3,0.3,0.3))
LIGHT   = np.array((45.0,45.0,45.0))


def normalize(v): # 単位ベクトルを計算
    return v / np.linalg.norm(v)

def reflect(v, n): # 正反射方向を計算
    r = -v * np.dot(v, n) + 2 * n
    return normalize(r)


class Ray: # レイ
    def __init__(self, o, d):
        self.orig = o; self.dir = d

    def pos(self, t):
        return self.orig + t * self.dir


class Intersection: # 交差点情報
    def __init__(self, p=None, n=None, m=None):
        self.p = p; self.n = n; self.mat = m


class Material: # Phongの反射モデル
    def __init__(self, d, s, a=5):
        self.kd = d; self.ks = s; self.alpha =a

    def shading(self, isect, v_dir, l_pos):
        v = normalize(v_dir); l = normalize(l_pos - isect.p)
        weight = np.dot(reflect(v, isect.n), l) ** self.alpha
        return self.kd * np.dot(isect.n, l) + self.ks * weight


class Sphere: # 球の形状モデル
    def __init__(self, r, c, m):
        self.r = r; self.c = c; self.mat = m

    def intersect(self, r, isect=None, t_max=INFTY, t_min=EPSILON):
        temp = r.orig - self.c
        a = np.dot(r.dir, r.dir)
        b_half = np.dot(r.dir, temp)
        c = np.dot(temp, temp) - self.r**2
        D = b_half**2 - a*c # 判別式
        if D < 0: return False
        b = b_half * 2
        d = 2 * np.sqrt(D)
        t = (-b - d) / (2 * a)
        if t < t_min or t > t_max:
            t = (-b + d) / (2 * a)
            if t < t_min or t > t_max:
                return False
        if isect is not None: # 必要なら交差点情報を更新
            isect.p = r.pos(t)
            isect.n = normalize(isect.p - self.c)
            isect.mat = self.mat
        return True


class PointLight: # 点光源
    def __init__(self, p, l):
        self.p = p; self.L = l

    def emit(self, obj_pos):
        d = obj_pos - self.p # 距離
        return self.L / np.dot(d, d)


class Scene: # シーン中のオブジェクト集合
    def __init__(self):
        self.objects = []; self.lights = []
        mr = Material(RED, WHITE, 40); 
        mg = Material(GREEN, WHITE, 20); 
        mb = Material(BLUE, WHITE, 10)
        s1 = Sphere(2.0, np.array((0.0,np.sqrt(3.0),-10.0)), mr)
        s2 = Sphere(2.0, np.array((2.0,-np.sqrt(3.0),-10.0)), mg)
        s3 = Sphere(2.0, np.array((-2.0,-np.sqrt(3.0),-10.0)), mb)
        self.objects.extend([s1,s2,s3])
        self.lights.append(PointLight(np.array((-3.0,4.0,-2.0)), LIGHT))


def background_color(r):
    t = normalize(r.dir)
    return GRAY * (0.5 + t[1] * 0.5)


def raytracing(r, scene):
    color = BLACK
    for obj in scene.objects:
        isect = Intersection()
        if obj.intersect(r, isect):
            for light in scene.lights:
                color = color + light.emit(isect.p) * obj.mat.shading(isect, r.dir, light.p)
            return color
    return background_color(r)


if __name__=="__main__":
    scene = Scene()
    array = np.zeros((WINDOW_HEIGHT,WINDOW_WIDTH,3))
    for i in range(WINDOW_HEIGHT):
        for j in range(WINDOW_WIDTH):
            u = j/WINDOW_WIDTH*-2+1.0; v = i/WINDOW_HEIGHT*-2+1.0
            dir = np.array((u*WINDOW_RATIO,v,-2.0))
            array[i][j] = raytracing(Ray(np.zeros(3),normalize(dir)),scene)
    array = 255 * (np.clip(array,0.0,1.0) ** (1/2.2)) # ガンマ補正
    img = Image.fromarray(array.astype(np.uint8))
    img.save("img.png")