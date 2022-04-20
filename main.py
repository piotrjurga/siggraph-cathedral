import numpy as np
import matplotlib.pyplot as plt
import cv2
from numba import jit

@jit
def dominant_blur(img, size):
    result = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            si = max(0, i-size//2)
            sj = max(0, j-size//2)
            px = img[si:si+size, sj:sj+size]
            counts = np.zeros(5)
            for row in px:
                for p in row: counts[p] += 1
            result[i,j] = np.argmax(counts)
    return result

@jit
def find_best(imgs):
    dim = (imgs[0].shape[0], imgs[0].shape[1])
    res_idx = np.zeros(dim, dtype=np.uint8)

    for i in range(dim[0]):
        for j in range(dim[1]):
            px = [ img[i,j] for img in imgs ]

            dists = [np.infty]*len(imgs)
            for pi in range(len(px)):
                for pj in range(len(px)):
                    if (pi == pj): continue
                    dist = np.sum((px[pi]-px[pj])**2)
                    dists[pi] = min(dist, dists[pi])

            for pi in range(len(px)):
                dists[pi] += 0.3 * np.sum((px[pi]-130)**2)
                dists[pi] += 50*(np.max(px[pi])-np.min(px[pi]))

            best = np.argmin(np.array(dists, dtype=np.uint32))
            res_idx[i,j] = best
    return res_idx

def main():
    img_names = [ 'd001.jpg', 'd002.jpg', 'd003.jpg', 'd004.jpg', 'd005.jpg' ]
    imgs = [ cv2.imread(name) for name in img_names ]
    dim = (imgs[0].shape[0], imgs[0].shape[1])

    res_idx = find_best(np.array(imgs, dtype=np.uint8))
    res_idx = dominant_blur(res_idx, 7)

    result = np.zeros(imgs[0].shape)
    for i in range(dim[0]):
        for j in range(dim[1]):
            result[i,j] = imgs[res_idx[i,j]][i,j]

    cv2.imwrite('res.png', result)
    print('ok')

main()
