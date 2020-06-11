import os
from pathlib import Path
import pickle

import numpy as np

from skimage.io import imread
from skimage.morphology import label, closing, opening, disk
from skimage.segmentation import clear_border
from skimage.measure import regionprops
from skimage.transform import downscale_local_mean
from skimage.filters import gaussian

from sklearn.svm import SVC
from sklearn.cluster import KMeans

FORMATS = ['.jpg', '.png', '.tif', '.tiff']


def analyze(fpath, thresh, minsize, clf, remove_spec=False):

    if os.path.isdir(fpath): # filepath is a directory
        results = {}
        for subpath in os.listdir(fpath):
            if any(subpath.lower().endswith(fmt) for fmt in FORMATS):
                results[subpath] = analyze(os.path.join(fpath, subpath),thresh, minsize, clf, remove_spec)
                means, classes = results[subpath]
        return results

    img = imread(fpath)
    ds_factor = int(np.sqrt(img.size) / 1200) + 1

    dsimg = downscale_local_mean(img, (ds_factor, ds_factor, 1), cval=0)

    # TODO factor this out?
    seg = np.linalg.norm(dsimg.astype(np.float32), axis=-1)
    seg = gaussian(seg, 5)

    seg = seg > thresh
    seg = clear_border(seg)
    seg = opening(seg, disk(2))
    seg = closing(seg, disk(7))
    seg = opening(seg, disk(int(np.sqrt(minsize)*0.2)))
    seg = clear_border(seg)

    lseg = label(seg)
    regions = regionprops(lseg)
    regions = sorted(regions, key=lambda r:r['centroid'][1]) #sort left to right
    while len(regions) > 3:
        minarea = min(r['convex_area'] for r in regions)
        regions = [r for r in regions if r['convex_area'] > minarea]

    maxrow = max(r['centroid'][0] for r in regions)
    regions = [r for r in regions if r['centroid'][0] > maxrow-r['equivalent_diameter']*0.7]
    labels = [r['label'] for r in regions]

    means = []
    for i, reg in enumerate(regions):
        minr, minc, maxr, maxc = tuple(np.array(reg['BoundingBox'])*ds_factor)

        deltar = maxr - minr
        deltac = maxc - minc
        minr, maxr = minr + deltar//3, maxr - deltar//4
        minc, maxc = minc + deltac//4, maxc - deltac//4

        pixels = np.reshape(img[minr:maxr,minc:maxc], (-1,3)).astype(np.float32)

        if remove_spec:
            kmeans = KMeans(n_clusters=2, random_state=0).fit(pixels)
            choice = np.argmin([np.linalg.norm(c) for c in kmeans.cluster_centers_])
            keepers = kmeans.labels_ == choice
            pixels = pixels[keepers]

        means.append(np.mean(pixels, axis=0))

    classes = [clf.predict(np.reshape(mean, (1,-1))) for mean in means]
    return means, classes


# threshold brightness for glowbox or transilluminator
thresh = {
    'glow box': 8,
    'transill': 20,
}

# minimum region size for the tube
# TODO this is determined by how far away the tablet is from the tubes
minsize = {
    'glow box': 500,
    'transill': 190
}



base_path = Path(__file__).parent
clf = {
    'glow box': pickle.load(open(os.path.join(base_path, 'models', 'svm-glow-box.pkl'), 'rb')),
    'transill': pickle.load(open(os.path.join(base_path, 'models', 'svm-transill.pkl'), 'rb'))
}

def glow_box_analysis(file):
    return analyze(file, thresh=thresh[box],
                         minsize=minsize[box],
                         clf = clf[box],
                         remove_spec=False)

def transilluminator_analysis(file):
    return analyze(file, thresh=thresh[transill],
                         minsize=minsize[transill],
                         clf = clf[transill],
                         remove_spec=True)

if __name__ == '__main__':

    # example for glow box
    for box in ('transill', 'glow box'):
        fpath = os.path.join('.', 'tests', box)
        results = analyze(fpath, thresh=thresh[box],
                                 minsize=minsize[box],
                                 clf = clf[box],
                                 remove_spec=False)
