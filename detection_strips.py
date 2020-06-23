# OLA-Simple Image Processing
# Copyright 2018, Parker Ruth

import os

import numpy as np

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import uniform_filter
from scipy.ndimage.measurements import maximum_position as ndimaxpos
from scipy.ndimage.morphology import binary_fill_holes

from skimage import io
from skimage.color import rgb2gray
from skimage.filters import rank
from skimage.morphology import disk, square
from skimage.transform import downscale_local_mean
from skimage.transform import rotate
from skimage.filters import threshold_isodata
from skimage.measure import label, regionprops

BOXH, BOXW = 3, 4  # constants scaling ttest box height and height

BAND_THRESHOLD = 1.5  # tstat value to recognize band
BANDS_TO_CALL = {
    (False, False, True): 'M',
    (True, True, True): 'N',
    (True, False, True): 'O',
    (True, True, False): 'P',
    (True, False, False): 'Q',
    (False, False, False): 'R'
}


def make_calls_from_tstats(strips_tstats):
    return [
        BANDS_TO_CALL[tuple(
            map(lambda tstat: tstat > BAND_THRESHOLD, strip_tstats))]
        for strip_tstats in strips_tstats
    ]


def process_image_from_file(file, trimmed=True):
    '''processes the given image file
    returns the tstats for each band for each strip
    if trimmed flag is false, attempts to isolate only the paper strip'''

    image = io.imread(file)
    if not trimmed:
        image = trim(image)
    strips = detect_strips(image)
    tstats = [extract_tstats(strip) for strip in strips]

    return tstats


def process_strip(filepath, trimmed=True):
    '''processes the given image file or all images in the given filepath
    returns the tstats for each band for each strip (for each image)
    if trimmed flag is false, attempts to isolate only the paper strip'''

    if os.path.isdir(filepath):  # filepath is a directory
        results = []
        for path in os.listdir(filepath):
            results.append(process_strip(filepath + '/' + path, trimmed))
        return results

    else:  # filepath is a single file

        formats = ['.jpg', '.png', '.tif', '.tiff']
        if any(filepath.lower().endswith(fmt) for fmt in formats):

            strip = io.imread(filepath)

            if not trimmed:
                strip = find_strip(strip)
            return extract_tstats(strip)


def trim(image):
    '''transforms the given image to crop and orient the strips'''
    scale_factor = 5
    temp = rgb2gray(image)
    temp = downscale_local_mean(temp, (scale_factor, scale_factor))

    e = rank.entropy(temp, disk(10))
    fred = binary_fill_holes(e > threshold_isodata(e))
    fred = rank.minimum(fred, disk(10))
    labels = label(fred)

    props = regionprops(labels)
    areas = [prop['area'] for prop in props]
    selection = labels == props[np.argmax(areas)]['label']
    angles = np.linspace(-45, 45)
    rotations = [rotate(selection, angle, resize=True) for angle in angles]
    props = [regionprops(label(r)) for r in rotations]
    bboxes = [prop[0]['bbox'] for prop in props]
    areas = [(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in bboxes]
    best = np.argmin(areas)

    rotated = rotations[best]
    mask = rank.minimum(rotated, square(10)) > 0

    bbox = np.array(regionprops(label(mask))[0]['bbox'])
    rmin, cmin, rmax, cmax = bbox * scale_factor

    transformed = rotate(image, angles[best], resize=True)
    transformed = transformed[rmin:rmax, cmin:cmax]
    return transformed


def detect_strips(image):
    '''returns an array of the paper strips in the given image of tests
    assumes the image has been cropped exactly around the plastic cases'''
    crops = crop(image)
    strips = [find_strip(cropped) for cropped in crops]
    return strips


def crop(strip_array):
    '''approximately crops the given strip array into seperate strip regions'''
    crops = []
    test_height, total_width = strip_array.shape[0], strip_array.shape[1]
    test_width = 0.28 * test_height
    number = round(total_width / test_width)
    test_width = round(total_width / number)
    for i in range(number):
        n = round(test_height * 0.33)
        s = round(test_height * 0.62)
        w = round(test_width * i + test_width / 4)
        e = round(test_width * (i + 1) - test_width / 4)
        cropped = strip_array[n:s, w:e]
        crops.append(cropped)
    return crops


def find_strip(cropped):
    '''finds the strip in the given cropped image; returns the strip alone'''
    cropped = cropped.astype('float')

    h, w, d = cropped.shape
    cropped = cropped[h//10:, :, :]

    r, g, b = cropped[:, :, 0], cropped[:, :, 1], cropped[:, :, 2]
    combo = 2 * r - 4.5 * g + 2.4 * b  # LDA to discriminate paper from plastic
    sigma = len(combo)//50
    combo = gaussian_filter(combo, sigma=sigma)

    strip_width = round(0.3 * combo.shape[1])
    strip_height = int(4 * strip_width)
    weighted = uniform_filter(combo, (strip_height, strip_width),
                              mode='constant', cval=min(combo.flatten()))
    apply_displacement_loss(weighted)

    cy, cx = ndimaxpos(weighted)
    n, s = int(round(cy - strip_height / 2)), int(round(cy + strip_height / 2))
    w, e = int(round(cx - strip_width / 2)), int(round(cx + strip_width / 2))

    margin = (e - w) // 8
    strip = cropped[n+margin:s-margin*2, w+margin:e-margin]

    return strip


def apply_displacement_loss(weighted):
    '''applies a loss function favoring the center of the weighted array
    NOTE - this function modifies the argument IN PLACE'''
    cy, cx = weighted.shape[0] // 2, weighted.shape[1] // 2
    for c, col in enumerate(weighted):
        for r, value in enumerate(col):
            v = ((c - cy) / cx) ** 2 + ((r - cx) / cx) ** 2
            weighted[c, r] = value - v * 0.05


def extract_tstats(strip):
    '''reduces the strip to a 1D signal, identifies the bands, and calculates
    the band intensities; returns the signal, maxima, and band intensities'''
    strip = combine_rgb(strip)  # LDA to discriminate band from background
    signal = smooth(strip.mean(axis=1))
    maxima = find_maxima(signal, 5)
    bands = select_bands(signal, maxima)
    regions = extract_regions(strip, bands)
    tstats = ttest(regions)
    return tstats


def combine_rgb(image):
    '''converts three separate rgb channels to a single band intensity signal
    using LDA coefficients tuned for distinguishing bands on a strip'''
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    return 0.12040484 * r - 0.656551 * g + 0.37544564 * b


def smooth(signal, window_len=11, window_type='hanning'):
    '''smooths the given 1D signal using the specified window type
    (window type can be hanning, hamming, bartlett, or blackman)'''
    if window_len < 3:  # do nothing if window length is less than 3
        return signal
    s = np.r_[signal[window_len-1:0:-1], signal, signal[-2:-window_len-1:-1]]
    if window_type == 'flat':
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window_type+'(window_len)')
    return np.convolve(w/w.sum(), s, mode='valid')[window_len//2:-window_len//2]


def find_maxima(signal, radius):
    '''detects all local maxima in the given signal with given radius
    returns peaks as numpy array of (index, value) pairs'''
    maxima = []
#    signal = signal[BOXH:-BOXH] # trim off problematic edges
    for i in range(radius, len(signal)):
        if max(signal[max(0, i-radius):min(len(signal), i+radius)]) <= signal[i]:
            maxima.append((i, signal[i]))
    return np.array(maxima)


def select_bands(signal, maxima):
    '''returns numpy array of [CTRL, WT, MUT] bands from the given
    signal and list of (index, value) local maxima points'''

    # This algorithm is heuristically adapted for Epson scanners

    div0 = BOXH
    div1 = round(0.25 * len(signal))
    div2 = round(0.6 * len(signal))
    div3 = len(signal) - 4 * BOXH

    done = False
    fish = 0

    while not done:

        band1, band2, band3 = None, None, None

        maxima = maxima[maxima[:, 1].argsort()][::-1]
        for m in maxima:
            if div0 < m[0] and m[0] < div1:
                if band1 is None:
                    band1 = m
            if div1 < m[0] and m[0] < div2:
                if band2 is None:
                    band2 = m
            if div2 < m[0] and m[0] < div3:
                if band3 is None:
                    band3 = m

        if band1 is None:
            band1_loc = signal[div0:div1].argmax() + div0
            band1 = (band1_loc, signal[band1_loc])
        if band2 is None:
            band2_loc = signal[div1:div2].argmax() + div1
            band2 = (band2_loc, signal[band2_loc])
        if band3 is None:
            band3_loc = signal[div2:div3].argmax() + div2
            band3 = (band3_loc, signal[band3_loc])

        bands = band1, band2, band3

        cond1 = (band2[1] - min(signal)) > 1.5 * (band1[1] - min(signal))
        cond2 = (band2[1] - min(signal)) > (band1[1] - min(signal))
        if fish >= 3:
            done = True
        elif cond1 or (cond2 and fish > 0):
            div1 += 10
            div2 += 10
            div3 = min(div3+10, len(signal)-1)
            fish += 1
        else:
            done = True

    return np.array(bands)


def extract_regions(strip, band_locs):
    '''extracts image regions around the given maximum for ttest comparison'''

    bl1, bl2, bl3 = tuple([int(bl[0]) for bl in band_locs])
    bg1, bg2, bg3 = bl1+3*BOXH, bl2+3*BOXH, bl3+3*BOXH
    h, w = strip.shape

    if bg3 + BOXH >= len(strip):
        bg3 = bl3-3*BOXH

    r1 = strip[bl1-BOXH:bl1+BOXH, BOXW:w-BOXW]
    r2 = strip[bl2-BOXH:bl2+BOXH, BOXW:w-BOXW]
    r3 = strip[bl3-BOXH:bl3+BOXH, BOXW:w-BOXW]

    bgr1 = strip[bg1-BOXH:bg1+BOXH, BOXW:w-BOXW]
    bgr2 = strip[bg2-BOXH:bg2+BOXH, BOXW:w-BOXW]
    bgr3 = strip[bg3-BOXH:bg3+BOXH, BOXW:w-BOXW]

    return r1, bgr1, r2, bgr2, r3, bgr3


def ttest(regions):
    '''returns t-statistics for ttest comparisons between given regions'''
    r1, bgr1, r2, bgr2, r3, bgr3 = [r.flatten() for r in regions]

    ssss1 = np.sqrt(np.std(r1)**2 + np.std(bgr1)**2)
    tstat1 = (np.mean(r1) - np.mean(bgr1)) / ssss1

    ssss2 = np.sqrt(np.std(r2)**2 + np.std(bgr2)**2)
    tstat2 = (np.mean(r2) - np.mean(bgr2)) / ssss2

    ssss3 = np.sqrt(np.std(r3)**2 + np.std(bgr3)**2)
    tstat3 = (np.mean(r3) - np.mean(bgr3)) / ssss3

    return tstat1, tstat2, tstat3
