import sys
import re
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import cv2
import numpy as np
from matplotlib import pyplot as plt
from pdf2image import convert_from_path
from skimage.measure import compare_ssim
import imutils
from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity
#import ImgNoiseFilters as noiseFilters
import subprocess

log=True
offset=-0.772

resFolderPath= r'C:\Projects\ESS\res\\'
config = ('-l eng --oem 1 --psm 3')


def sameSize(ref, test):
    return cv2.resize(test.copy(),(ref.shape[1],ref.shape[0]), cv2.INTER_AREA)

def open2PIL(img, reverse= False, RGB2BGR = True):
    if not reverse:  # opencv to PIL conversion
        if len(img.shape) == 3:
            RGB_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)  # color conversion for good visual
        else:
            RGB_img = img.copy()

        image = Image.fromarray(RGB_img)

    else:
        image = np.array(img)
        if RGB2BGR and len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image

def computeDiff_cv(ref,test):
    diff = cv2.absdiff(ref, test)
    if len(diff.shape)==3:
        mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    else:
        mask = diff.copy()
    th = 1
    imask =  mask>th

    canvas = np.zeros_like(test, np.uint8)
    canvas[imask] = test[imask]
    return canvas

def removeBackground(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    th, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    ## (2) Morph-op to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

    ## (3) Find the max-area contour
    cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnt = sorted(cnts, key=cv2.contourArea)[-1]

    ## (4) Crop and save it
    x,y,w,h = cv2.boundingRect(cnt)
    return img[y:y+h, x:x+w]


def show(img,title,usePIL=True):
        if usePIL:
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img.astype('uint8'))
            img.save(resFolderPath+title+'.jpg')
            try:
                subprocess.check_output(resFolderPath+title+'.jpg',shell = True)
            except subprocess.CalledProcessError:
                img.show(title=title)

        else:
            cv2.namedWindow(title, cv2.WINDOW_KEEPRATIO)
            cv2.imshow(title, img)
            cv2.waitKey(0)

def save(img, name):
    if not name.endswith('.jpg'):
        cv2.imwrite(resFolderPath+name+'.jpg',img)
    else:
        cv2.imwrite(resFolderPath+name,img)

def computeDiff(ref,test,drawImg = None):
    refImg = ref.copy()
    testImg = test.copy()
    if len(refImg.shape) == 3:
        grayA = cv2.cvtColor(ref.copy(), cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(test.copy(), cv2.COLOR_BGR2GRAY)
    else:
        grayA = refImg.copy()
        grayB = testImg.copy()
    diff = (abs(grayA-grayB)* 255).astype('uint8')
    thresh = cv2.threshold(diff, 0, 255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_TC89_L1)
    cnts = imutils.grab_contours(cnts)
    cntsCount = [(0,0,0,0)]
    for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            cntsCount.append((x,y,w,h))
            if drawImg is  None:
                cv2.rectangle(refImg, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.rectangle(testImg, (x, y), (x + w, y + h), (0, 0, 255), 2)
            else:
                cv2.rectangle(drawImg, (x, y), (x + w, y + h), (0, 0, 255), 2)
    if drawImg is None:
        return refImg, testImg, len(cntsCount)-1
    else:
        return drawImg,len(cntsCount)-1

def disSSIM(orig,test,showResults=True,threshold = 3,drawImg = None):
    if orig is None or test is None:
        print('No Img found')
        return
    else:
        origImg = orig.copy()
        testImg = test.copy()
   
    if len(testImg.shape) == 3:
        grayA = cv2.cvtColor(origImg, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(testImg, cv2.COLOR_BGR2GRAY)
    else:
        grayB = testImg.copy()
        grayA = origImg.copy()

    if log:
        print('\n grayA shape: '+ str(grayA.shape))
        print('\n grayB shape: '+ str(grayB.shape))

    (score, diff) = compare_ssim(grayA, grayB, full=True, use_sample_covariance =  False, sigma= 15, gaussian_weights= True) #  greater the signma value greater the accuracy

#    (score, diff) = compare_ssim(grayA, grayB, full = True)
    diff = (diff * 255).astype("uint8")
    print("SSIM: {}".format(score))

    thresh = cv2.threshold(diff, 0, 255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cntsCount = [(0,0,0,0)]

    for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            cntsCount.append((x,y,w,h))
            if abs(h-w) > threshold:
                cv2.rectangle(origImg, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.rectangle(testImg, (x, y), (x + w, y + h), (0, 0, 255), 2)
                if drawImg is not None:
                    cv2.rectangle(drawImg, (x, y), (x + w, y + h), (0, 0, 255), 2)

    if showResults and drawImg is not None:
        show(drawImg,title='SSIM results')
    else:
        show(testImg)
        
    cv2.imwrite('SSIM_result_test_Image.jpg',testImg)
    cv2.imwrite('SSIM_result_original_Image.jpg',origImg)

    if showResults:
        return len(cntsCount)-1, score
    else:
        return origImg, testImg, len(cntsCount)-1, score




def getAngle(img):
      #finding base box for tess as well
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
         # threshold the image, setting all foreground pixels to
         # 255 and all background pixels to 0
    thresh = cv2.threshold(gray, 0, 255,
    cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0)) 
    box = cv2.minAreaRect(coords)
    angle = box[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    return angle






def de_skew(img, center = None, scale = 1.0,cropping= True):
    tesData = pytesseract.image_to_osd(img)        #tess angle for orientaion correction
    pytesAngle = int(re.search('(?<=Rotate: )\d+', tesData).group(0))
    tess_angle=360- pytesAngle

    if tess_angle == 360:
        tess_angle = 0

    angle = getAngle(img) + tess_angle
    
    print('Angle estimation arg for de_skew(): ',angle)
    
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    rotation_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    M = rotation_mat
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

        # find the new width and height bounds
    bound_w = int(h * abs_sin + w * abs_cos)
    bound_h = int(h * abs_cos + w * abs_sin)

        # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - center[0]
    rotation_mat[1, 2] += bound_h/2 - center[1]
    if cropping:
        openCVrotated = cv2.warpAffine(img.copy(), rotation_mat, (bound_w, bound_h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
    else:
        openCVrotated = cv2.warpAffine(img.copy(), M, center ,flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        
    cv2.imwrite(resFolderPath + r'\rotated.jpg',openCVrotated)
    return openCVrotated

   
        
    
def getImg(path = None):
    good = True
    image = None
    if path is not None:
        im = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR)
        #im = cv2.cvtColor(im.copy(),cv2.COLOR_YCR_CB2BGR) 
        #im = cv2.cvtColor(im.copy(),cv2.COLOR_YUV2RGBA_I420)
    
    else:
        print('No path passed to getImg()')
        return

    if im is None:
        print('No image detected on this path')
        return
    
    try:
        im.copy()
    except:
        try:
            maxPages = 500
            print('\n Assuming a pdf, converting to JPEG')
            pages = convert_from_path(path, maxPages, strict = True)
            imgs = [None]*maxPages
            pageNo=1
            for page in pages:
                page.save(resFolderPath + r'\converted' + str(pageNo) + '.jpg', 'JPEG')
                imgs [pageNo] = page.copy()
                pageNo = pageNo + 1
            im = cv2.imread(resFolderPath + r'\converted1.jpg', cv2.IMREAD_COLOR)
        except:
            good = False
            print('File Load Failed, Invalid format', '\n' )
            raise

    if good:
        return im
    else:
        return None




def alignImages2(template, img2bAligned, warp_mode = cv2.MOTION_AFFINE, number_of_iterations = 10):
    if len(template.shape)==3:
        im1_gray = cv2.cvtColor(template.copy(),cv2.COLOR_BGR2GRAY)
        im2_gray = cv2.cvtColor(img2bAligned.copy(),cv2.COLOR_BGR2GRAY)
    else:
        im1_gray = template.copy()
        im2_gray = img2bAligned.copy()
    sz = template.shape
 
    # Define the motion model
    #warp_mode = cv2.MOTION_TRANSLATION
 
    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)
 
    # Specify the number of iterations.
    
 
    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10
 
    # Define termination criteria
    
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
 
    # Run the ECC algorithm. The results are stored in warp_matrix.
    print('in alignImages2 at:findTransformECC ')
    (cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria,None)

    print('in alignImages2 at: warping ')
        
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
    # Use warpPerspective for Homography
        im2_aligned = cv2.warpPerspective (img2bAligned.copy(), warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
    # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(img2bAligned.copy(), warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
 
    return im2_aligned

 
MAX_FEATURES = 1000
GOOD_MATCH_PERCENT = 0.50
 
 
def alignImages(im2bAligned, ref):
    im1 = im2bAligned.copy()
    im2 = ref.copy()
  # Convert images to grayscale
    if len(im1.shape)==3:
        im1Gray = cv2.cvtColor(im1.copy(), cv2.COLOR_BGR2GRAY)
        im2Gray = cv2.cvtColor(im2.copy(), cv2.COLOR_BGR2GRAY)
    else:
        im1Gray = im1.copy()
        im2Gray = im2.copy()
  # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
   
  # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
   
  # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)
 
  # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]
 
  # Draw top matches
    imMatches = cv2.drawMatches(im1.copy(), keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)
   
  # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
 
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
   
  # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
 
  # Use homography
    height, width = im2.shape[:2]
    im1Reg = cv2.warpPerspective(im1.copy(), h, (width, height),flags=cv2.INTER_CUBIC, borderMode= cv2.BORDER_REPLICATE)
   
    return im1Reg
 
 
def alignImges3(orig,test):
    prev= 0
    center = w,h = (test.shape[1]//2,test.shape[0]//2)
    diff = abs(orig[w,h]-test[w,h])
    set = []
    offset = 0
    windowSize = 100
    lowestDiffValOffset = []
    offIndex = []
    WIDTH_RIGHT, WIDTH_LEFT, HEIGHT_RIGHT, HEIGHT_LEFT = 0,1,2,3

    while(offset<500): #width iterations right
        diff = abs(orig[w:w+windowSize, h:h+windowSize] - test[w-offset:w+windowSize+offset, h:h+windowSize])
        set [offset] = diff
        offset = offset + 1

    lowestDiffValOffset [WIDTH_RIGHT] = min(set)
    offIndex[WIDTH_RIGHT] = set.index(min(set))

    while (offset < 500):  # width iterations left
        diff = abs(
            orig[w:w + windowSize, h:h + windowSize] - test[w + offset:w + windowSize - offset, h:h + windowSize])
        set[offset] = diff
        offset = offset + 1

    lowestDiffValOffset [WIDTH_LEFT] = min(set)
    offIndex[WIDTH_LEFT] = set.index(min(set))

    while (offset < 500):  # width iterations right
        diff = abs(
            orig[w:w + windowSize, h:h + windowSize] - test[w:w + windowSize, h:h + windowSize])
        set[offset] = diff
        offset = offset + 1

    return


def getXray(img, thresh1 = 199, thresh2 = 200):  # The smallest value between threshold1 and threshold2 is used for edge linking
    if len(img.shape) == 3: #remove BGR channels if present
        gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    return cv2.Canny(blurred, thresh1, thresh2)


def plot(im,title):
    plt.subplot(121), plt.imshow(im, cmap='gray')
    plt.title(title), plt.xticks([]), plt.yticks([])
    plt.show()


def addImgs(orig_edges,test_edges):
    return cv2.addWeighted(orig_edges.copy(), 1.0, test_edges.copy(), 1.0, 0.0)