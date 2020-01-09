import importlib
import HelperFunctions as hf
import HelperFunctions2 as hf2
import cv2 as cv2
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog

importlib.reload(hf)


# importlib.reload(nf)


def limited_zip(*iterables):
    minlength = float('inf')

    for it in iterables:
        if len(it) < minlength:
            minlength = len(it)

    iterators = [iter(it) for it in iterables]
    count = 0
    while iterators and count < minlength:
        yield tuple(map(next, iterators))
        count += 1


def mergeadjacentboxes(startpoints, endpoints):
    newspoint = []
    newepoint = []
    removeindex = []
    updatedstart = (0, 0)
    updatedend = (0, 0)
    for spoint, epoint in zip(startpoints, endpoints):
        updatedstart = spoint
        updatedend = epoint
        for index, (spoint2, epoint2) in enumerate(zip(startpoints, endpoints)):
            sj, si = spoint
            ej, ei = epoint
            s2j, s2i = spoint2
            e2j, e2i = epoint2
            if ei in range(s2i, e2i + 3):
                if ej in range(s2j - 3, s2j + 3):
                    updatedstart = (sj, si)
                    updatedend = (e2j, e2i)
                    removeindex.append(index)
                # elif ei in range(s2i-10,e2i+10):
                elif sj in range(s2j - 3, s2j + 3):
                    updatedstart = (s2j, si)
                    updatedend = (ej, e2i)
                    removeindex.append(index)
        ##            if sj in range(s2j-10,e2j+10):
        ##                if ei in range(s2i-10,s2i+10):
        ##                    updatedstart= (sj,s2i)
        ##                    updatedend=   (e2j,ei)
        ##                    removeindex.append(index)
        ##                    print("Merge box",sj,s2i,sj,si)
        ##            if ej in range(s2j-10,e2j+10):
        ##                if ei in range(s2i-10,s2i+10):
        ##                    updatedstart=(sj,si)
        ##                    updatedend=  (e2j,e2i)
        ##                    removeindex.append(index)
        ##                    print("Merge box2",sj,s2i,sj,si)
        newspoint.append(updatedstart)
        newepoint.append(updatedend)
    print("Remove Index", removeindex)
    removeindex.reverse()
    my_set = set(removeindex)
    removeindex = list(my_set)
    totalindex = len(startpoints)
    print("Remove Index after unique", removeindex)
    for index in removeindex:
        index2 = index - totalindex + 1
        newspoint.pop(index2)
        newepoint.pop(index2)
    return newspoint, newepoint


# img read block added by wajahat

testP, origP = None, None
root = tk.Tk()
root.withdraw()

temp_test = filedialog.askopenfilename()
temp_orig = filedialog.askopenfilename()

if temp_test is not None or temp_test != " ":
    testP = temp_test
if temp_orig is not None or temp_orig != " ":
    origP = temp_orig

# READ
print('Reading Images...')
test, orig = cv2.imread(testP), cv2.imread(origP)
test = hf.sameSize(orig, test)

# ALIGNMENT
print('At image alignment...')
test = hf.alignImages(test.copy(), orig.copy())
test = hf.alignImages(test.copy(), orig.copy())

# read block end

stepsw = [60, 30]  # Widths of comparision window in i direction
stepsh = [60, 30]  # Heights of comparision window in j direction
wjumps = [int(stepsw[0] / 4), int(stepsw[1] / 2)]  # Distance by which window shifts in each iteration in i direction
hjumps = [int(stepsh[0] / 4), int(stepsh[1] / 2)]  # Distance by which window shifts in each iteration in j direction

h_bias = 10  # Determines the distance from an error where you don't want another error. Error to error clearance.
w_bias = 10

meth = 'cv2.TM_CCOEFF_NORMED'
threshold = 0.19  # Comparision accuracy threshold.... Increasing it, increases number of errors

hf.resFolderPath = r'C:\Projects\ESS\res\\'

orig_with_box = orig.copy()
test_with_box = test.copy()
variant = test.copy()

# COLOR-CONVERSION

test_gray = cv2.cvtColor(test.copy(), cv2.COLOR_BGR2GRAY)
orig_gray = cv2.cvtColor(orig.copy(), cv2.COLOR_BGR2GRAY)


test_gray = hf.getXray(test_gray)
orig_gray = hf.getXray(orig_gray)


ws, hs = test_gray.shape
wo, ho = orig_gray.shape
lastj = 0

wsmall = min(ws, wo)
hsmall = min(hs, ho)
#print("Original Image size=", wo, ho)
#print("Test Image size", ws, hs)

timearray = []
iterationendtime = time.time()
starttime = time.time()
st_step = 10
br_neglect = 100
startpoints = []
endpoints = []
timearray.append(time.time())

print('template matching....')
for stepw, steph, wjump, hjump in zip(stepsw, stepsh, wjumps, hjumps):
    print("Steps=", stepw, steph)
    for i in range(st_step, wsmall - 2 * stepw, wjump):
        iterationendtime = time.time()
        timearray.append("i")
        timearray.append(time.time() - iterationendtime)
        for j in range(st_step, hsmall - 2 * steph, hjump):
            if hf.checkinrange(j, i, steph, stepw, startpoints, endpoints):
                # if hf.checknear(j,i,steph,stepw,startpoint,endpoint,h_bias,w_bias):
                continue
            template = test_gray[i:i + stepw, j:j + steph].copy()
            wt, ht = template.shape[::-1]
            if (i >= stepw) and (j >= steph):
                orig_area = orig_gray[i - stepw:i + int(1.5 * stepw), j - steph:j + int(1.5 * steph)].copy()
            else:
                orig_area = orig_gray[i:i + int(1.5 * stepw), j:j + int(1.5 * steph)].copy()

            template_mean = cv2.mean(template)

            if template_mean[0] < br_neglect or template_mean[1] < br_neglect or template_mean[
                2] < br_neglect:  # skip template matching until error window is crossed
                evmeth = eval(meth)
                res = cv2.matchTemplate(orig_area, template, evmeth)
                loc = np.where(res >= threshold)
                if len(loc[0]) <= 0:
                    timearray.append("e")
                    startpoints.append((j, i))
                    endpoints.append((j + steph, i + stepw))

            lastj = j
        # print(i, lastj)

# print("End Time = ", time.time() - starttime)
# print("Points before merge", startpoint, endpoint)
# startpoint,endpoint = mergeadjacentboxes(startpoint,endpoint)
# print("Points after merge",startpoint,endpoint)
thisRect = None
connectedPts = []
notConnectedRecs = []

this_is_connected = False

for startpoint_this, endpoint_this in limited_zip(startpoints.copy(), endpoints.copy()):
    print('startpoints length = ', len(startpoints))
    print(startpoints)
    this_is_connected = False
    cv2.putText(orig_with_box, str(startpoint_this) + '  ' + str(endpoint_this), startpoint_this,
                cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0))
    cv2.rectangle(orig_with_box, startpoint_this, endpoint_this, (0, 0, 255), 2)
    thisRect = (startpoint_this, endpoint_this)
    for startPoint_among_all, endPoint_among_all in limited_zip(startpoints.copy(), endpoints.copy()):
        # compare this rect with all the others
        if (startPoint_among_all, endPoint_among_all) != (startpoint_this, endpoint_this):
            if hf.isConnected(thisRect, (startPoint_among_all, endPoint_among_all), checkOffset=10):
                # find resultant of both
                res_this_st = startpoint_this
                res_this_end = endPoint_among_all
                # remove this rectangle & its counterpart from original list to add its resultant in list
                startpoints = [i for i in startpoints if i != startpoint_this]
                startpoints = [i for i in startpoints if i != startPoint_among_all]
                endpoints = [i for i in endpoints if i != endpoint_this]
                endpoints = [i for i in endpoints if i != endPoint_among_all]
                startpoints.append(res_this_st)  # resultant st pt
                endpoints.append(res_this_end)  # resultant end pt
                print('Resultant: ', res_this_st, res_this_end)
                this_is_connected = True
                break
    if len(startpoints) != 1 and not this_is_connected:
        print('************Loop through recs among all completed with no connection for rec.', thisRect)
        startpoints = [i for i in startpoints if i != startpoint_this]
        endpoints = [i for i in endpoints if i != endpoint_this]  # this rect is now extraneous for comparision
        cv2.rectangle(test_with_box, startpoint_this, endpoint_this, (0, 0, 255), 2)
        cv2.rectangle(variant, startpoint_this, endpoint_this, (0, 0, 255), 2)

for start, stop in zip(startpoints, endpoints):
    pixeloffset = 15
    cv2.rectangle(test_with_box, (start[0] - pixeloffset, start[1] - pixeloffset),
                  (stop[0] + pixeloffset, stop[1] + pixeloffset), (0, 100, 255), 2)
    cv2.putText(test_with_box, str(start) + '  ' + str(stop), start,
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))

startpoints, endpoints = hf.findResultants(startpoints, endpoints)
for start, stop in zip(startpoints, endpoints):
    cv2.rectangle(variant, start, stop, (127, 0, 127), 2)

# print("Error count=", len(startpoint))
# print(timearray[:100])
# show(test)
# print("Saving")
cv2.imwrite('Scan_with_boxes.png', test_with_box)
cv2.imwrite('Original_with_boxes.png', orig_with_box)
cv2.imwrite('variant.png', variant)

print("Saved")
