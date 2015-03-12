#__author__ = 'bliq'
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import cv2.cv as cv
import math
import scipy as sp
import sys
from matplotlib import pyplot as plt


def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated
    keypoints, as well as a list of DMatch data structure (matches)
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)

    # Show the image
    cv2.imshow('Matched Features', out)
    cv2.moveWindow("Matched Features", 0, image1.shape[0])

def search_circles(src): # Count c  ircles in image
    circles = cv2.HoughCircles(src, cv.CV_HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=0, maxRadius=0)
    circles = cv2.HoughCircles(src, cv2.HOUGH_GRADIENT, 1.2, 100)
    circles = np.uint16(np.around(circles))
    return circles

def search_lines(src): # Count lines in image

    #lines = cv2.HoughLines(src, cv.CV_HOUGH_PROBABILISTIC, np.pi/180, 100)
    lines = cv2.HoughLines(src, cv.CV_HOUGH_PROBABILISTIC, np.pi/180, 5, 45, 8)
    #lines = cv2.HoughLines(src, cv.CV_HOUGH_PROBABILISTIC, np.pi/180, 10, 7, 1)

    lines = cv2.HoughLines(src, 1, np.pi/180, 50, 20, 20)

    averadge_slope_line = 0.0

    for rho, theta in lines[0]:
        averadge_slope_line += math.degrees(theta)

    averadge_slope_line /= lines.size / 2

    return lines, averadge_slope_line

def analysis_pictures(img):

    param = {}

    gray  = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    blur  = cv2.blur(gray, (7, 7))
    gb    = cv2.GaussianBlur(blur, (7, 7), 0)
    edges = cv2.Canny(gb, 50, 150)
    edges = cv2.Canny(gb, 100, 200)
    #edges = cv2.Canny(gb, 300, 100)

    lines, averadge = search_lines(edges)

    param['lines']    = lines
    param['averadge'] = averadge

    # for rho, theta in lines[0]:
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a*rho
    #     y0 = b*rho
    #     x1 = int(x0 + 1000 * (-b))
    #     y1 = int(y0 + 1000 * (a))
    #     x2 = int(x0 - 1000 * (-b))
    #     y2 = int(y0 - 1000 * (a))

        #cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    #circle = cv2.HoughCircles(edges, cv.CV_HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)

    circle = cv2.HoughCircles(edges, cv.CV_HOUGH_GRADIENT, 5, 25)

    circles = np.uint16(np.around(circle))

    #for i in circle[0,:]:
        # draw the outer circle
        #cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)

    ret, thresh  = cv2.threshold(edges, 1, 255, 1)

    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    storage_ = cv.CreateMemStorage(0)
    contours = cv.FindContours(cv.fromarray(thresh.copy()), storage_, cv.CV_RETR_LIST, cv.CV_CHAIN_APPROX_SIMPLE)

    #contours2 = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    #cv2.drawContours(thresh, contours2[0], 0, 0, 2)
    #cv.DrawContours(cv.fromarray(thresh), contours, cv.CV_RGB(0,255,255), cv.CV_RGB(0,255,0), 2,2,-1)

    # cv2.namedWindow("Verifiable2", 1)
    # cv2.imshow("Verifiable2", thresh)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    param['contours'] = contours
    param['circle'] = circle

    tmp = analis_object_in_img(img)

    param['count'] = tmp['count']
    param['width'] = tmp['width']
    param['height'] = tmp['height']

    #cnt = contours
    #rect = cv2.minAreaRect(contours[0])

    #hull = convex_hull(cnt)
    # rect = cv2.minAreaRect(cnt)
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # img = cv2.drawContours(img,[box],0,(0,0,255),2)

    return param

def mapping_img(img):

    blur  = cv2.blur(img.copy(), (7, 7))
    gb    = cv2.GaussianBlur(blur, (7, 7), 0)
    edges = cv2.Canny(gb, 50, 150)
    edges = cv2.Canny(gb, 100, 200)

    return edges

def analis_object_in_img(img):

    #Преобразуем в оттенок серого
    gray = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY)

    #Вычисляем величину градиента в вертикальном и горизонтальном направлениях
    gradX = cv2.Sobel(gray, ddepth = cv2.cv.CV_32F, dx = 1, dy = 0, ksize = -1)
    gradY = cv2.Sobel(gray, ddepth = cv2.cv.CV_32F, dx = 0, dy = 1, ksize = -1)

    #Получаем изображение с высоким значением горизонтального градиента и низким значенм вертикального
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    #Убираем шумы на изображение
    blurred = cv2.blur(gradient, (1, 1))
    #Проведем бинаризацию изображения
    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

    #Превратим слова в белые полоски
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations = 1)
    closed = cv2.dilate(closed, None, iterations = 5)

    # cv2.namedWindow("Verifiable", 1)
    # cv2.imshow("Verifiable", closed)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #Находим контуры на изображение
    (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mass = [] #массив найденых слов

    #Считаем количество слов
    i = 0
    for element in cnts:
        c = sorted(cnts, key = cv2.contourArea, reverse = True)[i]
        rect = cv2.minAreaRect(c) #определим минимальный ограничивающий прямоугольник (координаты слов)
        if(rect[1][0] > 15 and rect[1][0] > 15): #отсеиваем мильчайшие контуры
            mass.append(rect) #Добавляем прямоугольник с найденным словом

            box = np.int0(cv2.cv.BoxPoints(rect))
            cv2.drawContours(img, [box], -1, (0, 255, 0), 3)

            i = i+1

    tmp = mass[:]

    tmp.sort(key=lambda rows: rows[0][0])  #сортируем массив по оси X
    mass.sort(key=lambda rows: rows[0][1]) #сортируем массив по оси Y

    max = (tmp[-1][0][0], mass[-1][0][1])
    min = (tmp[0][0][0], mass[0][0][1])

    count = len(mass)
    width  = max[0] - min[0]
    height = max[1] - min[1]

    param = {'count' : count, 'width' : width, 'height' : height}

    return param

def bFMatch(temp, varif):

    img1 = mapping_img(temp)   # queryImage
    img2 = mapping_img(varif)  # trainImage

    # Initiate SIFT detector
    sift = cv2.SIFT()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 10)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    BFMatch = float(len(good)) / len(matches)

    # Show only the top 10 matches
    drawMatches(img1, kp1, img2, kp2, good)

    return BFMatch


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

# Инциализируем имена изображений
template_image   = "var2_test3.png"
verifiable_image = "var2_main3.png"

template_image   = "test3.png"
verifiable_image = "test4.png"

# Открываем изображение
image1 = cv2.imread(template_image)
image2 = cv2.imread(verifiable_image)

image1_bfm = cv2.imread(template_image, 0)
image2_bfm = cv2.imread(verifiable_image, 0)

# Brute-Force Matching with SIFT Descriptors and Ratio Test
BFMatch = bFMatch(image1_bfm, image2_bfm)

# Анализируем изображение
param_template   = analysis_pictures(image1)
param_verifiable = analysis_pictures(image2)

# Сравнимаем контуры методом моментов
matchShapes = cv.MatchShapes(param_template['contours'], param_verifiable['contours'], 2, 0.0)

# Определяем длины, ширину и кол-во элементов в подписи

#height2, width2, count2 = analis_object_in_img(image2)

# Словарь выполнения критериев
criterion = {}

# Определяем выполнение критериев
criterion['count_line'] = "true" if math.fabs(param_template['lines'].size / 2  - param_verifiable['lines'].size / 2)  < 100 else "false"
criterion['count_circ'] = "true" if math.fabs(param_template['circle'].size / 2 - param_verifiable['circle'].size / 2) < 40  else "false"
criterion['averadge'] =   "true" if math.fabs(param_template['averadge'] - param_verifiable['averadge']) < 6.0 else "false"
criterion['height']   =   "true" if math.fabs(param_template['height']   - param_verifiable['height'])   < 10  else "false"
criterion['width']   =    "true" if math.fabs(param_template['width']    - param_verifiable['width'])    < 30  else "false"
criterion['count_elem'] = "true" if math.fabs(param_template['count']    - param_verifiable['count'])    < 2   else "false"

criterion['BFMatch']  =   "true" if BFMatch > 0.1     else "false"
criterion['MatchShapes']= "true" if matchShapes < 0.1 else "false"

# Вычисляем кол-во выполненных критериев
count_true = 0
for cr in criterion.values():
    if cr == "true":
        count_true = count_true + 1

# Вывод результата
print "+===========================================================================+"
print u"|\tНазвание критерия\t|\tШаблон\t| Проверяемый | Критерий выполнился?\t|"
print u"|\tКол-во линий\t\t|\t%5d\t|\t%5d\t  |\t\t\t%s\t\t\t|" % (param_template['lines'].size / 2, param_verifiable['lines'].size / 2, criterion['count_line'])
print u"|\tКол-во окруж.\t\t|\t%5d\t|\t%5d\t  |\t\t\t%s\t\t\t|" % (param_template['circle'].size / 2, param_verifiable['circle'].size / 2, criterion['count_circ'])
print u"|\tКол-во элементов\t|\t%5d\t|\t%5d\t  |\t\t\t%s\t\t\t|" % (param_template['count'], param_verifiable['count'], criterion['count_elem'])
print u"|\tУгол наклона\t\t|\t%5.3f\t|\t%5.3f\t  |\t\t\t%s\t\t\t|" % (param_template['averadge'], param_verifiable['averadge'], criterion['averadge'])
print u"|\tВысота\t\t\t\t|\t%5d\t|\t%5d\t  |\t\t\t%s\t\t\t|" % (param_template['height'], param_verifiable['height'], criterion['height'])
print u"|\tДлина\t\t\t\t|\t%5d\t|\t%5d\t  |\t\t\t%s\t\t\t|" % (param_template['width'], param_verifiable['width'], criterion['width'])
print "+---------------------------------------------------------------------------+"
print u"|\tBrute-Force Match\t|\t%13f\t\t  |\t\t\t%s\t\t\t|" % (BFMatch, criterion['BFMatch'])
print u"|\tМетод моментов\t\t|\t%13f\t\t  |\t\t\t%s\t\t\t|" % (matchShapes, criterion['MatchShapes'])
print "+===========================================================================+"

print u"| Проверка успешна пройден. Подписи совпадают.\t\t\t\t\t\t\t\t|" if count_true > 4 else \
      u"| Проверка не пройдена. Подписи не совпадают.\t\t\t\t\t\t\t\t|"

print "+===========================================================================+"


#ret = cv2.matchShapes((c1), (c2), 2, 0.0)

# sorted(cnt1)
# rect = cv.MinAreaRect2(cnt2)
# i = 0
# for element in cnt1:
#     c = sorted(cnt1, reverse = True)[i]
#     rect = cv.MinAreaRect2(cnt1)
#     if(rect[1][0] > 15 and rect[1][0] > 15):
#         #mass.append(rect)
#         box = np.int0(cv2.cv.BoxPoints(rect))
#         cv2.drawContours(image1, [box], -1, (0, 255, 0), 3)
#         i = i+1
#
#
# box = cv.BoxPoints(rect)
# #box = cv2.minAreaRect(c1[0])
# box = np.int0(box)
#
# #cv2.polylines(image1,[box],True,(255,0,255),2)# draw rectangle in blue color
# for j in [0, 1 ,2, 3]:
#     cv.Line( cv.fromarray(image2), grid(box[j]), grid(box[(j+1)%4]), (0,255,255), 20)
# cv2.drawContours(image1,c1[0],0,(255,0,255),2)
# cv.DrawContours(cv.fromarray(image2), cnt2, cv.CV_RGB(0,255,255), cv.CV_RGB(0,255,0), 2,2,-1)


# epsilon = 0.1*cv2.arcLength(cnt1,True)
# approx = cv2.approxPolyDP(cnt1,epsilon,True)
#
# th3 = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# ret = cv2.matchShapes(cnt1, cnt2, 1, 0.0)

# plt.subplot(2,1,1),plt.imshow(thresh2,'gray')
# plt.title("Azazaz")
# plt.xticks([]),plt.yticks([])
#
# plt.subplot(2,1,2),plt.imshow(th4,'gray')
# plt.title("Azazaz2")
# plt.xticks([]),plt.yticks([])
#
# plt.show()

cv2.namedWindow("Template", 1)
cv2.imshow("Template", image1)

cv2.namedWindow("Verifiable", 1)
cv2.imshow("Verifiable", image2)
cv2.moveWindow("Verifiable", image1.shape[1], 0)

cv2.waitKey(0)

cv2.destroyAllWindows()

