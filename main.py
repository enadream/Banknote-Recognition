# This project created by Enadream, enadream.com

import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt

print("OpenCV package is imported")


def takeImage():
    webcam = cv2.VideoCapture(1)

    webcam.set(3, 640)
    webcam.set(4, 480)

    count = 1

    while True:
        success, frame = webcam.read()
        if not success or frame is None:
            break

        mainImage = frame.copy()
        moneyAmount = ""
        # Blue limits
        blue_lower = 150
        blue_upper = 255

        # Blue threshold
        thresholdBlue = frame[:, :, 0]
        thresholdBlue = cv2.inRange(thresholdBlue, blue_lower, blue_upper)

        # Grayscale threshold
        thresholdGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresholdGray = cv2.inRange(thresholdGray, 150, 255)
        lastThreshold = cv2.bitwise_and(thresholdBlue, thresholdGray)

        # Remove some small noise if any
        erode = cv2.erode(lastThreshold, None, iterations=1)
        dilate = cv2.dilate(erode, None, iterations=10)
        erode2 = cv2.erode(dilate, None, iterations=7)

        # Finding contours
        imgContours, _ = cv2.findContours(erode2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(imgContours) != 0:
            # draw in blue the contours that were founded
            # cv2.drawContours(mainImage, imgContours, -1, 255, 3)

            # find the biggest contour (c) by the area
            biggest = max(imgContours, key=cv2.contourArea)
            x, y, width, height = cv2.boundingRect(biggest)

            epsilon = 0.01 * cv2.arcLength(biggest, True)
            approx = cv2.approxPolyDP(biggest, epsilon, True)
            if len(approx) == 4:
                # 4 vertex coordinates in approx
                x1 = approx[0][0][0]
                y1 = approx[0][0][1]

                x2 = approx[1][0][0]
                y2 = approx[1][0][1]

                x3 = approx[2][0][0]
                y3 = approx[2][0][1]

                x4 = approx[3][0][0]
                y4 = approx[3][0][1]

                # Creating a mask to do black of background of banknote
                mask = np.zeros(frame.shape, np.uint8)

                # Making white inside of banknote coordinates
                pointsArray = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
                cv2.fillPoly(mask, [pointsArray], (255, 255, 255))

                # Masking original image with mask
                masked = cv2.bitwise_and(frame, mask)

                # Calculating the degree of slope from money's width
                degree = SlopeCalculate(approx)

                # Taking boundingRect place from masked image
                boundRect = masked[y:y + height, x:x + width]

                # Rotating boundRect image according to slope of money's width, rotating clockwise
                rotated = imutils.rotate_bound(boundRect, degree)

                # Saving rotated masked banknote
                cv2.imwrite("masked.png", rotated)

                LastClean()

                banknoteLast = cv2.imread("banknote.png")
                cv2.imshow("banknote", banknoteLast)

                moneyAmount = Recognizer()
                cv2.drawContours(mainImage, biggest, -1, (0, 255, 0), 2)

                print(moneyAmount)
            else:
                moneyAmount = "Couldn't found banknote"

        cv2.putText(mainImage, moneyAmount, (0, 470), cv2.FONT_ITALIC, 0.7, (0, 0, 255), 2)
        cv2.imshow("Output", mainImage)

        # 33 ms to ensure 30 fps
        waiter = cv2.waitKey(33) & 0xFF

        # Break program if 'b' is pressed
        if waiter == ord('b') or waiter == ord('B'):
            break
        if waiter == ord("c") or waiter == ord("C"):
            Recognizer()
        if waiter == ord("s") or waiter == ord("S"):
            banknoteLast = cv2.imread("banknote.png")
            name = str("Low Res Banknotes/bank_" + str(count) + ".png")
            cv2.imwrite(name, banknoteLast)
            count += 1

    webcam.release()


def SlopeCalculate(rectangle):
    x1, y1 = float(rectangle[0][0][0]), float(rectangle[0][0][1])
    x2, y2 = float(rectangle[1][0][0]), float(rectangle[1][0][1])
    x3, y3 = float(rectangle[2][0][0]), float(rectangle[2][0][1])
    x4, y4 = float(rectangle[3][0][0]), float(rectangle[3][0][1])

    # Calculating the all distance between 4 vertex (2 width, 2 height, 2 diagonal)
    dist1 = int(np.sqrt((y1 - y2) * (y1 - y2) + (x1 - x2) * (x1 - x2)))
    dist2 = int(np.sqrt((y1 - y3) * (y1 - y3) + (x1 - x3) * (x1 - x3)))
    dist3 = int(np.sqrt((y1 - y4) * (y1 - y4) + (x1 - x4) * (x1 - x4)))
    dist4 = int(np.sqrt((y2 - y3) * (y2 - y3) + (x2 - x3) * (x2 - x3)))
    dist5 = int(np.sqrt((y2 - y4) * (y2 - y4) + (x2 - x4) * (x2 - x4)))
    dist6 = int(np.sqrt((y3 - y4) * (y3 - y4) + (x3 - x4) * (x3 - x4)))

    distList = [dist1, dist2, dist3, dist4, dist5, dist6]
    # Sorting distance according to ascending order (height1, height2, width1, width2, diagonal1, diagonal2)
    distList.sort()

    height = distList[1]
    width = distList[3]
    diagonal = distList[5]

    # Finding slope using y/x
    slope = 0
    if width == dist1:
        if (x2 - x1) != 0:
            slope = -(y2 - y1) / (x2 - x1)
    elif width == dist2:
        if (x3 - x1) != 0:
            slope = -(y3 - y1) / (x3 - x1)
    elif width == dist3:
        if (x4 - x1) != 0:
            slope = -(y4 - y1) / (x4 - x1)
    elif width == dist4:
        if (x3 - x2) != 0:
            slope = -(y3 - y2) / (x3 - x2)
    elif width == dist5:
        if (x4 - x2) != 0:
            slope = -(y4 - y2) / (x4 - x2)
    elif width == dist6:
        if (x4 - x3) != 0:
            slope = -(y4 - y3) / (x4 - x3)

    # Translating slope to radian
    radian = np.arctan(slope)
    # Translating radian to degree
    degree = np.degrees(radian)
    return degree


def LastClean():
    # Taking rotated and black masked banknote img
    rotatedImg = cv2.imread("masked.png")

    # Changing to gray img to create a mask, and making a white black image
    rotThresh = cv2.cvtColor(rotatedImg, cv2.COLOR_BGR2GRAY)
    rotThresh = cv2.inRange(rotThresh, 3, 255)

    # Removing noises
    rotThresh = cv2.erode(rotThresh, None, iterations=5)
    rotThresh = cv2.dilate(rotThresh, None, iterations=3)

    # Finding money image
    imgContours, _ = cv2.findContours(rotThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(imgContours) != 0:
        contour = max(imgContours, key=cv2.contourArea)

        x, y, width, height = cv2.boundingRect(contour)

        # Taking money's position from bounding rect
        roi = rotatedImg[y:y + height, x:x + width]
        cv2.imwrite("banknote.png", roi)


def Recognizer():
    unknownImg = cv2.imread("banknote.png")

    moneyName = "Doesn't recognized"

    banknote5_F = cv2.imread("Low Res Banknotes/bank5_F.png")
    banknote5_F_I = cv2.imread("Low Res Banknotes/bank5_F_I.png")
    banknote5_B = cv2.imread("Low Res Banknotes/bank5_B.png")
    banknote5_B_I = cv2.imread("Low Res Banknotes/bank5_B_I.png")

    banknote10_F = cv2.imread("Low Res Banknotes/bank10_F.png")
    banknote10_F_I = cv2.imread("Low Res Banknotes/bank10_F_I.png")
    banknote10_B = cv2.imread("Low Res Banknotes/bank10_B.png")
    banknote10_B_I = cv2.imread("Low Res Banknotes/bank10_B_I.png")

    banknote20_F = cv2.imread("Low Res Banknotes/bank20_F.png")
    banknote20_F_I = cv2.imread("Low Res Banknotes/bank20_F_I.png")
    banknote20_B = cv2.imread("Low Res Banknotes/bank20_B.png")
    banknote20_B_I = cv2.imread("Low Res Banknotes/bank20_B_I.png")

    banknote50_F = cv2.imread("Low Res Banknotes/bank50_F.png")
    banknote50_F_I = cv2.imread("Low Res Banknotes/bank50_F_I.png")
    banknote50_B = cv2.imread("Low Res Banknotes/bank50_B.png")
    banknote50_B_I = cv2.imread("Low Res Banknotes/bank50_B_I.png")

    banknote100_F = cv2.imread("Low Res Banknotes/bank100_F.png")
    banknote100_F_I = cv2.imread("Low Res Banknotes/bank100_F_I.png")
    banknote100_B = cv2.imread("Low Res Banknotes/bank100_B.png")
    banknote100_B_I = cv2.imread("Low Res Banknotes/bank100_B_I.png")

    banknote200_F = cv2.imread("Low Res Banknotes/bank200_F.png")
    banknote200_F_I = cv2.imread("Low Res Banknotes/bank200_F_I.png")
    banknote200_B = cv2.imread("Low Res Banknotes/bank200_B.png")
    banknote200_B_I = cv2.imread("Low Res Banknotes/bank200_B_I.png")

    width = unknownImg.shape[1]
    height = unknownImg.shape[0]

    HistCalculate(width, height)
    HistDif(unknownImg)

    difs = [dif1, dif2, dif3, dif4, dif5, dif6, dif7, dif8, dif9, dif10, dif11, dif12]
    difs.sort()

    index = -1
    difTresh = 17

    while True:
        if index < -12:
            break
        if difs[index] == dif1:
            result1 = CalculateDif(unknownImg, banknote5_F)
            result2 = CalculateDif(unknownImg, banknote5_F_I)

            # Finding smallest result
            if result1 < result2:
                res = result1
            else:
                res = result2

            # if result smaller then difTresh
            if res < difTresh:
                moneyName = "5 TL FRONT"
                break
            else:
                index -= 1
        elif difs[index] == dif2:
            result1 = CalculateDif(unknownImg, banknote5_B)
            result2 = CalculateDif(unknownImg, banknote5_B_I)

            # Finding smallest result
            if result1 < result2:
                res = result1
            else:
                res = result2

            # if result smaller then difTresh
            if res <= difTresh:
                moneyName = "5 TL BACK"
                break
            else:
                index -= 1
        elif difs[index] == dif3:
            result1 = CalculateDif(unknownImg, banknote10_F)
            result2 = CalculateDif(unknownImg, banknote10_F_I)

            # Finding smallest result
            if result1 < result2:
                res = result1
            else:
                res = result2

            # if result smaller then difTresh
            if res <= difTresh:
                moneyName = "10 TL FRONT"
                break
            else:
                index -= 1
        elif difs[index] == dif4:
            result1 = CalculateDif(unknownImg, banknote10_B)
            result2 = CalculateDif(unknownImg, banknote10_B_I)

            # Finding smallest result
            if result1 < result2:
                res = result1
            else:
                res = result2

            # if result smaller then difTresh
            if res <= difTresh:
                moneyName = "10 TL BACK"
                break
            else:
                index -= 1
        elif difs[index] == dif5:
            result1 = CalculateDif(unknownImg, banknote20_F)
            result2 = CalculateDif(unknownImg, banknote20_F_I)

            # Finding smallest result
            if result1 < result2:
                res = result1
            else:
                res = result2

            # if result smaller then difTresh
            if res <= difTresh:
                moneyName = "20 TL FRONT"
                break
            else:
                index -= 1
        elif difs[index] == dif6:
            result1 = CalculateDif(unknownImg, banknote20_B)
            result2 = CalculateDif(unknownImg, banknote20_B_I)

            # Finding smallest result
            if result1 < result2:
                res = result1
            else:
                res = result2

            # if result smaller then difTresh
            if res <= difTresh:
                moneyName = "20 TL BACK"
                break
            else:
                index -= 1
        elif difs[index] == dif7:
            result1 = CalculateDif(unknownImg, banknote50_F)
            result2 = CalculateDif(unknownImg, banknote50_F_I)

            # Finding smallest result
            if result1 < result2:
                res = result1
            else:
                res = result2

            # if result smaller then difTresh
            if res <= difTresh:
                moneyName = "50 TL FRONT"
                break
            else:
                index -= 1
        elif difs[index] == dif8:
            result1 = CalculateDif(unknownImg, banknote50_B)
            result2 = CalculateDif(unknownImg, banknote50_B_I)

            # Finding smallest result
            if result1 < result2:
                res = result1
            else:
                res = result2

            # if result smaller then difTresh
            if res <= difTresh:
                moneyName = "50 TL BACK"
                break
            else:
                index -= 1
        elif difs[index] == dif9:
            result1 = CalculateDif(unknownImg, banknote100_F)
            result2 = CalculateDif(unknownImg, banknote100_F_I)

            # Finding smallest result
            if result1 < result2:
                res = result1
            else:
                res = result2

            # if result smaller then difTresh
            if res <= difTresh:
                moneyName = "100 TL FRONT"
                break
            else:
                index -= 1
        elif difs[index] == dif10:
            result1 = CalculateDif(unknownImg, banknote100_B)
            result2 = CalculateDif(unknownImg, banknote100_B_I)

            # Finding smallest result
            if result1 < result2:
                res = result1
            else:
                res = result2

            # if result smaller then difTresh
            if res <= difTresh:
                moneyName = "100 TL BACK"
                break
            else:
                index -= 1
        elif difs[index] == dif11:
            result1 = CalculateDif(unknownImg, banknote200_F)
            result2 = CalculateDif(unknownImg, banknote200_F_I)

            # Finding smallest result
            if result1 < result2:
                res = result1
            else:
                res = result2

            # if result smaller then difTresh
            if res <= difTresh:
                moneyName = "200 TL FRONT"
                break
            else:
                index -= 1
        elif difs[index] == dif12:
            result1 = CalculateDif(unknownImg, banknote200_B)
            result2 = CalculateDif(unknownImg, banknote200_B_I)

            # Finding smallest result
            if result1 < result2:
                res = result1
            else:
                res = result2

            # if result smaller then difTresh
            if res <= difTresh:
                moneyName = "200 TL BACK"
                break
            else:
                index -= 1
        else:
            moneyName = "Doesn't recognized"
            break

    return moneyName


def CalculateDif(unknown, known):
    # img.shape = (height, width, channel)
    width = unknown.shape[1]
    height = unknown.shape[0]
    size = width * height

    # Resizing known image
    reknown = cv2.resize(known, (width, height))

    # Bluring both image
    blurAmount = (3, 3)
    reknown = cv2.blur(reknown, blurAmount)
    unknown = cv2.blur(unknown, blurAmount)

    grayKnown = cv2.cvtColor(reknown, cv2.COLOR_BGR2GRAY)
    grayUnknown = cv2.cvtColor(unknown, cv2.COLOR_BGR2GRAY)

    # Changing type to uint8 to int16 that's why i will be able to subtract
    grayKnown = grayKnown.astype(np.int16)
    grayUnknown = grayUnknown.astype(np.int16)

    npSubtractG = np.subtract(grayKnown[:, :], grayUnknown[:, :])
    npSubtractG = np.absolute(npSubtractG[:, :])
    grayDif = float(np.sum(npSubtractG))
    grayDif /= size

    return grayDif


def HistCalculate(width, heigt):
    global hist1, hist2, hist3, hist4, hist5, hist6, hist7, hist8, hist9, hist10, hist11, hist12

    size = (width, heigt)

    banknote5_F = cv2.imread("Low Res Banknotes/bank5_F.png", 0)
    banknote5_B = cv2.imread("Low Res Banknotes/bank5_B.png", 0)

    banknote10_F = cv2.imread("Low Res Banknotes/bank10_F.png", 0)
    banknote10_B = cv2.imread("Low Res Banknotes/bank10_B.png", 0)

    banknote20_F = cv2.imread("Low Res Banknotes/bank20_F.png", 0)
    banknote20_B = cv2.imread("Low Res Banknotes/bank20_B.png", 0)

    banknote50_F = cv2.imread("Low Res Banknotes/bank50_F.png", 0)
    banknote50_B = cv2.imread("Low Res Banknotes/bank50_B.png", 0)

    banknote100_F = cv2.imread("Low Res Banknotes/bank100_F.png", 0)
    banknote100_B = cv2.imread("Low Res Banknotes/bank100_B.png", 0)

    banknote200_F = cv2.imread("Low Res Banknotes/bank200_F.png", 0)
    banknote200_B = cv2.imread("Low Res Banknotes/bank200_B.png", 0)

    banknote5_F = cv2.resize(banknote5_F, size)
    banknote5_B = cv2.resize(banknote5_B, size)

    banknote10_F = cv2.resize(banknote10_F, size)
    banknote10_B = cv2.resize(banknote10_B, size)

    banknote20_F = cv2.resize(banknote20_F, size)
    banknote20_B = cv2.resize(banknote20_B, size)

    banknote50_F = cv2.resize(banknote50_F, size)
    banknote50_B = cv2.resize(banknote50_B, size)

    banknote100_F = cv2.resize(banknote100_F, size)
    banknote100_B = cv2.resize(banknote100_B, size)

    banknote200_F = cv2.resize(banknote200_F, size)
    banknote200_B = cv2.resize(banknote200_B, size)

    # Calculating histograms
    hist1 = cv2.calcHist([banknote5_F], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([banknote5_B], [0], None, [256], [0, 256])

    hist3 = cv2.calcHist([banknote10_F], [0], None, [256], [0, 256])
    hist4 = cv2.calcHist([banknote10_B], [0], None, [256], [0, 256])

    hist5 = cv2.calcHist([banknote20_F], [0], None, [256], [0, 256])
    hist6 = cv2.calcHist([banknote20_B], [0], None, [256], [0, 256])

    hist7 = cv2.calcHist([banknote50_F], [0], None, [256], [0, 256])
    hist8 = cv2.calcHist([banknote50_B], [0], None, [256], [0, 256])

    hist9 = cv2.calcHist([banknote100_F], [0], None, [256], [0, 256])
    hist10 = cv2.calcHist([banknote100_B], [0], None, [256], [0, 256])

    hist11 = cv2.calcHist([banknote200_F], [0], None, [256], [0, 256])
    hist12 = cv2.calcHist([banknote200_B], [0], None, [256], [0, 256])


def HistDif(unknownIMG):
    grayIMG = cv2.cvtColor(unknownIMG, cv2.COLOR_BGR2GRAY)
    global dif1, dif2, dif3, dif4, dif5, dif6, dif7, dif8, dif9, dif10, dif11, dif12

    hisUnk = cv2.calcHist([grayIMG], [0], None, [256], [0, 256])

    dif1 = cv2.compareHist(hist1, hisUnk, cv2.HISTCMP_CORREL)
    dif2 = cv2.compareHist(hist2, hisUnk, cv2.HISTCMP_CORREL)
    dif3 = cv2.compareHist(hist3, hisUnk, cv2.HISTCMP_CORREL)
    dif4 = cv2.compareHist(hist4, hisUnk, cv2.HISTCMP_CORREL)
    dif5 = cv2.compareHist(hist5, hisUnk, cv2.HISTCMP_CORREL)
    dif6 = cv2.compareHist(hist6, hisUnk, cv2.HISTCMP_CORREL)
    dif7 = cv2.compareHist(hist7, hisUnk, cv2.HISTCMP_CORREL)
    dif8 = cv2.compareHist(hist8, hisUnk, cv2.HISTCMP_CORREL)
    dif9 = cv2.compareHist(hist9, hisUnk, cv2.HISTCMP_CORREL)
    dif10 = cv2.compareHist(hist10, hisUnk, cv2.HISTCMP_CORREL)
    dif11 = cv2.compareHist(hist11, hisUnk, cv2.HISTCMP_CORREL)
    dif12 = cv2.compareHist(hist12, hisUnk, cv2.HISTCMP_CORREL)


takeImage()
cv2.destroyAllWindows()
