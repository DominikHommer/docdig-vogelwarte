import cv2
import numpy as np

def getYStartEndForLine(i, lines):
    hLine = lines[i]
    _, y1, _, y2 = hLine[0]

    # This basically gets the maximum (possible) height of the row
    start = 0
    if (i > 0):
        pLine = lines[i - 1]

        py1 = pLine[0][1]
        py2 = pLine[0][3]

        start = py1
        if (py2 < py1):
            start = py2

    end = y1
    if (y2 > end):
        end = y2

    return start, end

def rotateImg(img, verticalL):
        ##
        # Rotate skewed image based on vertical lines
        ##
        verticalL.sort(key=lambda x: x[0][0])
        filteredVerticalLines = [verticalL[0]]

        for i in range(1, len(verticalL)):
            current_line = verticalL[i]
            prev_line = verticalL[i-1]
            distance = current_line[0][0] - prev_line[0][0]

            if distance > 120:
                filteredVerticalLines.append(current_line)

        angles_to_horizontal = []

        for line in filteredVerticalLines:
            x1, y1, x2, y2 = line[0]

            if x2 - x1 != 0:
                slope = (y2 - y1) / (x2 - x1)
                angle_to_horizontal = np.degrees(np.arctan(slope))
                angles_to_horizontal.append(angle_to_horizontal)
            else :
                angles_to_horizontal.append(-90)

        filtered_angles = [90 - angle if angle > 0 else angle for angle in angles_to_horizontal]
        rotation_angle = filtered_angles[0]

        if rotation_angle < 0:
            rotation_angle = 90 + rotation_angle

        (h, w) = img.shape[:2]
        center = (0,0)
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def getVerticalLines(blur, xThres = 40, minFoundLines = 3, minLineLength = 50):
    ##
    # Tries to find vertical lines of blurred image based on threshold, minFoundLines, minLineLength
    ##
    denoised = cv2.fastNlMeansDenoising(blur, None, 30)
    dst_img = cv2.Canny(denoised, 10, 200)
    lines = cv2.HoughLinesP(dst_img, 1, np.pi / 180, 100, minLineLength= minLineLength, maxLineGap=2)

    if lines is None:
      return []

    # x-Threshold "decides" if a line is indeed a vertical detected line
    verticalLines = []
    for line in lines:
        x1,y1,x2,y2 = line[0]

        # Skip value, as it is not a vertical line
        if not ((x1 + xThres) > x2 and (x1 - xThres) < x2):
            continue

        shouldAdd = True
        for i in range(len(verticalLines)):
            vLine = verticalLines[i]
            v_x1, v_y1, v_x2, v_y2 = vLine[0]

            # Check if "line" Vector is above / under "vLine" Vector and in its x-threshold
            # Update verticalLines List accordingly
            if (y1 < v_y1) and ((v_x1 + xThres) > x1 and (v_x1 - xThres) < x1):
                verticalLines[i] = [[x1, y1, v_x2, v_y2], vLine[1]+1]

                shouldAdd = False
            elif (y2 > v_y2) and ((v_x2 + xThres) > x2 and (v_x2 - xThres) < x2):
                verticalLines[i] = [[v_x1, v_y1, x2, y2], vLine[1]+1]

                shouldAdd = False

        if shouldAdd == True:
            verticalLines.append([line[0], 0])

    # At least minFoundLines in a x-threshold must be found, otherwise we consider the line as invalid detected
    return list(filter(lambda l: l[1] >= minFoundLines, verticalLines))