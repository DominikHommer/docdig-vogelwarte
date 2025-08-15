import numpy as np
from pathlib import Path
from pdf2image import convert_from_path
import cv2 as cv
import math
import matplotlib.pyplot as plt

class SegmentationClient:
  def __init__(self, path = '/Users/MeinNotebook/Google Drive/Meine Ablage/Scans') -> None:
     self.MAIN_DIRECTORY = path

  def rotateImg(self, img, verticalL):
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

    if len(angles_to_horizontal) == 0:
        print("Keine Winkel gefunden, daher keine Rotation.")
        return img

    median_angle = np.median(angles_to_horizontal)
    
    filtered_angles = [angle for angle in angles_to_horizontal if abs(angle - median_angle) < 0.5]

    if len(filtered_angles) == 0:
        print("Keine gültigen Winkel nach Filterung gefunden, daher keine Rotation.")
        return img

    rotation_angle = np.mean(filtered_angles)
    
    if rotation_angle < -45:
        rotation_angle += 90
    elif rotation_angle > 45:
        rotation_angle -= 90

    (h, w) = img.shape[:2]
    center = self.calculate_center_of_table(img)

    M = cv.getRotationMatrix2D(center, rotation_angle, 1.0)
    rotated_img = cv.warpAffine(img, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
    
    return rotated_img

  def cellPostProcessing(self, cell):
      ##
      # Tries to post process the cutout cells by removing border lines and noise
      # Tries to repair the word / number if some parts of it were cutout during post processing
      ##
    blurCellWidth = np.copy(cell)
    exampleCell = np.copy(cell)

    h, w = exampleCell.shape[:2]

    # Blur horizontal to make the horizontal lines "thicker"
    blurCellWidth = cv.GaussianBlur(blurCellWidth, (21, 1), 0)
    blurCellWidth = cv.GaussianBlur(blurCellWidth, (31, 1), 0)
    blurCellWidth = cv.GaussianBlur(blurCellWidth, (41, 1), 0)

    # Removes all horizontal / vertical lines in array which are higher than defined threshold
    def cleanupLines(sumArray, cellArray, threshold, totalSize, isVertical = False):
      cleanLines = True
      while (cleanLines == True):
          maxIndex = np.argmax(sumArray)

          if isVertical:
              if ((sumArray[maxIndex] / totalSize) > threshold):
                  sumArray[maxIndex] = 0
                  for i in range(0, totalSize):
                      cellArray[i][maxIndex] = 255
              else:
                  cleanLines = False
          else:
              if ((sumArray[maxIndex] / totalSize) > threshold):
                  cellArray[maxIndex] = np.ones(totalSize) * 255
                  sumArray[maxIndex] = [0]
              else:
                  cleanLines = False

      return cellArray

    # Build histograms of horizontal / vertical black pixel distributions
    horizontal_hist = blurCellWidth.shape[1] - np.sum(blurCellWidth, axis=1, keepdims=True) / 255
    vertical_hist = exampleCell.shape[0] - np.sum(exampleCell, axis=0, keepdims=True) / 255

    exampleCell = cleanupLines(horizontal_hist, exampleCell, 0.3, w)
    exampleCell = cleanupLines(vertical_hist[0], exampleCell, 0.7, h, True)

    # Try to repair cell
    invCell = 255 - cell
    removed = 255 - np.copy(exampleCell)
    repair_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2,2))
    dilate = cv.dilate(removed, repair_kernel, iterations=4)
    pre_result = cv.bitwise_and(dilate, invCell)

    result = cv.morphologyEx(pre_result, cv.MORPH_CLOSE, repair_kernel, iterations=2)
    final = cv.bitwise_and(result, invCell)

    final = cv.dilate(final, repair_kernel, iterations=2)
    final = cv.GaussianBlur(final, (5, 5), 0)
    final = cv.bitwise_and(final, invCell)

    final = 255 - final

    return final


  def getVerticalLines(self, blur, xThres = 40, minFoundLines = 3, minLineLength = 50):
    ##
    # Tries to find vertical lines of blurred image based on threshold, minFoundLines, minLineLength
    ##
    denoised = cv.fastNlMeansDenoising(blur, None, 30)
    dst_img = cv.Canny(denoised, 10, 200)
    lines = cv.HoughLinesP(dst_img, 1, np.pi / 180, 100, minLineLength= minLineLength, maxLineGap=2)
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

  def imageToCells(self, imgPath, colNr):
    ##
    # Segment image to cells for column
    ##
    img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"

    xThres = 40
    minFoundLines = 3

    # Detect vertical lines of image and try to rotate it
    gray = cv.bitwise_not(img)
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

    iHeight, iWidth = thresh.shape[:2]
    iHeightPart = math.floor(iHeight / 8)
    if iHeightPart % 2 == 0:
      iHeightPart = iHeightPart + 1

    blur = cv.GaussianBlur(thresh, (1, 31), 0)
    blur = cv.GaussianBlur(blur, (3, 1), 0)
    blur = cv.GaussianBlur(blur, (3, 21), 0)
    blur = cv.GaussianBlur(blur, (3, 31), 0)
    blur = cv.GaussianBlur(blur, (3, 1), 0)

    vl = self.getVerticalLines(blur, xThres, minFoundLines)
    if not vl:
      return

    img = self.rotateImg(img, vl)
    xThres = 40
    minFoundLines = 2

    gray = cv.bitwise_not(img)
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

    blur = cv.GaussianBlur(thresh, (5, iHeightPart), 0)
    blur = cv.GaussianBlur(blur, (3, iHeightPart), 0)
    blur = cv.GaussianBlur(blur, (1, iHeightPart), 0)

    threshRGB = cv.cvtColor(blur,cv.COLOR_GRAY2RGB)
    threshCopy = np.copy(threshRGB)

    verticalLines = self.getVerticalLines(blur, xThres, minFoundLines)

    # Skip image if no vertical lines found
    if not verticalLines:
      return

    xSplits = [[0, 0]]

    for i in range(len(verticalLines)):
        vLine = verticalLines[i]
        x1, y1, x2, y2 = vLine[0]

        # This basically gets the maximum (possible) size of the column
        start = 0
        if (i > 0):
            pLine = verticalLines[i - 1]

            px1 = pLine[0][0]
            px2 = pLine[0][2]

            start = px1
            if (px2 < px1):
                start = px2

        end = x1
        if (x2 > end):
            end = x2

        xSplits.append([start, end])
        #cv.line(threshCopy,(x1,y1),(x2,y2),(0,255,0),2)

    #cv2_imshow(threshCopy)

    xSplits.append([xSplits[-1][-1], iWidth])
    xSplits = np.sort(xSplits, axis=0)

    doneSplits = []
    for splits in xSplits:
        start, end = splits

        if start == 0:
            continue

        if end - start <= xThres:
            continue

        doneSplits.append(thresh[:, start:end + 15])

    # Skip image if not enough columns are detected
    if len(doneSplits) < 10:
      return

    colTest = doneSplits[colNr]
    iHeight, iWidth = colTest.shape[:2]

    # Make width uneven so we can use it as kernel for blurring an image
    if iWidth % 2 == 0:
      iWidth = iWidth + 1

    # TODO: remove the header cell of a column


    # Segement column into cells
    copyTest = np.copy(colTest)
    copyRGB = np.copy(copyTest)
    copyRGB = cv.cvtColor(copyRGB,cv.COLOR_GRAY2RGB)

    copyTest = cv.bitwise_not(copyTest)

    if iWidth % 2 == 0:
      iWidth = iWidth + 1

    iWidthPart = math.floor(iWidth / 4)
    if iWidthPart % 2 == 0:
      iWidthPart = iWidthPart + 1

    colBlur = cv.GaussianBlur(colTest, (iWidth, 5), 0)
    colBlur = cv.GaussianBlur(colTest, (iWidth, 3), 0)
    colBlur = cv.GaussianBlur(colTest, (iWidth, 1), 0)

    colDenoised = cv.fastNlMeansDenoising(colBlur, None, 30)
    colCanny = cv.Canny(colDenoised, 10, 200)

    houghThreshold = 100
    minLineLength = 35

    # Small cells should have "smaller" thresholds to detect lines
    if iWidth < 200:
      houghThreshold = 50
      minLineLength = 10

    lines = cv.HoughLinesP(colCanny, 1, np.pi / 180, threshold=houghThreshold, minLineLength=minLineLength, maxLineGap=2)

    # Skip image if no horizontal lines found
    if lines is None:
      return

    # We define a 30px y-Threshold, which "decides" if a line is indeed a horizontal detected line
    yThres = 30
    horizontalLines = []
    for line in lines:
        x1,y1,x2,y2 = line[0]

        # Skip value, as it is not a horizontal line
        if not ((y1 + yThres) > y2 and (y1 - yThres) < y2):
            continue

        shouldAdd = True
        for i in range(len(horizontalLines)):
            hLine = horizontalLines[i]
            h_x1, h_y1, h_x2, h_y2 = hLine[0]

            # Check if "line" Vector is above / under "hLine" Vector and in its y-threshold
            # Update horizontalLines List accordingly
            if (x1 < h_x1) and ((h_y1 + yThres) > y1 and (h_y1 - yThres) < y1):
                horizontalLines[i] = [[x1, y1, h_x2, h_y2], hLine[1]+1]

                shouldAdd = False
            elif (x2 > h_x2) and ((h_y2 + yThres) > y2 and (h_y2 - yThres) < y2):
                horizontalLines[i] = [[h_x1, h_y1, x2, y2], hLine[1]+1]

                shouldAdd = False

        if shouldAdd == True:
            horizontalLines.append([line[0], 0])

    minFoundLines = 1
    if iWidth < 200:
      minFoundLines = 0

    ySplits = [[0, 0]]

    horizontalLines = list(filter(lambda l: l[1] >= minFoundLines, horizontalLines))

    for i in range(len(horizontalLines)):
        hLine = horizontalLines[i]
        x1, y1, x2, y2 = hLine[0]

        # This basically gets the maximum (possible) height of the row
        start = 0
        if (i > 0):
            pLine = horizontalLines[i - 1]

            py1 = pLine[0][1]
            py2 = pLine[0][3]

            start = py1
            if (py2 < py1):
                start = py2

        end = y1
        if (y2 > end):
            end = y2

        ySplits.append([start, end])
        #cv.line(copyRGB,(x1,y1),(x2,y2),(0,255,0),2)

    #cv2_imshow(copyRGB)

    ySplits.append([ySplits[-1][-1], iHeight])
    ySplits = np.sort(ySplits, axis=0)

    doneRowSplits = []
    for splits in ySplits:
        start, end = splits

        if start == 0:
            continue

        if end - start <= yThres:
            continue

        doneRowSplits.append(copyTest[start:end + 15,:])

    return doneRowSplits

  def pdf_scan_to_cells_of_columns(self, path, columnNumber):
    ##
    # Transform pdf into cells of defined column and safe them as seperate images into a folder
    ##
    fileName = path[-path.rfind('/') - 1:]
    pdf_images = convert_from_path(path)

    dPath = self.MAIN_DIRECTORY + "/gen_img/" + fileName
    Path(dPath).mkdir(parents=True, exist_ok=True)
    Path(dPath + "/cells/").mkdir(parents=True, exist_ok=True)
    Path(dPath + "/cells/col-" + str(columnNumber)).mkdir(parents=True, exist_ok=True)

    result = []
    for idx in range(len(pdf_images)):
        imgPath = dPath + "/" + str(idx+1) +'.png'
        pdf_images[idx].save(imgPath, 'PNG')

        result.append([])

        cells = self.imageToCells(imgPath, columnNumber)
        if not cells:
          continue

        for cidx, cell in enumerate(cells):
            cell = self.cellPostProcessing(cell)

            result[idx].append(cell)
            #cv.imwrite(dPath + "/cells/col-" + str(columnNumber) + "/" + str(idx) + '-' + str(cidx) + '.png', cell)

    return result
  
  def calculate_center_of_table(self, table):
    #gray = cv.cvtColor(table, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(table, 150, 255, cv.THRESH_BINARY_INV)
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv.contourArea)
    
    (h, w) = table.shape[:2]
    default_center = (w // 2, h // 2)
    
    if contour is None:
      return default_center
    
    global_min_y = float('inf')
    global_max_y = float('-inf')

    global_min_y = min(point[0][1] for point in contour)
    global_max_y = max(point[0][1] for point in contour)

    middle_y = (global_min_y + global_max_y) / 2

    upper_half = [point[0] for point in contour if point[0][1] < middle_y]
    lower_half = [point[0] for point in contour if point[0][1] >= middle_y]
    
    top_left = (min(point[0] for point in upper_half), min(point[1] for point in upper_half))
    top_right = (max(point[0] for point in upper_half), min(point[1] for point in upper_half))
    bottom_left = (min(point[0] for point in lower_half), max(point[1] for point in lower_half))
    bottom_right = (max(point[0] for point in lower_half), max(point[1] for point in lower_half))
    
    center_x = (top_left[0] + top_right[0] + bottom_right[0] + bottom_left[0]) // 4
    center_y = (top_left[1] + top_right[1] + bottom_right[1] + bottom_left[1]) // 4
    center = (center_x, center_y)
         
    allowed_deviation = 100 
    if (abs(center[0] - default_center[0]) > allowed_deviation or abs(center[1] - default_center[1]) > allowed_deviation):
        return default_center
    
    return center
    