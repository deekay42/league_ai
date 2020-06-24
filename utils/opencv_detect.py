import cv2 as cv
import numpy as np
import math
from shapely.geometry import *
import sys
from utils import misc

LINE_CERTAINTY_THRESHOLD = 20
MAX_IMAGE_ROTATION = 30
minLineLength = 500
maxLineGap = 50
MIN_CONTOUR_LENGTH = 2000

MAX_LINE_GAP = 35
MAX_LINE_ANGLE_DIFF = 2

num_down = 2  # number of downsampling steps
num_bilateral = 10  # number of bilateral filtering steps


def equivalencePartition(iterable, relation):
    """Partitions a set of objects into equivalence classes

    Args:
        iterable: collection of objects to be partitioned
        relation: equivalence relation. I.e. relation(o1,o2) evaluates to True
            if and only if o1 and o2 are equivalent

    Returns: classes, partitions
        classes: A sequence of sets. Each one is an equivalence class
        partitions: A dictionary mapping objects to equivalence classes
    """
    classes = []
    partitions = {}
    for o in iterable:  # for each object
        # find the class it is in
        found = False
        for c in classes:
            if relation(next(iter(c)), o):  # is it equivalent to this class?
                c.append(o)
                found = True
                break
        if not found:  # it is in a new class
            classes.append([o])
    return classes


def averageNormalizedLines(lines):
    length = len(lines)
    x0 = sum([line.coords[0][0] for line in lines]) // length
    y0 = sum([line.coords[0][1] for line in lines]) // length
    x1 = sum([line.coords[1][0] for line in lines]) // length
    y1 = sum([line.coords[1][1] for line in lines]) // length

    return LineString([(x0, y0), (x1, y1)])


def intersecOfTwoLines(l1, l2):
    return intersecOfTwoLinesGiven2PointsOnEach(l1.coords[0][0], l1.coords[0][1], l1.coords[1][0], l1.coords[1][1],
                                                l2.coords[0][0],
                                                l2.coords[0][1], l2.coords[1][0], l2.coords[1][1])


def intersecOfTwoLinesGiven2PointsOnEach(x1, y1, x2, y2, x3, y3, x4, y4):
    x0 = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
            (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    y0 = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
            (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    return x0, y0


def dot(vA, vB):
    return vA[0] * vB[0] + vA[1] * vB[1]


def angle(lineA, lineB):
    # Get nicer vector form
    vA = [(lineA[0][0] - lineA[1][0]), (lineA[0][1] - lineA[1][1])]
    vB = [(lineB[0][0] - lineB[1][0]), (lineB[0][1] - lineB[1][1])]
    dot_prod = dot(vA, vB)
    magA = math.sqrt(dot(vA, vA))
    magB = math.sqrt(dot(vB, vB))
    angle = math.acos(dot_prod / (magB * magA))
    angle = math.degrees(angle) % 180
    if angle > 90:
        return 180 - angle
    else:
        return angle


def lineSegmentsAreEqual(l1, l2):
    m = angle(l1.coords, l2.coords)
    if m > MAX_LINE_ANGLE_DIFF:
        return False
    if m == 0:
        if l1.distance(l2) < MAX_LINE_GAP:
            return True
        else:
            return False
    if l1.distance(l2) < MAX_LINE_GAP:
        return True
    else:
        return False


def stretchLineSegmentstoFitFrame(l1, frameBounds):
    try:
        # left side border
        left_border_intersection = intersecOfTwoLines(l1, LineString(
            [(frameBounds[0][0], frameBounds[0][1]), (frameBounds[0][0], frameBounds[1][1])]))
        # right side border
        right_border_intersection = intersecOfTwoLines(l1, LineString(
            [(frameBounds[1][0], frameBounds[0][1]), (frameBounds[1][0], frameBounds[1][1])]))
    except ZeroDivisionError:
        # lines are parallel to these borders
        left_border_intersection = right_border_intersection = None
    try:
        # upper side border
        upper_border_intersection = intersecOfTwoLines(l1, LineString(
            [(frameBounds[0][0], frameBounds[0][1]), (frameBounds[1][0], frameBounds[0][1])]))
        # lower side border
        lower_border_intersection = intersecOfTwoLines(l1, LineString(
            [(frameBounds[0][0], frameBounds[1][1]), (frameBounds[1][0], frameBounds[1][1])]))
    except ZeroDivisionError:
        # lines are parallel to these borders
        upper_border_intersection = lower_border_intersection = None

    # handle lines parallel to the frame borders
    if left_border_intersection is None:
        return LineString([upper_border_intersection, lower_border_intersection])
    if upper_border_intersection is None:
        return LineString([left_border_intersection, right_border_intersection])

    # handle all the other cases
    if insideBounds(left_border_intersection, frameBounds):
        if insideBounds(right_border_intersection, frameBounds):
            return LineString([left_border_intersection, right_border_intersection])
        if insideBounds(upper_border_intersection, frameBounds):
            return LineString([left_border_intersection, upper_border_intersection])
        else:
            return LineString([left_border_intersection, lower_border_intersection])
    if insideBounds(upper_border_intersection, frameBounds):
        if insideBounds(lower_border_intersection, frameBounds):
            return LineString([upper_border_intersection, lower_border_intersection])
        else:
            return LineString([upper_border_intersection, right_border_intersection])
    else:
        return LineString([right_border_intersection, lower_border_intersection])


def insideBounds(point, frameBounds):
    return (frameBounds[0][0] <= point[0] <= frameBounds[1][0]) and (
            frameBounds[0][1] <= point[1] <= frameBounds[1][1])


def boundingBoxesAreEqual(box1, box2):
    return math.fabs(box1[0][0] - box2[0][0]) <= 20 and \
           math.fabs(box1[1][1] - box2[1][1]) <= 20 and \
           math.fabs(box1[0][1] - box2[0][1]) <= 20 and \
           math.fabs(box1[1][0] - box2[1][0]) <= 20


def normalizeItemBoundingBoxes(box_partition):
    size = len(box_partition)
    x0 = sum([rect[0][0] for rect in box_partition]) // size
    y0 = sum([rect[0][1] for rect in box_partition]) // size
    x1 = sum([rect[1][0] for rect in box_partition]) // size
    y1 = sum([rect[1][1] for rect in box_partition]) // size
    return ((x0, y0), (x1, y1))


def preprocessImage(img_gray):
    # downsample image using Gaussian pyramid

    # for _ in range(num_down):
    #     img_color = cv.pyrDown(img_color)

    # cv.namedWindow("downsampled", cv.WINDOW_NORMAL)
    # cv.imshow('downsampled', img_color)
    # cv.waitKey(0)

    # repeatedly apply small bilateral filter instead of
    # applying one large filter
    img_blurred = img_gray.copy()
    for _ in range(num_bilateral):
        img_blurred = cv.bilateralFilter(img_blurred, d=5,
                                         sigmaColor=9,
                                         sigmaSpace=7)
    for i in range(num_bilateral):
        img_blurred = cv.medianBlur(img_blurred, 5)

    img_blurred = cv.adaptiveThreshold(img_blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blockSize=91,
                                       C=-2)
    # th = np.append([th], [cv.adaptiveThreshold(blurred_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 0)], axis=0)
    # th = np.append(th, [cv.adaptiveThreshold(blurred_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, 0)], axis=0)
    # th = np.append(th, [cv.adaptiveThreshold(blurred_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 31, 0)], axis=0)
    # th = np.append(th, [cv.adaptiveThreshold(blurred_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 41, 0)], axis=0)
    # th = np.append(th, [cv.adaptiveThreshold(blurred_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 51, 0)], axis=0)
    # th = np.append(th, [cv.adaptiveThreshold(blurred_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 61, 0)], axis=0)
    # th = np.append(th, [cv.adaptiveThreshold(blurred_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 71, 0)], axis=0)
    # th = np.append(th, [cv.adaptiveThreshold(blurred_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 81, 0)], axis=0)
    # th = np.append(th, [cv.adaptiveThreshold(blurred_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 91, 0)], axis=0)
    # th = np.append(th, [cv.adaptiveThreshold(blurred_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 101, 0)], axis=0)
    #
    # for i,v in enumerate(th):
    #     cv.imwrite(str(i)+'.png', v)
    # #

    # kernel = np.ones((3, 3), np.uint8)

    # erosion = cv.erode(th,kernel,iterations = 2)
    # cv.namedWindow("dilation", cv.WINDOW_NORMAL)
    # cv.imshow('dilation', erosion)
    # cv.waitKey(0)
    # dilation = cv.dilate(th,kernel,iterations = 4)
    # cv.namedWindow("dilation", cv.WINDOW_NORMAL)
    # cv.imshow('dilation', dilation)
    # cv.waitKey(0)

    # finding contours

    # im2, contours, hierarchy = cv.findContours(th, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #
    # longContours = [cnt for cnt in contours if cv.arcLength(cnt,True) > MIN_CONTOUR_LENGTH]
    # longContours.sort(key=lambda x:cv.arcLength(x,True))
    #
    # cnt=longContours[-1]
    # epsilon = 0.*cv.arcLength(cnt,True)
    # approx = cv.approxPolyDP(cnt,epsilon,True)
    #
    # cv.drawContours(img_rgb, [cnt], -1, (0,255,255), 10)
    #
    # cv.namedWindow("dilation", cv.WINDOW_NORMAL)
    # cv.imshow('dilation', img_rgb)
    # cv.waitKey(0)

    return img_blurred


def findScoreboardBoundaryLines(img_blurred):
    hlines = cv.HoughLinesP(img_blurred, 1, np.pi / 180, threshold=100, minLineLength=minLineLength,
                            maxLineGap=maxLineGap)
    imgBoundary = ((0, 0), (len(img_blurred[0]), len(img_blurred)))
    lines = []
    for line in hlines:
        old_line = LineString([(line[0][0], line[0][1]), (line[0][2], line[0][3])])
        stretched_line = stretchLineSegmentstoFitFrame(old_line, imgBoundary)
        lines.append([stretched_line, old_line])

    classes = equivalencePartition(lines, lambda x, y: lineSegmentsAreEqual(x[0], y[0]))
    horizontalLine = LineString([(0, 0), (100, 0)])
    verticalLine = LineString([(0, 0), (0, 100)])
    horizontalLines = [line for line in classes if
                       angle(line[0][0].coords, horizontalLine.coords) <= MAX_IMAGE_ROTATION]
    verticalLines = [line for line in classes if angle(line[0][0].coords, verticalLine.coords) <= MAX_IMAGE_ROTATION]

    # which cluster has the longest, aka most significant lines?
    horizontalLines.sort(key=lambda lineCluster: sum(line[1].length ** 2 for line in lineCluster))
    verticalLines.sort(key=lambda lineCluster: sum(line[1].length ** 2 for line in lineCluster))
    significantLines = horizontalLines[-5:] + verticalLines[-5:]
    lines = [averageNormalizedLines([line[0] for line in lineCluster]) for lineCluster in significantLines]

    # cv.namedWindow("dilaftion", cv.WINDOW_NORMAL)
    # cv.imshow('dilaftion', img_blurred)
    # cv.waitKey(0)
    # for line in lines:
    #     cv.line(img_rgb, (int(line.coords[0][0]), int(line.coords[0][1])), (int(line.coords[1][0]), int(line.coords[1][1])), (0, 255, 255), 5)
    # cv.namedWindow("dilation", cv.WINDOW_NORMAL)
    # cv.imshow('dilation', img_rgb)
    # cv.waitKey(0)

    img_x_size = len(img_blurred[0])
    img_y_size = len(img_blurred)
    # super crude selection logic.. TODO
    for line in lines:
        if (1 / 20) * img_x_size <= line.coords[0][0] <= (1 / 4) * img_x_size:
            scoreboard_left_border = line
        if (3 / 4) * img_x_size <= line.coords[0][0] <= (19 / 20) * img_x_size:
            scoreboard_right_border = line
        if (1 / 4) * img_y_size <= line.coords[1][1] <= (1 / 2) * img_y_size:
            scoreboard_top_border = line
        if (1 / 2) * img_y_size <= line.coords[1][1] <= (4 / 5) * img_y_size:
            scoreboard_bot_border = line

    return scoreboard_left_border, scoreboard_right_border, scoreboard_top_border, scoreboard_bot_border


def buildScoreboardRawRect(scoreboard_left_border, scoreboard_right_border, scoreboard_top_border,
                           scoreboard_bot_border):
    top_left_corner = scoreboard_left_border.intersection(scoreboard_top_border)
    top_left_corner = (int(top_left_corner.x), int(top_left_corner.y))
    bottom_left_corner = scoreboard_left_border.intersection(scoreboard_bot_border)
    bottom_left_corner = (int(bottom_left_corner.x), int(bottom_left_corner.y))
    top_right_corner = scoreboard_right_border.intersection(scoreboard_top_border)
    top_right_corner = (int(top_right_corner.x), int(top_right_corner.y))
    bottom_right_corner = scoreboard_right_border.intersection(scoreboard_bot_border)
    bottom_right_corner = (int(bottom_right_corner.x), int(bottom_right_corner.y))

    raw_boundary = np.array([top_left_corner, top_right_corner, bottom_right_corner, bottom_left_corner],
                            dtype=np.float32)

    return raw_boundary


def buildScoreboardDesiredRect(raw_boundary):
    boundary_bounding_box = LineString(raw_boundary).bounds
    boundary_bounding_box = ((int(boundary_bounding_box[0]), int(boundary_bounding_box[1])),
                             (int(boundary_bounding_box[2]), int(boundary_bounding_box[3])))
    boundary_bounding_box = np.array(
        [boundary_bounding_box[0], (boundary_bounding_box[1][0], boundary_bounding_box[0][1]), boundary_bounding_box[1],
         (boundary_bounding_box[0][0], boundary_bounding_box[1][1])], dtype=np.float32)

    scoreboard_width = math.fabs(boundary_bounding_box[2][0] - boundary_bounding_box[0][0])
    scoreboard_height = math.fabs(boundary_bounding_box[2][1] - boundary_bounding_box[0][1])
    desired_box = boundary_bounding_box

    required_height = scoreboard_width * (STD_SCOREBOARD_HEIGHT / STD_SCOREBOARD_WIDTH)
    required_width = scoreboard_height * (STD_SCOREBOARD_WIDTH / STD_SCOREBOARD_HEIGHT)

    if scoreboard_height < required_height:
        scoreboard_height = desired_box[2][1] = desired_box[3][1] = desired_box[0][1] + required_height
    elif scoreboard_width < required_width:
        scoreboard_width = desired_box[1][0] = desired_box[2][0] = desired_box[0][0] + required_width

    return desired_box


#
# cv.rectangle(img_rgb, (int(boundary_bounding_box[0][0]), int(boundary_bounding_box[0][1])), (int(boundary_bounding_box[2][0]), int(boundary_bounding_box[2][1])), (0,255,255), 5)
# # cv.polylines(img_rgb, raw_boundary, True, (0,255,255), 5)
# cv.namedWindow("dilation", cv.WINDOW_NORMAL)
# cv.imshow('dilation', img_rgb)
# cv.waitKey(0)


# cv.rectangle(img_warped, (int(boundary_bounding_box[0][0]), int(boundary_bounding_box[0][1])), (int(boundary_bounding_box[2][0]), int(boundary_bounding_box[2][1])), (0,255,255), 5)
# # cv.polylines(img_rgb, raw_boundary, True, (0,255,255), 5)
# for line in [scoreboard_bot_border, scoreboard_top_border, scoreboard_left_border, scoreboard_right_border]:
#     # line = LineString([(lul[0][0], lul[0][1]),(lul[0][2],lul[0][3])])
#     cv.line(img_rgb, (int(line.coords[0][0]), int(line.coords[0][1])), (int(line.coords[1][0]), int(line.coords[1][1])), (0, 255, 0), 5)
#
# cv.namedWindow("dilation", cv.WINDOW_NORMAL)
# cv.imshow('dilation', img_rgb)
# cv.waitKey(0)

# TODO: Make absolute pixel values relative!! Pixel values differ from camera to camera

def locateIncompleteBoundingBoxCoordinates(templates, size, img):
    templates = [cv.resize(template, (size, size), interpolation=cv.INTER_AREA) for template in templates]

    normalized_boxes = []

    for template in templates:
        res = cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED)
        threshold = 0.7
        loc = np.where(res >= threshold)
        bounding_boxes = [(pt, (pt[0] + size, pt[1] + size)) for pt in zip(*loc[::-1])]
        partitioned_boxes = equivalencePartition(bounding_boxes, boundingBoxesAreEqual)

        normalized_boxes = normalized_boxes + [normalizeItemBoundingBoxes(partition) for partition in partitioned_boxes]
    return normalized_boxes


# for item_pt in normalized_spell_boxes:
#     cv.rectangle(img_warped, item_pt[0], item_pt[1], (255, 255, 0), 2)
#
# cv.namedWindow("results", cv.WINDOW_NORMAL)
# cv.imshow('results', img_warped)
# cv.waitKey(0)


def determineXOffsets(normalized_boxes):
    left_side_x = normalized_boxes[0][0][0]
    right_side_x = 0
    for box in normalized_boxes:
        if math.fabs(box[0][0] - left_side_x) >= 50:
            right_side_x = box[0][0]
            break

    assert right_side_x != 0

    if left_side_x > right_side_x:
        left_side_x, right_side_x = right_side_x, left_side_x
    return left_side_x, right_side_x


def sortAndSplitIntoLeftRight(normalized_boxes, left_side_x, right_side_x):
    left_side_boxes = [box for box in normalized_boxes if math.fabs(left_side_x - box[0][0]) < 100]
    right_side_boxes = [box for box in normalized_boxes if math.fabs(right_side_x - box[0][0]) < 100]

    assert len(left_side_boxes) != 0 and len(right_side_boxes) != 0
    left_side_boxes.sort(key=lambda box: box[1][1])
    right_side_boxes.sort(key=lambda box: box[1][1])
    return left_side_boxes, right_side_boxes


def determineYOffset(normalized_boxes, min_diff):
    boxes_vert_diff = sys.maxsize
    for i in range(len(normalized_boxes)):
        for j in range(i, len(normalized_boxes)):
            y_diff = math.fabs(normalized_boxes[i][1][1] - normalized_boxes[j][1][1])
            # noise
            if y_diff <= min_diff:
                continue
            if y_diff < boxes_vert_diff:
                boxes_vert_diff = y_diff
    return boxes_vert_diff


def projectBoxColumnToBottom(box_column, boxes_vert_diff, bottom):
    while math.fabs(box_column[-1][1][1] - bottom) > 100:
        new_y = box_column[-1][0][1] + boxes_vert_diff
        new_x = box_column[-1][0][0]
        box_column.append(((new_x, new_y), (new_x + ITEM_SLOT_HEIGHT, new_y + ITEM_SLOT_HEIGHT)))


def getItemCoordinates(left_side_boxes, right_side_boxes, boxes_vert_diff, left_side_x, right_side_x, size, bottom):
    if math.fabs(left_side_boxes[-1][1][1] - bottom) < 100:
        top_left_trinket_y = left_side_boxes[-1][0][1] - 4 * boxes_vert_diff
        item_coordinates = misc._generate_item_coords(size, left_side_x, right_side_x, boxes_vert_diff, top_left_trinket_y)
    elif math.fabs(right_side_boxes[-1][1][1] - bottom) < 100:
        top_left_trinket_y = right_side_boxes[-1][0][1] - 4 * boxes_vert_diff
        item_coordinates = misc._generate_item_coords(size, left_side_x, right_side_x, boxes_vert_diff, top_left_trinket_y)
    else:
        projectBoxColumnToBottom(left_side_boxes, boxes_vert_diff, bottom)
        top_left_trinket_y = left_side_boxes[-1][0][1] - 4 * boxes_vert_diff
        item_coordinates = misc._generate_item_coords(size, left_side_x, right_side_x, boxes_vert_diff, top_left_trinket_y)
    return item_coordinates


def lowestBoxIsLower(box_column, SPELL_SLOT_HEIGHT):
    return math.fabs(box_column[-1][0][1] - box_column[-2][0][1] - SPELL_SLOT_HEIGHT) < 40


def getSpellCoordinates(left_side_boxes, right_side_boxes, boxes_vert_diff, left_side_x, right_side_x, size, bottom):
    if math.fabs(left_side_boxes[-1][1][1] - bottom) < 100:
        top_left_spell_y = left_side_boxes[-1][0][1] - 4 * (boxes_vert_diff + size) - size
        spell_coordinates = misc.generateSpellCoordinates(size, left_side_x, right_side_x, boxes_vert_diff, top_left_spell_y)
    elif math.fabs(right_side_boxes[-1][1][1] - bottom) < 100:
        top_left_spell_y = right_side_boxes[-1][0][1] - 4 * (boxes_vert_diff + size) - size
        spell_coordinates = misc.generateSpellCoordinates(size, left_side_x, right_side_x, boxes_vert_diff, top_left_spell_y)
    else:
        lowest_box_is_lower = lowestBoxIsLower(left_side_boxes, size)
        projectSpellColumnToBottom(left_side_boxes, boxes_vert_diff, bottom, lowest_box_is_lower, size)
        top_left_spell_y = left_side_boxes[-1][0][1] - 4 * (boxes_vert_diff + size) - size
        spell_coordinates = misc.generateSpellCoordinates(size, left_side_x, right_side_x, boxes_vert_diff, top_left_spell_y)
    return spell_coordinates


def projectSpellColumnToBottom(box_column, boxes_vert_diff, bottom, lowest_box_is_lower, box_size):
    if not lowest_box_is_lower:
        new_y = box_column[-1][0][1] + box_size
        new_x = box_column[-1][0][0]
        box_column.append(((new_x, new_y), (new_x + box_size, new_y + box_size)))
    if math.fabs(box_column[-1][1][1] - bottom) < 100:
        return
    while math.fabs(box_column[-1][1][1] - bottom) > 100:
        new_y_1 = box_column[-1][0][1] + boxes_vert_diff
        new_x = box_column[-1][0][0]
        new_y_2 = new_y_1 + box_size
        box_column.append(((new_x, new_y_1), (new_x + box_size, new_y_1 + box_size)))
        box_column.append(((new_x, new_y_2), (new_x + box_size, new_y_2 + box_size)))


def fixWarp(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_blurred = preprocessImage(img_gray)
    scoreboard_left_border, scoreboard_right_border, scoreboard_top_border, scoreboard_bot_border = findScoreboardBoundaryLines(
        img_blurred)
    raw_boundary = buildScoreboardRawRect(scoreboard_left_border, scoreboard_right_border, scoreboard_top_border,
                                          scoreboard_bot_border)
    desired_box = buildScoreboardDesiredRect(raw_boundary)
    scoreboard_width = desired_box[2][0] - desired_box[0][0]
    perspectiveTransform = cv.getPerspectiveTransform(raw_boundary, desired_box)
    img_warped = cv.warpPerspective(img, perspectiveTransform, dsize=(img_gray.shape[1], img_gray.shape[0]))
    ITEM_SLOT_HEIGHT = int((26 / STD_SCOREBOARD_WIDTH) * scoreboard_width)
    SPELL_SLOT_HEIGHT = int((20 / STD_SCOREBOARD_WIDTH) * scoreboard_width)
    CHAMP_SLOT_HEIGHT = 2 * SPELL_SLOT_HEIGHT
    img_scoreboard = img_warped[int(desired_box[0][1]):int(desired_box[2][1]), int(desired_box[0][0]):int(desired_box[2][0])]
    return img_warped, img_scoreboard, ITEM_SLOT_HEIGHT, SPELL_SLOT_HEIGHT, CHAMP_SLOT_HEIGHT


def processScoreboard(img_scoreboard, ITEM_SLOT_HEIGHT, SPELL_SLOT_HEIGHT, CHAMP_SLOT_HEIGHT):
    # cv.namedWindow("results", cv.WINDOW_NORMAL)
    # cv.imshow('results', img_scoreboard)
    # cv.waitKey(0)
    item_templates = misc.getItemTemplateDict()
    item_templates = [item_templates["Sweeping Lens (Trinket)"], item_templates["Warding Totem (Trinket)"],
                      item_templates["Farsight Alteration"]]
    # item_templates_gray = [cv.cvtColor(template, cv.COLOR_BGR2GRAY) for template in item_templates]

    spell_templates = misc.getSpellTemplateDict()
    spell_templates = [spell_templates["Teleport"], spell_templates["Flash"], spell_templates["Heal"]]
    # spell_templates_gray = [cv.cvtColor(template, cv.COLOR_BGR2GRAY) for template in spell_templates]

    normalized_item_boxes = locateIncompleteBoundingBoxCoordinates(item_templates, ITEM_SLOT_HEIGHT, img_scoreboard)
    normalized_spell_boxes = locateIncompleteBoundingBoxCoordinates(spell_templates, SPELL_SLOT_HEIGHT, img_scoreboard)

    # for item_pt in normalized_item_boxes:
    #     cv.rectangle(img_scoreboard, (item_pt[0]), (item_pt[1]), (255, 255, 0), 2)
    #
    # for item_pt in normalized_spell_boxes:
    #     cv.rectangle(img_scoreboard, (item_pt[0]), (item_pt[1]), (255, 255, 0), 2)
    # cv.namedWindow("results", cv.WINDOW_NORMAL)
    # cv.imshow('results', img_scoreboard)
    # cv.waitKey(0)
    items_left_side_x, items_right_side_x = determineXOffsets(normalized_item_boxes)

    spells_left_side_x, spells_right_side_x = determineXOffsets(normalized_spell_boxes)
    left_side_item_boxes, right_side_item_boxes = sortAndSplitIntoLeftRight(normalized_item_boxes, items_left_side_x,
                                                                            items_right_side_x)
    left_side_spell_boxes, right_side_spell_boxes = sortAndSplitIntoLeftRight(normalized_spell_boxes,
                                                                              spells_left_side_x,
                                                                              spells_right_side_x)
    item_boxes_vert_diff = determineYOffset(left_side_item_boxes, 50)
    spell_boxes_vert_diff = determineYOffset(left_side_spell_boxes, SPELL_SLOT_HEIGHT + 20)
    item_coordinates = getItemCoordinates(left_side_item_boxes, right_side_item_boxes, item_boxes_vert_diff,
                                          items_left_side_x, items_right_side_x, ITEM_SLOT_HEIGHT,
                                          img_scoreboard.shape[0])
    spell_coordinates = getSpellCoordinates(left_side_spell_boxes, right_side_spell_boxes, spell_boxes_vert_diff,
                                            spells_left_side_x, spells_right_side_x, SPELL_SLOT_HEIGHT,
                                            img_scoreboard.shape[0])

    champ_coordinates = misc.generateChampionCoordsBasedOnSpellCoords(spell_coordinates[0], spell_coordinates[1])

    return item_coordinates, spell_coordinates, champ_coordinates, ITEM_SLOT_HEIGHT, SPELL_SLOT_HEIGHT, CHAMP_SLOT_HEIGHT

#
# for item_pt in item_coordinates.reshape((2 * 5 * 6, 2)):
#     cv.rectangle(img_scoreboard, (int(item_pt[0]), int(item_pt[1])),
#                  (int(item_pt[0] + ITEM_SLOT_HEIGHT), int(item_pt[1] + ITEM_SLOT_HEIGHT)), (255, 255, 0), 2)
#
# for item_pt in spell_coordinates.reshape((2 * 5 * 2, 2)):
#     cv.rectangle(img_scoreboard, (int(item_pt[0]), int(item_pt[1])),
#                  (int(item_pt[0] + SPELL_SLOT_HEIGHT), int(item_pt[1] + SPELL_SLOT_HEIGHT)), (255, 255, 0), 2)
#
# for item_pt in champ_coordinates.reshape((2 * 5, 2)):
#     cv.rectangle(img_scoreboard, (int(item_pt[0]), int(item_pt[1])),
#                  (int(item_pt[0] + CHAMP_SLOT_HEIGHT), int(item_pt[1] + CHAMP_SLOT_HEIGHT)), (255, 255, 0), 2)
#
# cv.namedWindow("results", cv.WINDOW_NORMAL)
# cv.imshow('results', img_scoreboard)
# cv.waitKey(0)

def bestMatch(templates, size, img_area):
    high_score = -1
    high_score_name = ''
    for name, template in templates.items():
        res = cv.matchTemplate(img_area, template, cv.TM_CCOEFF_NORMED)
        max = np.amax(res)
        if max > high_score:
            high_score = max
            high_score_name = name
    return high_score_name


# sharp coordinate bounding boxes will give poor results for convolution. relax bounding boxes a little bit
def addPaddingToBoundingBox(img, top_left_corner, slot_height, overspill):
    top_left_corner_x = top_left_corner[0]
    top_left_corner_y = top_left_corner[1]
    top_left_corner = (top_left_corner_x - overspill, top_left_corner_y - overspill)
    bottom_right_corner = (top_left_corner_x + slot_height + overspill,
                           top_left_corner_y + slot_height + overspill)
    return img[top_left_corner[1]: bottom_right_corner[1], top_left_corner[0]: bottom_right_corner[0]]



def getScoreboardStatsImgs():
    img_rgb = cv.imread('res/sample.jpg')
    img_warped, img_scoreboard, ITEM_SLOT_HEIGHT, SPELL_SLOT_HEIGHT, CHAMP_SLOT_HEIGHT = fixWarp(img_rgb)
    item_coordinates, spell_coordinates, champ_coordinates, ITEM_SLOT_HEIGHT, SPELL_SLOT_HEIGHT, CHAMP_SLOT_HEIGHT = processScoreboard(
        img_scoreboard, ITEM_SLOT_HEIGHT, SPELL_SLOT_HEIGHT, CHAMP_SLOT_HEIGHT)

    CHAMP_CONV_PAD = 5
    ITEM_CONV_PAD = 5
    SPELL_CONV_PAD = 5

    # TODO: make overlap dependent on whether spell is top or bot. if its top include top overlap and vice versa
    items = getItems(img_scoreboard, item_coordinates, ITEM_SLOT_HEIGHT, ITEM_CONV_PAD)
    spells = getSpells(img_scoreboard, spell_coordinates, SPELL_SLOT_HEIGHT, SPELL_CONV_PAD)
    champs = getChamps(img_scoreboard, champ_coordinates, CHAMP_SLOT_HEIGHT, CHAMP_CONV_PAD)

    return items, spells, champs



def getItems(img, item_coordinates, item_slot_height, item_conv_pad):
    items = np.zeros((2, 5, 6, item_slot_height + 2*item_conv_pad, item_slot_height + 2*item_conv_pad, 3), dtype=np.uint8)
    for team_index in range(2):
        for champ_index in range(5):
            for item_index in range(6):
                items[team_index][champ_index][item_index] = addPaddingToBoundingBox(img, item_coordinates[team_index][
                    champ_index][item_index],
                                                                                     item_slot_height, item_conv_pad)
    return items


def getSpells(img, spell_coordinates, spell_slot_height, spell_conv_pad):
    spells = np.zeros((2, 5, 2, spell_slot_height + 2*spell_conv_pad, spell_slot_height + 2*spell_conv_pad, 3),
                      dtype=np.uint8)
    for team_index in range(2):
        for champ_index in range(5):
            spells[team_index][champ_index][0] = addPaddingToBoundingBox(img,
                                                                         spell_coordinates[team_index][champ_index][0],
                                                                         spell_slot_height, spell_conv_pad)
            spells[team_index][champ_index][1] = addPaddingToBoundingBox(img,
                                                                         spell_coordinates[team_index][champ_index][1],
                                                                         spell_slot_height, spell_conv_pad)
    return spells


def getChamps(img, champ_coordinates, champ_slot_height, champ_conv_pad):
    champs = np.zeros((2, 5, champ_slot_height + 2*champ_conv_pad, champ_slot_height + 2*champ_conv_pad, 3), dtype=np.uint8)
    for team_index in range(2):
        for champ_index in range(5):
            champs[team_index][champ_index] = addPaddingToBoundingBox(img, champ_coordinates[team_index][champ_index],
                                                                      champ_slot_height, champ_conv_pad)
    return champs
#
# champs = np.empty((2, 5), dtype="S50")
# items = np.empty((2, 5, 6), dtype="S50")
# spells = np.empty((2, 5, 2), dtype="S50")
#
# for team_index in range(2):
#     for champ_index in range(5):
#         sub_area = addPaddingToBoundingBox(img_warped, champ_coordinates[team_index][champ_index], CHAMP_SLOT_HEIGHT,
#                                            CHAMP_CONV_PAD)
#         champs[team_index][champ_index] = bestMatch(getChampTemplateDict(), CHAMP_SLOT_HEIGHT, sub_area)
#
#         sub_area_spell1 = addPaddingToBoundingBox(img_warped, spell_coordinates[team_index][champ_index][0],
#                                                   SPELL_SLOT_HEIGHT, SPELL_CONV_PAD)
#         sub_area_spell2 = addPaddingToBoundingBox(img_warped, spell_coordinates[team_index][champ_index][1],
#                                                   SPELL_SLOT_HEIGHT, SPELL_CONV_PAD)
#
#         spells[team_index][champ_index][0] = bestMatch(getSpellTemplateDict(), SPELL_SLOT_HEIGHT, sub_area_spell1)
#         spells[team_index][champ_index][1] = bestMatch(getSpellTemplateDict(), SPELL_SLOT_HEIGHT, sub_area_spell2)
#
#         for item_index in range(6):
#             sub_area = addPaddingToBoundingBox(img_warped, item_coordinates[team_index][champ_index][item_index],
#                                                ITEM_SLOT_HEIGHT, ITEM_CONV_PAD)
#             items[team_index][champ_index][item_index] = bestMatch(getItemTemplateDict(), ITEM_SLOT_HEIGHT, sub_area)
#
# cv.namedWindow("results", cv.WINDOW_NORMAL)
# cv.imshow('results', img_warped)
# cv.waitKey(0)
