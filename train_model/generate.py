import glob
import random

import cv2 as cv
import numpy as np

from utils import utils

#
#
# tmp = np.ones((1740,1740), dtype=np.uint8)*255
# img_size = tmp.shape[0]
# tmp[644:1095, 366:1373] = np.zeros((451,1007), dtype=np.uint8)
#
# src = np.int64([[[644, 366], [1095, 366], [644, 1373], [1095, 1373]]])
#
# M = cv.getRotationMatrix2D((img_size//2,img_size//2), 45, 1)
#
#
#
# tmp2 = cv.warpAffine(tmp, M, (img_size, img_size))
#
# # perspectiveTransform = cv.getPerspectiveTransform(src, dst)
# # tmp2 = cv.warpPerspective(tmp, perspectiveTransform, dsize=(1740, 1740))
# cv.namedWindow("ff", cv.WINDOW_NORMAL)
# cv.imshow('ff', tmp)
# cv.namedWindow("f", cv.WINDOW_NORMAL)
# cv.imshow('f', tmp2)
# cv.waitKey(0)


base_img_dims = (1740, 1740)

base_img = cv.imread('res/sample3.png')


def skewColor(img, rate):
    r_channel_skew = np.random.randint(-int(rate * 256), int(rate * 256))
    g_channel_skew = np.random.randint(-int(rate * 256), int(rate * 256))
    b_channel_skew = np.random.randint(-int(rate * 256), int(rate * 256))

    result = np.array(img.copy(), dtype=np.int64)

    result[:, :, 0] += r_channel_skew
    result[:, :, 1] += g_channel_skew
    result[:, :, 2] += b_channel_skew

    return np.uint8(np.clip(result, 0, 255, out=result))


def move(img, x_dst, y_dst):
    img_size = img.shape[0]
    x_dst = np.random.randint(-x_dst, x_dst)
    y_dst = np.random.randint(-y_dst, y_dst)
    M = np.float32([[1, 0, x_dst], [0, 1, y_dst]])
    return cv.warpAffine(img, M, (img_size, img_size))


def blur(img, rate):
    rate = np.random.randint(rate)
    kernel = np.random.choice([3, 5, 7])
    for i in range(rate):
        img = cv.medianBlur(img, kernel)
    return img


def noisy(img, noise_typ, rate=1):
    rate *= np.random.random()
    if noise_typ == "gauss":
        noisy = img + gauss
        return np.uint8(np.clip(noisy, 0, 255, out=noisy))
    elif noise_typ == "s&p":
        img_height = img.shape[0]
        img_width = img.shape[1]
        result = img.copy()

        coords = [np.random.randint(0, img_height - 1, int(rate * img_width ** 2)),
                  np.random.randint(0, img_width - 1, int(rate * img_width ** 2))]

        if len(img.shape) == 3:
            result[coords] = (np.random.randint(0, 80), np.random.randint(0, 80), np.random.randint(0, 80))
        else:
            result[coords] = np.random.randint(0, 80)
        coords = [np.random.randint(0, img_height - 1, int(rate * img_width ** 2)),
                  np.random.randint(0, img_width - 1, int(rate * img_width ** 2))]
        if len(img.shape) == 3:
            result[coords] = (np.random.randint(180, 255), np.random.randint(180, 255), np.random.randint(180, 255))
        else:
            result[coords] = np.random.randint(180, 255)
        return result


def thin(img, rate):
    kernel = np.ones((3, 3), np.uint8)
    rate = np.random.randint(rate)
    return cv.erode(img, kernel, iterations=rate)


def thick(img, rate):
    kernel = np.ones((3, 3), np.uint8)
    rate = np.random.randint(rate)
    return cv.dilate(img, kernel, iterations=rate)


def pixelate(img, rate):
    rate *= np.random.random()
    img_height = img.shape[0]
    img_width = img.shape[1]
    new_size = int(img_width * (1 - rate)), int(img_height * (1 - rate))
    small_img = cv.resize(img, new_size, interpolation=cv.INTER_NEAREST)
    return cv.resize(small_img, (img_width, img_height), interpolation=cv.INTER_NEAREST)


def grayOut(img, rate):
    if np.random.random() < rate:
        return np.stack((cv.cvtColor(img, cv.COLOR_BGR2GRAY),) * 3, -1)
    else:
        return img


def mirror(img):
    if np.random.random() > 0.5:
        return cv.flip(img, 1)
    else:
        return img


def changeBrightness(img, value):
    value = np.random.randint(-value, value, dtype=np.int8)

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    if value > 0:
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
    else:
        lim = -value
        v[v < lim] = 0
        v[v >= lim] = np.array(v[v >= lim] + value).astype(np.uint8)

    final_hsv = cv.merge((h, s, v))
    img = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
    return img


def rasterize(img, rate):
    rate *= np.random.random()
    step = int(1 / rate)
    result = img.copy()
    mask = np.zeros(img.shape)
    result[::step] = mask[::step]
    result[:, ::step] = mask[:, ::step]
    return result


def addLines(img, rate):
    img_height = img.shape[0]
    img_width = img.shape[1]
    rate *= np.random.random()
    result = img.copy()
    for i in range(int(rate)):
        cv.line(result, (np.random.randint(0, img_height), np.random.randint(0, img_width)),
                (np.random.randint(0, img_height), np.random.randint(0, img_width)),
                (255, 255, 255), 1)
    return result


def placeSurroundingImages(img, side_images, vertical):
    size = img.shape[0]
    if vertical:
        overlap = np.random.randint(10)
        if random.random() > 0.5:
            top_side_img = random.choice(side_images)[-overlap:]
            result = np.concatenate([top_side_img, img], axis=0)
            result = np.pad(result, ((0, overlap), (overlap, overlap), (0, 0)), 'constant',
                            constant_values=(0, 0))
        else:
            bot_side_img = random.choice(side_images)[:overlap]
            result = np.concatenate([img, bot_side_img], axis=0)
            result = np.pad(result, ((overlap, 0), (overlap, overlap), (0, 0)), 'constant',
                            constant_values=(0, 0))
        result = cv.resize(result, (size, size), interpolation=cv.INTER_NEAREST)
        return result
    else:
        left_overlap = np.random.randint(10)  # int(np.random.random() * size / 3) + 1
        right_overlap = np.random.randint(10)
        extra_width = left_overlap + right_overlap
        vert_offset = int(np.random.random() * extra_width)
        right_side_img = random.choice(side_images)[:, :right_overlap]
        left_side_img = random.choice(side_images)[:, -left_overlap:]
        result = np.concatenate([left_side_img, img, right_side_img], axis=1)
        result = np.pad(result, ((vert_offset, extra_width - vert_offset), (0, 0), (0, 0)), 'constant',
                        constant_values=(0, 0))
        result = cv.resize(result, (size, size), interpolation=cv.INTER_NEAREST)
        return result


def overlayCircle(img, angle):
    if np.random.random() < 0.9:
        return img
    angle = np.random.randint(0, angle)
    img_height = img.shape[0]
    img_width = img.shape[1]
    circle_mask = np.zeros((img_height, img_width), np.uint8)
    circle_segment = cv.ellipse2Poly((img_height // 2, img_width // 2), (img_height, img_width), 270, 0, angle, 1)
    circle_segment = np.append(circle_segment, [(img_height // 2, img_width // 2)], axis=0)
    circle_mask = cv.fillPoly(circle_mask, [circle_segment], 255)

    img_circle_slice = cv.bitwise_and(img, img, mask=circle_mask)
    img_gray = np.stack((cv.cvtColor(img, cv.COLOR_BGR2GRAY),) * 3, -1)
    img_gray = cv.fillPoly(img_gray, [circle_segment], 0)
    result = cv.bitwise_xor(img_gray, img_circle_slice)
    result = cv.polylines(result, [circle_segment], True, (255, 255, 255), 1)

    return result


def overlayText(img):
    if np.random.random() < 0.5:
        return img
    img_height = img.shape[0]
    img_width = img.shape[1]
    for i in range(2):
        text = chr(np.random.randint(33, 126))
        org = np.random.randint(0, int(img_width)), np.random.randint(0, int(img_height))
        color = np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)
        size = np.random.normal(1, 1)
        thickness = np.random.randint(2, 4)
        img = cv.putText(img, text, org, cv.FONT_HERSHEY_PLAIN, size, color, thickness, bottomLeftOrigin=False)
    return img


def occlude(img, num_circles, circle_size, darkness_rate):
    circle_size *= np.random.random()
    num_circles = np.random.randint(0, num_circles)
    result = img.copy().astype(np.int16)

    img_height = img.shape[0]
    img_width = img.shape[1]

    for _ in range(num_circles):
        center = np.random.randint(0, img_height), np.random.randint(0, img_width)
        radius = int(circle_size * img_width)
        circle_mask = np.zeros((img.shape), dtype=np.int16)
        darkness = np.random.randint(int(-darkness_rate * 255), int(darkness_rate * 255))
        for i in range(radius):
            circle_mask = cv.circle(circle_mask, center, i, (
                darkness * (radius - i) / radius, darkness * (radius - i) / radius, darkness * (radius - i) / radius),
                                    2)
        # circle_mask = np.pad(circle_mask, ((max(0, center[1]-radius), max(0, img_size-center[1]-radius)),(max(0, center[0]-radius), max(0, img_size-center[0]-radius)),(0,0)), 'constant', constant_values=(0, 0))
        result -= circle_mask
        result[result < 0] = 0
        result[result > 255] = 255

    return result.astype(np.uint8)


def translucentOverlay(img):
    pass


def changeContrast(img, rate):
    img = np.int16(img)
    rate = np.random.randint(0, rate)
    rate -= rate
    img = img * (rate / 127 + 1) - rate
    img = np.clip(img, 0, 255, out=img)

    return np.uint8(img)


def resize(img, new_width, new_height):
    return cv.resize(img, (new_width, new_height), interpolation=cv.INTER_AREA)


def translate(img, rate):
    rows, cols = img.shape[:2]
    rate = img.shape[0] * rate
    M = np.float32([[1, 0, np.random.randint(-rate, rate)], [0, 1, np.random.randint(-rate, rate)]])
    result = cv.warpAffine(img, M, (cols, rows))
    return result


def embeddedTranslate(coord, rate):
    if int(rate) == 0: return coord
    return max(0, coord[0] + np.random.randint(-rate, rate)), max(0, coord[1] + np.random.randint(-rate, rate))


def embeddedRandResize(size, rate):
    rate = np.random.uniform(-rate, rate)
    size *= (1 + rate)
    return int(size)


def randResize(img, rate):
    orig_shape = img.shape
    rate = np.random.uniform(-rate, rate)
    new_width, new_height = int(img.shape[1] * (1 + rate)), int(img.shape[0] * (1 + rate))
    result = cv.resize(img, (new_height, new_width), interpolation=cv.INTER_AREA)
    # zoom in
    if rate > 0:
        return result[(new_height - orig_shape[0]) // 2:(new_height - orig_shape[0]) // 2 + orig_shape[0],
               (new_width - orig_shape[1]) // 2:(new_width - orig_shape[1]) // 2 + orig_shape[1]]
    # zoom out
    else:
        h_border = (orig_shape[1] - new_width) // 2
        v_border = (orig_shape[0] - new_height) // 2
        if new_width % 2 == 1:
            result = cv.copyMakeBorder(result, v_border, v_border + 1, h_border + 1, h_border, cv.BORDER_CONSTANT,
                                       value=(0, 0, 0))
        else:
            result = cv.copyMakeBorder(result, v_border, v_border, h_border, h_border, cv.BORDER_CONSTANT,
                                       value=(0, 0, 0))
        return result


def mess_up_current_gold(img, config):
    color_change_rate = config["color_change_rate"]

    blur_rate = config["blur_rate"]

    pixelate_rate = config["pixelate_rate"]
    brightness_rate = config["brightness_rate"]

    contrast_rate = config["contrast_rate"]

    gaussian_rate = config["gaussian_rate"]
    gray_rate = config["gray_rate"]

    img = skewColor(img, color_change_rate)
    img = changeBrightness(img, brightness_rate)
    img = changeContrast(img, contrast_rate)

    img = grayOut(img, gray_rate)

    img = pixelate(img, pixelate_rate)
    img = blur(img, blur_rate)

    img = noisy(img, "gauss", gaussian_rate)

    return img


def messUpChamp(img, config):
    angle = config["angle"]
    color_change_rate = config["color_change_rate"]

    blur_rate = config["blur_rate"]

    pixelate_rate = config["pixelate_rate"]
    brightness_rate = config["brightness_rate"]
    line_rate = config["line_rate"]
    num_circles = config["num_circles"]
    contrast_rate = config["contrast_rate"]
    circle_size = config["circle_size"]
    darkness_rate = config["darkness_rate"]
    translate_rate = config["translate_rate"]
    resize_rate = config["resize_rate"]
    gaussian_rate = config["gaussian_rate"]
    gray_rate = config["gray_rate"]

    img = addLines(img, line_rate)
    img = overlayText(img)
    img = overlayCircle(img, angle)
    img = occlude(img, num_circles, circle_size, darkness_rate)

    img = skewColor(img, color_change_rate)
    img = changeBrightness(img, brightness_rate)
    img = changeContrast(img, contrast_rate)

    img = grayOut(img, gray_rate)

    img = noisy(img, "gauss", gaussian_rate)

    return img


def messUpItem(img, config):
    angle = config["angle"]
    color_change_rate = config["color_change_rate"]

    blur_rate = config["blur_rate"]

    pixelate_rate = config["pixelate_rate"]
    brightness_rate = config["brightness_rate"]
    line_rate = config["line_rate"]
    num_circles = config["num_circles"]
    contrast_rate = config["contrast_rate"]
    circle_size = config["circle_size"]
    darkness_rate = config["darkness_rate"]
    translate_rate = config["translate_rate"]
    resize_rate = config["resize_rate"]
    gaussian_rate = config["gaussian_rate"]
    gray_rate = config["gray_rate"]

    img = addLines(img, line_rate)
    img = overlayText(img)
    img = overlayCircle(img, angle)
    img = occlude(img, num_circles, circle_size, darkness_rate)

    img = skewColor(img, color_change_rate)
    img = changeBrightness(img, brightness_rate)
    img = changeContrast(img, contrast_rate)

    img = grayOut(img, gray_rate)

    img = pixelate(img, pixelate_rate)
    img = blur(img, blur_rate)

    img = noisy(img, "gauss", gaussian_rate)

    return img


def messUpImage(img, config):
    color_change_rate = config["color_change_rate"]

    blur_rate = config["blur_rate"]
    gaussian_rate = config["gaussian_rate"]
    s_p_rate = config["s_p_rate"]

    pixelate_rate = config["pixelate_rate"]
    brightness_rate = config["brightness_rate"]
    line_rate = config["line_rate"]
    rasterization_rate = config["rasterization_rate"]
    num_circles = config["num_circles"]
    contrast_rate = config["contrast_rate"]
    circle_size = config["circle_size"]
    darkness_rate = config["darkness_rate"]
    gray_rate = config["gray_rate"]

    # add some random lines
    img = addLines(img, line_rate)
    img = occlude(img, num_circles, circle_size, darkness_rate)

    # mess with colors & brightness
    img = skewColor(img, color_change_rate)
    img = changeBrightness(img, brightness_rate)
    img = grayOut(img, gray_rate)
    img = changeContrast(img, contrast_rate)

    # mess with contours
    # img = thin(img,thin_rate)
    # img = thick(img, thick_rate)

    # mess with resolution
    # img = pixelate(img, pixelate_rate)
    img = blur(img, blur_rate)

    img = rasterize(img, rasterization_rate)

    # mess with signal
    img = noisy(img, "s&p", s_p_rate)
    img = noisy(img, "gauss", gaussian_rate)
    return img


def rotate(img, angle, top_left_x, top_left_y, top_right_x, top_right_y, bottom_right_x, bottom_right_y, bottom_left_x,
           bottom_left_y):
    img_size = img.shape[0]
    angle %= 181

    # find rotation where all points are in bounds
    # super ugly i know
    while True:
        try:
            newangle = np.random.randint(-angle, angle)
        except:
            pass
        if angle > 0:
            M = cv.getRotationMatrix2D((img_size / 2, img_size / 2), newangle, 1)
        else:
            M = cv.getRotationMatrix2D((img_size / 2, img_size / 2), 360 + newangle, 1)

        rotated_corners = np.int64([M.dot(np.array((point[0], point[1], 1))) for point in
                                    [[top_left_x, top_left_y], [top_right_x, top_right_y],
                                     [bottom_right_x, bottom_right_y], [bottom_left_x, bottom_left_y]]])
        if not np.any((rotated_corners > img_size) + (rotated_corners < 0)):
            break
        else:
            pass

    result = cv.warpAffine(img, M, (img_size, img_size))
    return result, rotated_corners


def rotateSimple(img, angle):
    img_size = img.shape[0]
    angle %= 181
    angle = np.random.randint(-angle, angle)
    if angle > 0:
        M = cv.getRotationMatrix2D((img_size / 2, img_size / 2), angle, 1)
    else:
        M = cv.getRotationMatrix2D((img_size / 2, img_size / 2), 360 + angle, 1)

    result = cv.warpAffine(img, M, (img_size, img_size))
    return result


def distort(img, x0, y0, x1, y1, rate):
    img_height = img.shape[0]
    img_width = img.shape[1]
    rate *= np.random.random() * img_width
    min_dst_x = (x1 - x0) // 2
    min_dst_y = (y1 - y0) // 2

    top_left_x = random.uniform(max(0, x0 - rate), min(x0 + rate, img_width))
    top_left_y = random.uniform(max(0, y0 - rate), min(y0 + rate, img_height))
    top_right_x = random.uniform(max(min(img_width, top_left_x + min_dst_x), x1 - rate), min(x1 + rate, img_width))
    top_right_y = random.uniform(max(0, y0 - rate), min(y0 + rate, img_height))
    bottom_left_x = random.uniform(max(0, x0 - rate), min(x0 + rate, img_width))
    bottom_left_y = random.uniform(max(min(img_height, top_left_y + min_dst_y), y1 - rate), min(y1 + rate, img_height))
    bottom_right_x = random.uniform(max(min(img_width, bottom_left_x + min_dst_x), x1 - rate),
                                    min(x1 + rate, img_width))
    bottom_right_y = random.uniform(max(min(top_right_y + min_dst_y, img_height), y1 - rate),
                                    min(y1 + rate, img_height))

    src = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    dst = np.float32([
        (top_left_x, top_left_y),
        (top_right_x, top_right_y),
        (bottom_right_x, bottom_right_y),
        (bottom_left_x, bottom_left_y)
    ])
    perspectiveTransform = cv.getPerspectiveTransform(src, dst)
    result = cv.warpPerspective(img, perspectiveTransform, dsize=(img_height, img_width))

    return result, (top_left_x, top_left_y), (top_right_x, top_right_y), (bottom_right_x, bottom_right_y), (
        bottom_left_x, bottom_left_y)


def transform(img, config):
    rotation = config["rotation"]
    distort_rate = config["distort_rate"]
    x0, y0, x1, y1 = config["distort_params"]
    # affine transformations

    img, (top_left_x, top_left_y), (top_right_x, top_right_y), (bottom_right_x, bottom_right_y), (
        bottom_left_x, bottom_left_y) = distort(img, x0, y0, x1, y1, distort_rate)

    img, corners = rotate(img, rotation, top_left_x, top_left_y, top_right_x, top_right_y, bottom_right_x,
                          bottom_right_y, bottom_left_x, bottom_left_y)

    # img = move(img, x_dst, y_dst)
    # img = mirror(img)
    return img, corners


def overlayRandomImgs(img, overlays, iterations=1):
    img_height = img.shape[0]
    img_width = img.shape[1]

    for _ in range(iterations):
        coords = [np.random.randint(0, img_height - 1, len(overlays)),
                  np.random.randint(0, img_width - 1, len(overlays))]
        for overlay, (y, x) in zip(overlays, np.transpose(coords)):
            overlay_width = min(overlay.shape[1], img_width - x)
            overlay_height = min(overlay.shape[0], img_height - y)
            overlay = overlay[0:overlay_height, 0:overlay_width]
            roi = img[y:y + overlay_height, x:x + overlay_width, :3]
            mask = cv.bitwise_not(overlay[:, :, 3])
            roi = cv.bitwise_and(roi, roi, mask=mask)
            mask = overlay[:, :, 3]
            overlay_fg = cv.bitwise_and(overlay, overlay, mask=mask)[:, :, :3]
            roi = cv.bitwise_xor(roi, overlay_fg)
            img[y:y + overlay_height, x:x + overlay_width, :3] = roi

    return img


def drawBlackBorder(img, x0, y0, x1, y1, thickness=1):
    for i in range(thickness):
        cv.rectangle(img, (x0 - i, y0 - i), (x1 + i, y1 + i), (0, 0, 0, 255), 1)


def getScoreboardElementXCoords(scoreboard_new_tile_width, new_champ_size, new_item_size, new_spell_size):
    items_width = 7 * new_item_size + utils.SCOREBOARD_ITEM_BORDER_WIDTH
    champ_width = new_champ_size + utils.SCOREBOARD_ITEM_BORDER_WIDTH
    spells_width = new_spell_size + utils.SCOREBOARD_ITEM_BORDER_WIDTH

    item_x_offset = np.random.randint(utils.SCOREBOARD_ITEM_BORDER_WIDTH, scoreboard_new_tile_width - items_width)
    if item_x_offset - utils.SCOREBOARD_ITEM_BORDER_WIDTH > utils.SCOREBOARD_ITEM_BORDER_WIDTH + champ_width:
        if scoreboard_new_tile_width - item_x_offset - items_width > utils.SCOREBOARD_ITEM_BORDER_WIDTH + champ_width:
            if np.random.random() > 0.5:
                champ_x_offset = np.random.randint(utils.SCOREBOARD_ITEM_BORDER_WIDTH,
                                                   item_x_offset - utils.SCOREBOARD_ITEM_BORDER_WIDTH - champ_width)
            else:
                try:
                    champ_x_offset = np.random.randint(
                        item_x_offset + items_width + utils.SCOREBOARD_ITEM_BORDER_WIDTH,
                        scoreboard_new_tile_width - champ_width)
                except:
                    pass
        else:
            champ_x_offset = np.random.randint(utils.SCOREBOARD_ITEM_BORDER_WIDTH,
                                               item_x_offset - utils.SCOREBOARD_ITEM_BORDER_WIDTH - champ_width)
    else:
        try:
            champ_x_offset = np.random.randint(
                item_x_offset + items_width + utils.SCOREBOARD_ITEM_BORDER_WIDTH,
                scoreboard_new_tile_width - champ_width)

        except:
            pass

    position_wide_enough = [False, False, False]

    left_side_offset = min(champ_x_offset, item_x_offset)
    right_side_offset = max(champ_x_offset, item_x_offset)
    left_side_width = items_width
    right_side_width = champ_width
    if left_side_offset != item_x_offset:
        left_side_width, right_side_width = right_side_width, left_side_width

    # left side wide enough?
    if left_side_offset - utils.SCOREBOARD_ITEM_BORDER_WIDTH > utils.SCOREBOARD_ITEM_BORDER_WIDTH + spells_width:
        position_wide_enough[0] = True
    # middle wide enough?
    if right_side_offset - left_side_offset - left_side_width > utils.SCOREBOARD_ITEM_BORDER_WIDTH + spells_width:
        position_wide_enough[1] = True
    # right side wide enough?
    if scoreboard_new_tile_width - right_side_offset - right_side_width > utils.SCOREBOARD_ITEM_BORDER_WIDTH + spells_width:
        position_wide_enough[2] = True

    assert position_wide_enough != [False, False, False]
    while True:
        choice = np.random.randint(0, len(position_wide_enough))
        if position_wide_enough[choice]:
            if choice == 0:
                try:
                    spells_x_offset = np.random.randint(utils.SCOREBOARD_ITEM_BORDER_WIDTH,
                                                        left_side_offset - utils.SCOREBOARD_ITEM_BORDER_WIDTH - spells_width)
                except:
                    pass
            elif choice == 1:
                try:
                    spells_x_offset = np.random.randint(left_side_offset + left_side_width,
                                                        right_side_offset - utils.SCOREBOARD_ITEM_BORDER_WIDTH - spells_width)
                except:
                    pass
            elif choice == 2:
                try:
                    spells_x_offset = np.random.randint(right_side_offset + right_side_width,
                                                        scoreboard_new_tile_width - utils.SCOREBOARD_ITEM_BORDER_WIDTH - spells_width)
                except:
                    pass
            break

    return item_x_offset, champ_x_offset, spells_x_offset


paths = glob.glob('res/random_background_elements/inside_scoreboard/*')
overlays_inner = [cv.imread(path, cv.IMREAD_UNCHANGED) for path in paths]
overlays_inner = [rotateSimple(overlay, 180) for overlay in overlays_inner]
paths = glob.glob('res/random_background_elements/outside_scoreboard/*')
overlays_outer = [cv.imread(path, cv.IMREAD_UNCHANGED) for path in paths]
overlays_outer = [rotateSimple(overlay, 180) for overlay in overlays_outer]
scoreboard_raw = cv.imread('res/scoreboard_only.png', cv.IMREAD_UNCHANGED)
scoreboard_top_tile_raw = cv.imread('res/top_scoreboard_tile.png', cv.IMREAD_UNCHANGED)
# scoreboard_raw = resize(scoreboard_raw, *np.int32(np.array(scoreboard_raw.shape[:2])*utils.SCOREBOARD_SCALING))
# scoreboard_top_tile_raw = resize(scoreboard_top_tile_raw, *np.int32(np.array(scoreboard_top_tile_raw.shape[:2])*utils.SCOREBOARD_SCALING*0.99))

background_config = \
    {
        "color_change_rate": 0.2,
        "blur_rate": 1,
        "gaussian_rate": 25,
        "s_p_rate": 0.03,
        "pixelate_rate": 0.3,
        "brightness_rate": 20,
        "num_circles": 5,
        "line_rate": 5,
        "rasterization_rate": 0.25,
        "contrast_rate": 50,
        "circle_size": 1.0,
        "darkness_rate": 0.3,
        "distort_rate": 0.2,
        "rotation": 180
    }

item_config = \
    {
        "angle": 360,
        "num_circles": 3,
        "circle_size": 1.0,
        "darkness_rate": 0.4,
        "color_change_rate": 0.25,
        "blur_rate": 6,
        "pixelate_rate": 0.4,
        "brightness_rate": 20,
        "line_rate": 2,
        "contrast_rate": 20,
        "translate_rate": 0.3,
        "resize_rate": 0.2,
        "gaussian_rate": 10,
        "gray_rate": 0.3
    }

champ_config = \
    {
        "angle": 360,
        "num_circles": 3,
        "circle_size": 1.0,
        "darkness_rate": 0.4,
        "color_change_rate": 0.2,
        "blur_rate": 6,
        "pixelate_rate": 0.4,
        "brightness_rate": 20,
        "line_rate": 2,
        "contrast_rate": 20,
        "translate_rate": 0.3,
        "resize_rate": 0.2,
        "gaussian_rate": 10,
        "gray_rate": 0.3
    }

current_gold_config = \
    {
        "angle": 360,
        "num_circles": 0,
        "circle_size": 1.0,
        "darkness_rate": 0.4,
        "color_change_rate": 0.05,
        "blur_rate": 6,
        "pixelate_rate": 0.4,
        "brightness_rate": 20,
        "line_rate": 0,
        "contrast_rate": 20,
        "translate_rate": 0.3,
        "resize_rate": 0.2,
        "gaussian_rate": 10,
        "gray_rate": 0.3
    }

kda_config = \
    {
        "darkness_rate": 0.4,
        "color_change_rate": 0.05,
        "blur_rate": 1,
        "pixelate_rate": 0.2,
        "brightness_rate": 20,
        "contrast_rate": 20,
        "translate_rate": 0.3,
        "gaussian_rate": 10,
        "gray_rate": 0.3
    }


#
# spell_imgs_global = utils.getSpellTemplateDict()
# champ_imgs_global = utils.getChampTemplateDict()
# item_imgs_global = utils.getItemTemplateDict()
#


# scoreboard_top_tile_coords = np.nonzero(scoreboard_top_tile_raw[:, :, 2] >= 250)
# scoreboard_top_tile_shifted_coords = tuple(
#     np.transpose(np.transpose(scoreboard_top_tile_coords) + (utils.SCOREBOARD_TOP_TILE_Y_OFFSET, utils.SCOREBOARD_INNER_LEFT_X_OFFSET)))
# scoreboard_top_tile_raw[:] = 0


def generateRandomScoreboard():
    # base_img = np.random.randint(0,50, (base_img_dims[0], base_img_dims[1], 3), dtype=np.uint8)
    base_img = np.zeros((base_img_dims[0], base_img_dims[1], 4), dtype=np.uint8)
    # tmp = np.array([base_img[:,:,0], base_img[:,:,1], base_img[:,:,2], np.ones((base_img_dims[0], base_img_dims[1]), dtype=np.uint8)])
    # base_img = cv.merge(tmp)

    scoreboard_only = scoreboard_raw.copy()
    # scoreboard_alpha = scoreboard_only[:, :, 3]

    scoreboard_top_tile = scoreboard_top_tile_raw.copy()
    scoreboard_top_tile = overlayRandomImgs(scoreboard_top_tile, overlays_inner, 2)

    # scoreboard_only[shifted_coords] = (0,0,0,255)
    scoreboard_alpha = scoreboard_only[:, :, 3].copy()
    scoreboard_only[scoreboard_top_tile_shifted_coords] = scoreboard_top_tile[scoreboard_top_tile_coords]
    scoreboard_only[:, :, 3] = scoreboard_alpha

    champ_imgs = champ_imgs_global.copy()
    item_imgs = item_imgs_global.copy()
    spell_imgs = spell_imgs_global.copy()
    new_champ_size = np.random.randint(utils.CHAMP_MIN_SIZE, utils.CHAMP_MAX_SIZE)
    new_item_size = np.random.randint(utils.ITEM_MIN_SIZE, utils.ITEM_MAX_SIZE)
    new_spell_size = np.random.randint(utils.SPELL_MIN_SIZE, utils.SPELL_MAX_SIZE)
    champ_imgs = {k: resize(img, new_champ_size, new_champ_size) for k, img in champ_imgs.items()}
    item_imgs = {k: resize(img, new_item_size, new_item_size) for k, img in item_imgs.items()}
    spell_imgs = {k: resize(img, new_spell_size, new_spell_size) for k, img in spell_imgs.items()}

    for i in range(5):
        top_border = utils.SCOREBOARD_INNER_Y_BOT_OFFSET - i * (utils.SCOREBOARD_INNER_TILE_HEIGHT)
        bottom_border = top_border + utils.SCOREBOARD_INNER_TILE_HEIGHT - utils.SCOREBOARD_INNER_BORDER_WIDTH
        left_border = utils.SCOREBOARD_INNER_LEFT_X_OFFSET
        right_border = left_border + utils.SCOREBOARD_INNER_TILE_WIDTH
        # scoreboard_only[top_border: bottom_border, left_border:right_border] = np.random.randint(0,50,(bottom_border-top_border, right_border-left_border, 3))
        scoreboard_only[top_border: bottom_border, left_border:right_border] = overlayRandomImgs(
            scoreboard_only[top_border: bottom_border, left_border:right_border], overlays_inner, 2)
        left_border = utils.SCOREBOARD_INNER_RIGHT_X_OFFSET
        right_border = left_border + utils.SCOREBOARD_INNER_TILE_WIDTH + utils.SCOREBOARD_INNER_BORDER_WIDTH
        # scoreboard_only[top_border: bottom_border, left_border:right_border] = np.random.randint(0, 50, (
        # bottom_border - top_border, right_border - left_border, 3))
        scoreboard_only[top_border: bottom_border, left_border:right_border] = overlayRandomImgs(
            scoreboard_only[top_border: bottom_border, left_border:right_border],
            overlays_inner, 2)

    scoreboard_new_tile_height = int(np.random.uniform(
        max(2 * new_spell_size, new_champ_size, new_item_size) + 2 * utils.SCOREBOARD_ITEM_BORDER_WIDTH + 10,
        utils.SCOREBOARD_ITEM_BORDER_WIDTH * 2 + 2 * utils.SPELL_MAX_SIZE))
    scoreboard_new_tile_width = int(np.random.uniform(
        7 * new_item_size + 6 * utils.SCOREBOARD_ITEM_BORDER_WIDTH + 2 * new_champ_size + 3 * new_spell_size,
        6 * utils.SCOREBOARD_ITEM_BORDER_WIDTH + 7 * utils.ITEM_MAX_SIZE + 2 * utils.CHAMP_MAX_SIZE + 3 * utils.SPELL_MAX_SIZE))

    scoreboard_new_height = int(
        (scoreboard_new_tile_height / utils.SCOREBOARD_INNER_TILE_HEIGHT) * utils.STD_SCOREBOARD_HEIGHT)
    scoreboard_new_width = int(
        (scoreboard_new_tile_width / utils.SCOREBOARD_INNER_TILE_WIDTH) * utils.STD_SCOREBOARD_WIDTH)

    scoreboard_inner_border_width = int(
        (scoreboard_new_tile_height / utils.SCOREBOARD_INNER_TILE_HEIGHT) * utils.SCOREBOARD_INNER_BORDER_WIDTH)

    scoreboard_inner_bot_offset = utils.SCOREBOARD_INNER_Y_BOT_OFFSET * scoreboard_new_height / utils.STD_SCOREBOARD_HEIGHT
    scoreboard_inner_left_x_offset = int(
        (scoreboard_new_width / utils.STD_SCOREBOARD_WIDTH) * utils.SCOREBOARD_INNER_LEFT_X_OFFSET)
    scoreboard_inner_right_x_offset = int(
        (scoreboard_new_width / utils.STD_SCOREBOARD_WIDTH) * utils.SCOREBOARD_INNER_RIGHT_X_OFFSET)

    scoreboard_only = resize(scoreboard_only, scoreboard_new_width, scoreboard_new_height)
    # scoreboard_alpha = resize(scoreboard_alpha, scoreboard_new_width, scoreboard_new_height)

    item_left_x_offset, champ_left_x_offset, spells_left_x_offset = getScoreboardElementXCoords(
        scoreboard_new_tile_width, new_champ_size, new_item_size, new_spell_size)
    item_right_x_offset, champ_right_x_offset, spells_right_x_offset = getScoreboardElementXCoords(
        scoreboard_new_tile_width, new_champ_size, new_item_size, new_spell_size)

    top_y_offset = scoreboard_inner_bot_offset - 4 * (scoreboard_new_tile_height)

    spells_y_offset = np.random.randint(top_y_offset + utils.SCOREBOARD_ITEM_BORDER_WIDTH,
                                        top_y_offset + scoreboard_new_tile_height - 2 * new_spell_size - utils.SCOREBOARD_ITEM_BORDER_WIDTH)
    champs_y_offset = np.random.randint(top_y_offset + utils.SCOREBOARD_ITEM_BORDER_WIDTH,
                                        top_y_offset + scoreboard_new_tile_height - new_champ_size - utils.SCOREBOARD_ITEM_BORDER_WIDTH)
    items_y_offset = np.random.randint(top_y_offset + utils.SCOREBOARD_ITEM_BORDER_WIDTH,
                                       top_y_offset + scoreboard_new_tile_height - new_item_size - utils.SCOREBOARD_ITEM_BORDER_WIDTH)

    item_coords = utils._generate_item_coords(new_item_size, item_left_x_offset + scoreboard_inner_left_x_offset,
                                              item_right_x_offset + scoreboard_inner_right_x_offset,
                                              scoreboard_new_tile_height, items_y_offset)
    spell_coords = utils.generateSpellCoordinates(new_spell_size,
                                                  spells_left_x_offset + scoreboard_inner_left_x_offset,
                                                  spells_right_x_offset + scoreboard_inner_right_x_offset,
                                                  scoreboard_new_tile_height - new_spell_size, spells_y_offset)
    champ_coords = utils._generate_champ_coords(new_champ_size,
                                                champ_left_x_offset + scoreboard_inner_left_x_offset,
                                                champ_right_x_offset + scoreboard_inner_right_x_offset,
                                                scoreboard_new_tile_height, champs_y_offset)

    champ_names = []
    spell_names = []
    item_names = []
    for team_index in range(2):
        for champ_index in range(5):
            current_champ_coords = champ_coords[team_index][champ_index]
            current_champ_name = random.choice(list(champ_imgs.keys()))
            champ_names += [current_champ_name]
            current_champ_img = champ_imgs[current_champ_name].copy()
            coords = np.nonzero((current_champ_img == [0, 0, 0]).all(axis=2))
            current_champ_img = messUpItem(current_champ_img, item_config)
            # offset_coords = ([coord+current_champ_coords[1] for coord in coords[0]], [coord+current_champ_coords[0] for coord in coords[1]])
            current_champ_img[coords] = (0, 0, 0)
            scoreboard_only[current_champ_coords[1]:current_champ_coords[1] + new_champ_size,
            current_champ_coords[0]:current_champ_coords[0] + new_champ_size][:, :, :3] = current_champ_img
            drawBlackBorder(scoreboard_only, current_champ_coords[0], current_champ_coords[1],
                            current_champ_coords[0] + new_champ_size, current_champ_coords[1] + new_champ_size, 10)
            for spell_index in range(2):
                current_spell_coords = spell_coords[team_index][champ_index][spell_index]
                current_spell_name = random.choice(list(spell_imgs.keys()))
                spell_names += [current_spell_name]
                current_spell_img = messUpItem(spell_imgs[current_spell_name], item_config).copy()
                scoreboard_only[current_spell_coords[1]:current_spell_coords[1] + new_spell_size,
                current_spell_coords[0]: current_spell_coords[0] + new_spell_size][:, :, :3] = current_spell_img
            for item_index in range(7):
                current_item_coords = item_coords[team_index][champ_index][item_index]
                current_item_name = random.choice(list(item_imgs.keys()))
                item_names += [current_item_name]
                current_item_img = messUpItem(item_imgs[current_item_name], item_config).copy()
                scoreboard_only[current_item_coords[1]:current_item_coords[1] + new_item_size,
                current_item_coords[0]: current_item_coords[0] + new_item_size][:, :, :3] = current_item_img
            top_spell_coords = spell_coords[team_index][champ_index][0]
            bot_spell_coords = spell_coords[team_index][champ_index][1] + (new_spell_size, new_spell_size)
            left_item_coords = item_coords[team_index][champ_index][0]
            right_item_coords = item_coords[team_index][champ_index][-1] + (new_item_size, new_item_size)
            drawBlackBorder(scoreboard_only, top_spell_coords[0], top_spell_coords[1], bot_spell_coords[0],
                            bot_spell_coords[1], 10)
            drawBlackBorder(scoreboard_only, left_item_coords[0], left_item_coords[1], right_item_coords[0],
                            right_item_coords[1], 10)

    base_img_top = (base_img.shape[0] - scoreboard_new_height) // 2
    base_img_bot = base_img_top + scoreboard_new_height
    base_img_left = (base_img.shape[1] - scoreboard_new_width) // 2
    base_img_right = base_img_left + scoreboard_new_width

    base_img[base_img_top:base_img_bot, base_img_left:base_img_right] = scoreboard_only

    background_config['distort_params'] = (base_img_left, base_img_top, base_img_right, base_img_bot)

    base_img, corners = transform(base_img, background_config)

    scoreboard_only = cv.bitwise_and(base_img, base_img, mask=base_img[:, :, 3])
    base_img = overlayRandomImgs(base_img, overlays_outer, 2)
    base_img = cv.bitwise_and(base_img, base_img, mask=cv.bitwise_not(base_img[:, :, 3]))
    base_img = cv.bitwise_xor(base_img, scoreboard_only)

    base_img = messUpImage(base_img, background_config)

    return base_img, corners, item_coords, champ_coords, spell_coords, item_names, champ_names, spell_names


gauss = None


def changeContrast2(img):
    clahe = cv.createCLAHE(clipLimit=20.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def getBatch(size, img_size):
    global gauss
    gauss = np.random.normal(0, background_config["gaussian_rate"], (*base_img_dims, 3))
    training_imgs = []
    labels = []

    for _ in range(size):
        img, corners, item_coords, champ_coords, spell_coords, item_names, champ_names, spell_names = generateRandomScoreboard()
        list(map(lambda x: x.flatten(), [corners, item_coords, champ_coords, spell_coords]))
        img_orig_size = img.shape
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = changeContrast2(img)
        img = cv.resize(img, img_size, interpolation=cv.INTER_AREA)

        x_scale = img_size[0] / img_orig_size[1]
        y_scale = img_size[1] / img_orig_size[0]
        corners = np.array([(corner[0] * x_scale, corner[1] * y_scale) for corner in corners])

        item_coords = np.array([(x_coord * x_scale, y_coord * y_scale) for x_coord, y_coord in
                                zip(item_coords[0::2], item_coords[1::2])]).flatten()
        champ_coords = np.array([(x_coord * x_scale, y_coord * y_scale) for x_coord, y_coord in
                                 zip(champ_coords[0::2], champ_coords[1::2])]).flatten()
        spell_coords = np.array([(x_coord * x_scale, y_coord * y_scale) for x_coord, y_coord in
                                 zip(spell_coords[0::2], spell_coords[1::2])]).flatten()
        training_imgs += [img]
        labels += [[corners, item_coords, champ_coords, spell_coords, item_names, champ_names, spell_names]]

    return np.array(training_imgs), np.array(labels)


#
# cv.namedWindow("f", cv.WINDOW_NORMAL)
# cv.imshow('f', np.zeros((50,50), dtype=np.uint8))
# cv.waitKey(0)

# while True:
#     imgs = list(utils.getChampTemplateDict().values())
#     gauss = np.random.normal(0, 10, (112,112,3))
#     for img in imgs:
#         cv.namedWindow("f", cv.WINDOW_NORMAL)
#         cv.imshow('f', messUpItem(img, item_config))
#         cv.waitKey(0)
#

#
# while True:
#     generateRandomScoreboard()

# def profiling():
#         getBatch(100)
#
# pr = cProfile.Profile()
# pr.enable()
# profiling()
# pr.disable()
# s = io.StringIO()
# sortby = 'cumulative'
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# ps.print_stats()
# print(s.getvalue())

def generate_training_data_champs(imgs, epochs, new_size, extra_dim=False):
    max_crop_rate = 0.25
    images = []
    classes = []
    mykey = list(imgs.keys())[0]
    img_size = imgs[mykey].shape[0]
    global gauss
    gauss = np.random.normal(0, 10, (*new_size, 3))
    for _ in range(epochs):

        for key, image in imgs.items():
            # we're not using the whole image. instead, we're cropping it to a square shape of a random sub portion of the original image
            new_top_left_x = np.random.randint(0, int(img_size * max_crop_rate))
            new_top_left_y = np.random.randint(0, int(img_size * max_crop_rate))
            max_size = img_size - max(new_top_left_x, new_top_left_y)
            min_size = max_size - max_crop_rate * img_size
            crop_size = np.random.randint(min_size, max_size)

            gauss = np.random.normal(0, 10, (*new_size, 3))

            # cv.imshow('original', image)
            # cv.imshow('gray', grayimg)
            # cv.imshow('mod', mod)
            # cv.waitKey(0)
            image = pixelate(image, champ_config["pixelate_rate"])
            image = blur(image, champ_config["blur_rate"])
            image = image[new_top_left_x:new_top_left_x + crop_size, new_top_left_y:new_top_left_y + crop_size]

            image = cv.resize(image, new_size, interpolation=cv.INTER_AREA)

            image = messUpChamp(image, champ_config)

            images.append(image)
            classes.append(key)
    if extra_dim:
        classes = np.array(classes)[:, np.newaxis]

    return (images, classes)


def generate_training_data(imgs, epochs, new_size, extra_dim=False):
    max_crop_rate = 0.25
    images = []
    classes = []
    mykey = list(imgs.keys())[0]
    img_size = imgs[mykey].shape[0]
    global gauss
    gauss = np.random.normal(0, 10, (*new_size, 3))
    for _ in range(epochs):

        for key, image in imgs.items():
            # we're not using the whole image. instead, we're cropping it to a square shape of a random sub portion of the original image
            new_top_left_x = np.random.randint(0, int(img_size * max_crop_rate))
            new_top_left_y = np.random.randint(0, int(img_size * max_crop_rate))
            max_size = img_size - max(new_top_left_x, new_top_left_y)
            min_size = max_size - max_crop_rate * img_size
            crop_size = np.random.randint(min_size, max_size)

            gauss = np.random.normal(0, 10, (*new_size, 3))

            # cv.imshow('original', image)
            # cv.imshow('gray', grayimg)
            # cv.imshow('mod', mod)
            # cv.waitKey(0)

            image = image[new_top_left_x:new_top_left_x + crop_size, new_top_left_y:new_top_left_y + crop_size]

            image = cv.resize(image, new_size, interpolation=cv.INTER_AREA)

            image = messUpItem(image, item_config)

            images.append(image)
            classes.append(key)
    if extra_dim:
        classes = np.array(classes)[:, np.newaxis]

    return (images, classes)



def generate_training_data_nonsquare(imgs, epochs, new_size):
    images = []
    classes = []
    mykey = list(imgs.keys())[0]
    img_size_y, img_size_x = imgs[mykey].shape[:2]
    global gauss
    gauss = np.random.normal(0, 10, (*new_size, 3))
    for _ in range(epochs):

        for key, image in imgs.items():
            # we're not using the whole image. instead, we're cropping it to a square shape of a random sub portion of the original image
            a_min = min(img_size_x / 2, img_size_y / 2)
            a_max = min(img_size_x, img_size_y)

            crop_size = np.random.randint(a_min, a_max)

            top_left_x_min = 0
            top_left_x_max = img_size_x - crop_size
            top_left_y_min = 0
            top_left_y_max = img_size_y - crop_size

            top_left_x = np.random.randint(top_left_x_min, top_left_x_max)
            top_left_y = np.random.randint(top_left_y_min, top_left_y_max)

            gauss = np.random.normal(0, 10, (*new_size, 3))

            # cv.imshow('original', image)
            # cv.imshow('gray', grayimg)
            # cv.imshow('mod', mod)
            # cv.waitKey(0)

            image = image[top_left_y:top_left_y + crop_size, top_left_x:top_left_x + crop_size]

            image = cv.resize(image, new_size, interpolation=cv.INTER_AREA)

            image = mess_up_current_gold(image, current_gold_config)

            images.append(image)
            classes.append(key)

    return images, classes


def generate_training_data_rect(imgs, epochs, new_size):
    images = []
    classes = []
    x_pad = 3
    y_pad = 1
    imgs = {key: cv.copyMakeBorder(imgs[key], y_pad, y_pad, x_pad, x_pad, cv.BORDER_CONSTANT, value=(0, 0,
                                                                                                     0)) for key in
            imgs}
    mykey = list(imgs.keys())[0]
    img_size_y, img_size_x = imgs[mykey].shape[:2]
    global gauss
    gauss = np.random.normal(0, 10, (*new_size, 3))




    for _ in range(epochs):

        for key, image in imgs.items():
            # we're not using the whole image. instead, we're cropping it to a square shape of a random sub portion of the original image
            a_min = img_size_x / 2
            a_max = img_size_x
            b_min = img_size_y / 2
            b_max = img_size_y

            crop_size_a = np.random.randint(a_min, a_max + 1)
            crop_size_b = np.random.randint(b_min, b_max + 1)

            top_left_x_min = 0
            top_left_x_max = img_size_x - crop_size_a
            top_left_y_min = 0
            top_left_y_max = img_size_y - crop_size_b

            top_left_x = np.random.randint(top_left_x_min, top_left_x_max) if top_left_x_min < top_left_x_max else 0
            top_left_y = np.random.randint(top_left_y_min, top_left_y_max) if top_left_y_min < top_left_y_max else 0

            gauss = np.random.normal(0, 10, (*new_size, 3))

            # cv.imshow('original', image)
            # cv.imshow('gray', grayimg)
            # cv.imshow('mod', mod)
            # cv.waitKey(0)

            image = image[top_left_y:top_left_y + crop_size_b, top_left_x:top_left_x + crop_size_a]

            image = cv.resize(image, (new_size[1], new_size[0]), interpolation=cv.INTER_AREA)

            image = mess_up_current_gold(image, kda_config)

            images.append(image)
            classes.append(key)

    return images, classes


# while True:
#     self_imgs = utils.init_self_data_for_training()
#     self_imgs = dict(zip([1],[self_imgs[1]]))
#     imgs,keys = _load_data(self_imgs, 1)
#     for img in imgs:
#         pass

if __name__ == "__main__":
    import copy

    img_orig = cv.imread('../assets/train_imgs/items/0.png')
    cv.imshow("original", img_orig)
    while True:
        img = copy.deepcopy(img_orig)
        # img = generate_training_data_champs({"lol": img}, 1, (20, 20))[0][0]
        img = generate_training_data({"lol": img}, 1, (20, 20))[0][0]
        cv.imshow("text", img)
        cv.waitKey(0)
