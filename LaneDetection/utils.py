import os
import math
import cv2
import numpy as np
from scipy import stats
from collections import deque

def isolate_yellow_hsl(img):
    # Caution - OpenCV encodes the data in ****HLS*** format
    # Lower value equivalent pure HSL is (30, 45, 15)
    low_threshold = np.array([15, 80, 115], dtype=np.uint8) # [15, 38, 115]
    # Higher value equivalent pure HSL is (75, 100, 80)
    high_threshold = np.array([220, 255, 255], dtype=np.uint8) # [35, 204, 255]  
    return cv2.inRange(img, low_threshold, high_threshold)
                            
def isolate_white_hsl(img):
    # Caution - OpenCV encodes the data in ***HLS*** format
    # Lower value equivalent pure HSL is (30, 45, 15)
    low_threshold = np.array([0, 200, 0], dtype=np.uint8) # [0,200,0]
    # Higher value equivalent pure HSL is (360, 100, 100)
    high_threshold = np.array([255, 255, 255], dtype=np.uint8) # [180,255,255]
    return cv2.inRange(img, low_threshold, high_threshold)

def filter_img_hsl(img):
    hsl_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hsl_yellow = isolate_yellow_hsl(hsl_img)
    hsl_white = isolate_white_hsl(hsl_img)
    hsl_mask = cv2.bitwise_or(hsl_yellow, hsl_white)
    return cv2.bitwise_and(img, img, mask=hsl_mask)
    # return hsl_img

def get_vertices_for_img(img):
    height = img.shape[0]
    width = img.shape[1] 
    region_bottom_left = (0, height)
    region_top_left = (7*width//16, height//2)
    region_top_right = (9*width//16, height//2)
    region_bottom_right = (width, height)
    vert = np.array([[region_bottom_left , region_top_left, region_top_right, region_bottom_right]], dtype=np.int32)
    return vert

def region_of_interest(img):
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    vert = get_vertices_for_img(img)    
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vert, ignore_mask_color)
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=10, make_copy=True):
    # Copy the passed image
    img_copy = np.copy(img) if make_copy else img
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img_copy, (x1, y1), (x2, y2), color, thickness)
    return img_copy

def separate_lines(lines, img):
    img_shape = img.shape
    middle_x = img_shape[1] / 2
    left_lane_lines = []
    right_lane_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            dx = x2 - x1 
            if dx == 0:
                #Discarding line since we can't, gradient is undefined at this dx
                continue
            dy = y2 - y1
            # Similarly, if the y value remains constant as x increases, discard line
            if dy == 0:
                continue
            slope = dy / dx
            # Get rid of lines with a small slope as they are likely to be horizontal one
            epsilon = 0.3
            if abs(slope) <= epsilon:
                continue
            if slope < 0 and x1 < middle_x and x2 < middle_x:
                # Lane should be within the left hand side of region of interest
                left_lane_lines.append([[x1, y1, x2, y2]])
            elif x1 >= middle_x and x2 >= middle_x:
                # Lane should be within the right hand side of region of interest
                right_lane_lines.append([[x1, y1, x2, y2]])
    return left_lane_lines, right_lane_lines

def find_lane_lines_formula(lines):
    xs = []
    ys = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            xs.append(x1)
            xs.append(x2)
            ys.append(y1)
            ys.append(y2)
    slope, intercept, r_value, p_value, std_err = stats.linregress(xs, ys)
    return (slope, intercept)

def trace_lane_line(img, lines, top_y, make_copy=True):
    A, b = find_lane_lines_formula(lines)
    vert = get_vertices_for_img(img)
    img_shape = img.shape
    bottom_y = img_shape[0] - 1
    # y = Ax + b, therefore x = (y - b) / A
    x_to_bottom_y = (bottom_y - b) / A
    top_x_to_y = (top_y - b) / A 
    new_lines = [[[int(x_to_bottom_y), int(bottom_y), int(top_x_to_y), int(top_y)]]]
    return draw_lines(img, new_lines, make_copy=make_copy)

def trace_both_lane_lines(img, left_lane_lines, right_lane_lines):
    vert = get_vertices_for_img(img)
    region_top_left = vert[0][1]
    temp = np.zeros(img.shape,dtype=np.uint8)
    full_left_lane_img = trace_lane_line(temp, left_lane_lines, region_top_left[1], make_copy=True)
    full_left_right_lanes_img = trace_lane_line(full_left_lane_img, right_lane_lines, region_top_left[1], make_copy=False)
    full_left_right_lanes_img = cv2.cvtColor(full_left_right_lanes_img,cv2.COLOR_BGR2GRAY)
    full_left_right_lanes_img[full_left_right_lanes_img.shape[0]-1,:] = full_left_right_lanes_img.max()
    contours, hierarchy = cv2.findContours(full_left_right_lanes_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted(contours,key=cv2.contourArea, reverse=True)
    contour = contours[0]
    temp2 = cv2.fillPoly(temp,[contour],(0,255,0))
    img_with_lane_weight =  cv2.addWeighted(img, 1.0, temp2, 0.3, 0.0)
    return img_with_lane_weight

def trace_lane_line_with_coefficients(img, line_coefficients, top_y, make_copy=True):
    A = line_coefficients[0]
    b = line_coefficients[1]
    img_shape = img.shape
    bottom_y = img_shape[0] - 1
    # y = Ax + b, therefore x = (y - b) / A
    
    try:
        x_to_bottom_y = (bottom_y - b) / A
    except ZeroDivisionError:
        x_to_bottom_y = 2*img.shape[1]/10
    try:
        top_x_to_y = int((top_y - b) / A)
    except:
        top_x_to_y = 4*img.shape[1]/10

    new_lines = [[[int(x_to_bottom_y), int(bottom_y), int(top_x_to_y), int(top_y)]]]
    return draw_lines(img, new_lines, make_copy=make_copy)

def trace_both_lane_lines_with_lines_coefficients(img, left_line_coefficients, right_line_coefficients):
    vert = get_vertices_for_img(img)
    region_top_left = vert[0][1]
    temp = np.zeros(img.shape,dtype=np.uint8)
    full_left_lane_img = trace_lane_line_with_coefficients(temp, left_line_coefficients, region_top_left[1], make_copy=True)
    full_left_right_lanes_img = trace_lane_line_with_coefficients(full_left_lane_img, right_line_coefficients, region_top_left[1], make_copy=False)
    full_left_right_lanes_img = cv2.cvtColor(full_left_right_lanes_img,cv2.COLOR_BGR2GRAY)
    full_left_right_lanes_img[full_left_right_lanes_img.shape[0]-1,:] = full_left_right_lanes_img.max()
    contours, hierarchy = cv2.findContours(full_left_right_lanes_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted(contours,key=cv2.contourArea, reverse=True)
    contour = contours[0]
    full_left_right_lanes_img = cv2.fillPoly(temp,[contour],(0,255,0))
    # image1 * α + image2 * β + λ
    # image1 and image2 must be the same shape.
    full_left_right_lanes_img[:62*full_left_right_lanes_img.shape[0]//100,:] = 0 # only consider bottom 38% of the mask(near car)
    # img_with_lane_weight =  cv2.addWeighted(img, 1.0, full_left_right_lanes_img, 0.3, 0.0)
    # return img_with_lane_weight
    return full_left_right_lanes_img
