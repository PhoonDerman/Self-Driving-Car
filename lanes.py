import cv2
import numpy as np
import matplotlib.pyplot as plt



#coordinates for starting at the bottom of image
def make_coordinates(image, line_parameters):
    slope,intercept=line_parameters
    y1=image.shape[0]
    y2=int(y1*(3/5))
    x1=int((y1-intercept)/slope)
    x2=int((y2-intercept)/slope)
    return np.array((x1,y1,x2,y2))



def average_slope_intercept(image, lines):
    left_fit=[]
    right_fit=[]
    for line in lines:
        x1,y1,x2,y2=line.reshape(4)
        parameters=np.polyfit((x1,x2),(y1,y2),1)
        slope=parameters[0]
        intercept=parameters[1]
        if slope<0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    left_fit_average=np.average(left_fit,axis=0)
    right_fit_average=np.average(right_fit,axis=0)
    left_line=make_coordinates(image,left_fit_average)
    right_line=make_coordinates(image,right_fit_average)
    return np.array((left_line,right_line))





def canny(image):
    #Grayscale
    grap=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #GaussianBlur
    blur=cv2.GaussianBlur(grap, (5, 5), 0)
    #Canny
    canny=cv2.Canny(blur, 50, 150)
    return canny


def display_lines(image, lines):
    line_image=np.zeros_like(image)
    if lines is not None:
        for x1, y1,x2,y2 in lines:
              #2d array(1row and 2colm) into 1d array
            cv2.line(line_image, (x1,y1), (x2,y2), (255, 0, 0), 10)
            #(coordinates the lines to be drawn, color, thickness of lines)

    return line_image

def region_of_interest(image):
    #traingle for lane outline:
    height=image.shape[0]
    triangle=np.array([
    [(200,height ), (1100, height), (550, 250)]
    ])
    mask=np.zeros_like(image)
    #filling the triangle with white
    cv2.fillPoly(mask, triangle, 255)
    # computing bitwise of the image
    masked_image=cv2.bitwise_and(image, mask)
    return masked_image

# image =cv2.imread('test_image.jpg')
# lane_image=np.copy(image)
# canny_image=canny(lane_image)
# cropped_image=region_of_interest(canny_image)
#
# #line=cv2.HoughLines(1st argument[image to detect lines],
# #2nd and 3rd argument[resolution of hough accumilator array ],
# #4th argu[threshold aka bins with highest votes], 5th argu[Placeholder array],
# #6th argu[lenght of line in pixels to accpt into output],
# #7th aru[max distant in pixels between sigmented lines which connected into single line ]  )
#
#
# lines=cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5 )
# #1 degree to radians= pi/180
# averaged_lines=average_slope_intercept(lane_image, lines)
# line_image=display_lines(lane_image, averaged_lines)
# #adding weight to the image
# combo_image=cv2.addWeighted(lane_image, 0.8, line_image, 1,1) #imge weight, iw, gamma
# cv2.imshow('result', combo_image)
# #wkey function
# cv2.waitKey(0)

cap=cv2.VideoCapture('test2.mp4')
while(cap.isOpened()):
    _, frame=cap.read()
    canny_image=canny(frame)
    cropped_image=region_of_interest(canny_image)

    lines=cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5 )
    #1 degree to radians= pi/180
    averaged_lines=average_slope_intercept(frame, lines)
    line_image=display_lines(frame, averaged_lines)
    #adding weight to the image
    combo_image=cv2.addWeighted(frame, 0.8, line_image, 1,1) #imge weight, iw, gamma
    cv2.imshow('result', combo_image)
    #wkey function
    cv2.waitKey(1) #1milliseconds
    if cv2.waitKey(1) == ord('q'): #to close video after press button
        break
cap.release()
cv2.destroyAllWindows()
