# -*- coding: utf-8 -*-
"""
Created on Fri May 31 22:02:08 2019

@author: mahesh.s.reddy315
"""



         
def KeyClick(inp):
    import ctypes
    import time
    
    SendInput = ctypes.windll.user32.SendInput
    
    # C struct redefinitions 
    PUL = ctypes.POINTER(ctypes.c_ulong)
    class KeyBdInput(ctypes.Structure):
        _fields_ = [("wVk", ctypes.c_ushort),
                    ("wScan", ctypes.c_ushort),
                    ("dwFlags", ctypes.c_ulong),
                    ("time", ctypes.c_ulong),
                    ("dwExtraInfo", PUL)]
    
    class HardwareInput(ctypes.Structure):
        _fields_ = [("uMsg", ctypes.c_ulong),
                    ("wParamL", ctypes.c_short),
                    ("wParamH", ctypes.c_ushort)]
    
    class MouseInput(ctypes.Structure):
        _fields_ = [("dx", ctypes.c_long),
                    ("dy", ctypes.c_long),
                    ("mouseData", ctypes.c_ulong),
                    ("dwFlags", ctypes.c_ulong),
                    ("time",ctypes.c_ulong),
                    ("dwExtraInfo", PUL)]
    
    class Input_I(ctypes.Union):
        _fields_ = [("ki", KeyBdInput),
                     ("mi", MouseInput),
                     ("hi", HardwareInput)]
    
    class Input(ctypes.Structure):
        _fields_ = [("type", ctypes.c_ulong),
                    ("ii", Input_I)]
    
    # Actuals Functions
    
    def PressKey(hexKeyCode):
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
        x = Input( ctypes.c_ulong(1), ii_ )
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
    
    def ReleaseKey(hexKeyCode):
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
        x = Input( ctypes.c_ulong(1), ii_ )
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
    
    # directx scan codes http://www.gamespp.com/directx/directInputKeyboardScanCodes.html
    if(inp==1):
        PressKey(0xcd)#0x11)
    #time.sleep(1)
    elif(inp==2):
        ReleaseKey(0xcd)#0x11)
    #time.sleep(1)
    elif(inp==3):
        PressKey(0xcb)
    elif(inp==4):
        ReleaseKey(0xcb)



import cv2
import matplotlib.pyplot as plt

background = None
accumulated_weight=0.5
global z

roi_top=20                  #     20
roi_bottom=200              #    300
roi_right=50             #    50
roi_left=250              #350

roi2_top=20                  #     20
roi2_bottom=200              #    300
roi2_right=400               #    50
roi2_left=600            #350


def cal_accum_avg(frame,accumulated_weight):
    global background
    
    if background is None:
        background=frame.copy().astype('float')
        return None
    
    cv2.accumulateWeighted(frame,background,accumulated_weight)

def cal_accum_avg2(frame,accumulated_weight):
    global background
    
    if background is None:
        background=frame.copy().astype('float')
        return None
    
    cv2.accumulateWeighted(frame,background,accumulated_weight)


def segment(frame,threshold_min=25):
    diff=cv2.absdiff(background.astype('uint8'),frame)
    _,thresholded=cv2.threshold(diff,threshold_min,255,cv2.THRESH_BINARY)
    img,contours,hir=  cv2.findContours(thresholded.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    if len(contours)==0:
        return None
    
    else:
        hand_segment=max(contours,key=cv2.contourArea)
        
#        global z
#        z.append(contours)
        cnt=hand_segment
        area=cv2.contourArea(cnt)
        
        if(area>10000):
            
            print('Release')
            KeyClick(4)
        else:
            print('Accelerate')
            KeyClick(3)
        #print(area)
        return(thresholded,hand_segment)
    
def segment2(frame,threshold_min=25):
    diff=cv2.absdiff(background.astype('uint8'),frame)
    _,thresholded=cv2.threshold(diff,threshold_min,255,cv2.THRESH_BINARY)
    img,contours,hir=  cv2.findContours(thresholded.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    if len(contours)==0:
        return None
    
    else:
        hand_segment=max(contours,key=cv2.contourArea)
        
#        global z
#        z.append(contours)
        cnt=hand_segment
        area=cv2.contourArea(cnt)
        if(area>10000):
            print('Release2')
            KeyClick(2)
        else:
            print('Accelerate2')
            KeyClick(1)
        #print(area)
        return(thresholded,hand_segment)
        
        
        
video=cv2.VideoCapture(0)
num_frames=0
while True:
    ret1,frame=video.read()
    frame_copy=frame.copy()
    frame_copy=cv2.flip(frame,1)
    roi=frame_copy[roi_top:roi_bottom,roi_right:roi_left]
    roi2=frame_copy[roi2_top:roi2_bottom,roi2_right:roi2_left]
    
    gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    gray2=cv2.cvtColor(roi2,cv2.COLOR_BGR2GRAY)
    
    gray=cv2.GaussianBlur(gray,(3,3),0)
    gray2=cv2.GaussianBlur(gray2,(3,3),0)
    
    if(num_frames<60):
        cal_accum_avg(gray,accumulated_weight)
        cal_accum_avg2(gray2,accumulated_weight)
        
        if(num_frames<=59):
            cv2.putText(frame_copy,'Wait',(200,300),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            cv2.imshow("Finger Count",frame_copy)
            
    else:
        hand=segment(gray)
        hand2=segment2(gray2)
        
        if hand is not None and hand2 is not None:
            thresholded,hand_segment=hand
            thresholded2,hand_segment2=hand2
            
            cv2.drawContours(frame_copy,[hand_segment+(roi_right,roi_top)],-1,(255,0,0),5)
            cv2.drawContours(frame_copy,[hand_segment2+(roi2_right,roi2_top)],-1,(255,0,0),5)
            
            #fingers=count_fingers(thresholded,hand_segment)
            #cv2.putText(frame_copy,str(fingers),(70,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            
           # cv2.imshow('Thresholded',thresholded)
    
    
    cv2.rectangle(frame_copy,(roi_left,roi_top),(roi_right,roi_bottom),(0,0,255),5)
    cv2.rectangle(frame_copy,(roi2_left,roi2_top),(roi2_right,roi2_bottom),(0,0,255),5)
    num_frames+=1                #600    20        300      300
    
    cv2.imshow('Finger Count',frame_copy)
    
    key=cv2.waitKey(1)
    if(key==27):
        break
video.release()
print("Hello")
cv2.destroyAllWindows()        
        
        


        

#video=cv2.VideoCapture(0)
#
#while(1):
#    check,frame=video.read()
#    flip=cv2.flip(frame,1)    
#    cv2.imshow('video',flip)
#    key=cv2.waitKey(1)& 0xFF
#    if key==27:
#        break
#
#video.release()
#cv2.destroyAllWindows()




#z=plt.imread('Hand.jpg')
#plt.imshow(z)
#
#import numpy as np
#import cv2
#import matplotlib.pyplot as plt
#
#img=cv2.imread('Hand.jpg',1)
#img1=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#ret,threshold=cv2.threshold(img1,200,255,cv2.THRESH_BINARY_INV)
#kernel=np.ones((5,5),np.uint8)
#erosion=cv2.erode(threshold,kernel,iterations=2)
#contours,hier=  cv2.findContours(erosion.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(img,contours,-1,(255,0,0),2)
#cv2.imshow('Image',img)
#cv2.imshow('Gray',img1)
#cv2.imshow('Threshold',threshold)
#cv2.imshow('Eroded',erosion)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#print(contours[0][:][0].shape)
#
#plt.plot([235,236,237,238],[35,36,37,38])
#
#
#
#
#cnt=contours[0]
#M = cv2.moments(cnt)
#area=cv2.contourArea(cnt)