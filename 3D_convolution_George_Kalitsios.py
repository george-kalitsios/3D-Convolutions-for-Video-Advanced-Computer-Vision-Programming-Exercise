# HOMEWORK ProgrammingExerciseACV2020 
# Student : George Kalitsios
# Student Number : 62 
# Final Version

import numpy as np
import cv2
import skvideo
import skvideo.io
import time

def myConv3D(A, B ,param):
    '''
    your code for 3D convolution should be placed here
    '''
    checkparam = param["same"]
    nH=A.shape[1]
    nW=A.shape[2]
    f=B.shape[0]
    m=A.shape[0]
    pad = 1 
    stride = 1
    nH_new = int((nH-f+2*pad)/stride)+1	    
    nW_new = int((nW-f+2*pad)/stride)+1
    conv_output_3D = np.zeros((m-2, nH_new, nW_new))	    	
    if checkparam==1 :						
        A_pad_image = pad_image(A, f)
    else:
        print("be careful after padding we dont have same dimensions with matrix A !") 	    		
    for i in range(m-2):
        for h in range(nH_new):                        							  			
            vertical_start = stride*h
            vertical_end = stride*h+f            
            for w in range(nW_new):      										 	  			
                horizontal_start = stride*w
                horizontal_end = stride*w+f
                a_slice_1 = A_pad_image[i, vertical_start:vertical_end, horizontal_start:horizontal_end]
                a_slice_2 = A_pad_image[i+1, vertical_start:vertical_end, horizontal_start:horizontal_end]
                a_slice_3 = A_pad_image[i+2, vertical_start:vertical_end, horizontal_start:horizontal_end]
                conv_output_3D[i,h,w] = conv_single_step(a_slice_1,B[0])+conv_single_step(a_slice_2,B[1])+conv_single_step(a_slice_3,B[2])     
    return conv_output_3D


def create_smooth_kernel(size):
	'''
	your code for kernel creation should be placed here
	'''
	return np.ones((size,size,size))/size**3


def pad_image(A, size):
	'''
	your code for volume padding should be placed here
	'''
	padding=int((size-1)/2)
	A_pad_image = np.pad(A, ((0,0), (padding,padding), (padding,padding)), mode='constant', constant_values = (0,0)) 	 	
	return A_pad_image

def conv_single_step(slice, filt): 
    s = np.multiply(slice,filt)
    Z = np.sum(s)
    return Z

def main():
    start_time = time.time()
    kernel=create_smooth_kernel(3)
    print("kernel.shape =\n",kernel.shape)
    param = {"same" : 1, "valid": 0}
    # Extract frames from Video & grayscale transform
    cap = cv2.VideoCapture('video.mp4')
    frame_Count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_Width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_Height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buf = np.empty((frame_Count, frame_Height, frame_Width), np.dtype('uint8'))
    fc = 0
    while (fc < frame_Count):
        ret, img = cap.read()
        buf[fc]  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        fc += 1
    cap.release()
    cv2.destroyAllWindows() 
    # Calculate 3D convolution
    conv_output_3D = myConv3D(buf, kernel,param)
    # Writing video after 3D Convolution
    conv_output_3D = conv_output_3D.astype(np.uint8)
    skvideo.io.vwrite("videofinal.mp4", conv_output_3D)
    print("Total running time : %s seconds " % (time.time() - start_time))
    
if __name__ == "__main__":
    main()