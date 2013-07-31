'''
Submit Mar 30, 2010

@author: Elisabeth Childs
'''


import numpy
import math
import imgutil
import cv
import optparse
from scipy.ndimage import filters, interpolation


def homogeneous(xywIn):
	'''
	Converts a 2xN or 3xN set of points to homogeneous coordinates.
		- for 2xN arrays, adds a row of ones
		- for 3xN arrays, divides all rows by the third row
	'''
	if xywIn.shape[0] == 3:
		xywOut = numpy.zeros_like(xywIn)
		for i in range(3):
			xywOut[i, :] = xywIn[i, :] / xywIn[2, :]
	elif xywIn.shape[0] == 2:
		xywOut = numpy.vstack((xywIn, numpy.ones((1, xywIn.shape[1]), dtype=xywIn.dtype)))
	
	return xywOut

def transformImage(input, tform, output_range="same"):
	'''
	Transform the input image using a 3x3 transform.  The array tform should
	specify the 3x3 transform from input to output coordinates, a forward
	mapping.  This matrix needs to have an inverse.
	
	output_range should be one of:
		"same" - match the range from the input image
		"auto" - determine a range by applying the transform to the corners of the input image
		ndarray - a 2x2 numpy array like [(x1, y1), (x2, y2)]
	'''
	h, w = input.shape[0:2]
	
	yy, xx = numpy.mgrid[0:h, 0:w]
	yc, xc = h/2.0, w/2.0
	dist = (yy-yc)**2 + (xx-xc)**2
	sigma = float(min(h,w))/4
	gausianWeight = numpy.exp(-dist/(2*sigma**2))
	
	input = numpy.dstack((input, gausianWeight))
	
	# determine pixel coordinates of output image
	if output_range=="auto":
		
		xmin, xmax, ymin, ymax = getCorners(w,h, tform)
		
		yy, xx = numpy.mgrid[ymin:ymax, xmin:xmax]
		
		h = ymax - ymin
		w = xmax - xmin
		 
	elif type(output_range) == numpy.ndarray:

		xmin,  ymin, xmax, ymax = output_range[0, 0], output_range[0, 1], output_range[1, 0], output_range[1, 1]
		
		
		yy, xx = numpy.mgrid[ymin:ymax, xmin:xmax]
		
		h = ymax - ymin
		w = xmax - xmin
		
	else:
		yy, xx = numpy.mgrid[0:h, 0:w]
		
	# transform output pixel coordinates into input image coordinates
	xywOut = numpy.vstack((xx.flatten(), yy.flatten()))
	xywOut = homogeneous(xywOut)
	xywIn = numpy.dot(numpy.linalg.inv(tform), xywOut)
	xywIn = homogeneous(xywIn)
	
	 
	# reshape input image coordinates
	xxIn = xywIn[0,:].reshape((h, w))
	yyIn = xywIn[1,:].reshape((h, w))
	
	# sample output image
	if input.ndim == 3:
		output = numpy.zeros((h, w, input.shape[2]), dtype=input.dtype)
		for d in range(input.shape[2]):
			output[..., d] = interpolation.map_coordinates(input[..., d], [yyIn, xxIn])
	else:
		output = interpolation.map_coordinates(input, [yyIn, xxIn])
		
	return output

def makeCenteredRotation(angle, center=(0, 0)):
	'''
	Creates a 3x3 transform matrix for centered rotation.
	
	angle is the angle for the rotation 
	center specifies the axis location for the rotation
	
	'''
	# center
	Cf = numpy.array([[1, 0, -center[0]],
					  [0, 1, -center[1]],
					  [0, 0, 1]])
	Cb = numpy.linalg.inv(Cf)
	
	# rotate
	cost = numpy.cos(numpy.deg2rad(angle))
	sint = numpy.sin(numpy.deg2rad(angle))
	R = numpy.array([[cost, -sint, 0],
					 [sint, cost, 0],
					 [0, 0, 1]])
	
	# compose
	A = numpy.dot(R, Cf)
	A = numpy.dot(Cb, A)
	return A

def homography( p1, p2):
	'''
	A function that returns the transformation matrix that gets the two sets of points 
	into the same space 
	
	p1 is a list of points in a space that correspond the the list of points p2 that are
	in a different space. 
	
	Returns a matrix that represent the transformation of the points p2 to the space of p1
	'''
		 
	# the matrix used to find the homography  
	A = numpy.zeros((p1.shape[0]*2, 8))
	
	for i in range(p1.shape[0]*2):
		if ( i % 2 == 0):
			A[i,0] = p1[i/2,0]
			A[i,1] = p1[i/2,1]
			A[i,2] = 1
			A[i,6] = -p1[i/2,0]*p2[i/2,0]
			A[i,7] = -p1[i/2,1]*p2[i/2,0]
		else:
			A[i,3] = p1[i/2,0]
			A[i,4] = p1[i/2,1]
			A[i,5] = 1
			A[i,6] = -p1[i/2,0]*p2[i/2,1]
			A[i,7] = -p1[i/2,1]*p2[i/2,1]
	
	#the list of the points in the second image
	b = p2.reshape(p2.shape[0]*p2.shape[1], 1)
	
	# use the idea Ax = B to find the homagraphy matrix
	x = numpy.linalg.lstsq(A, b)[0]
	
	x = numpy.vstack((x, numpy.array([1])))
	
	x = x.reshape((3,3))
	
	return x

def getCorners(w,h, tform):
	'''
	returns the corners of a rectangle that encompasses the transformed original rectangle
	that has the width and hight that was passed
	'''	  
	
	inputCorners = numpy.array( [[0, w, w, 0],
								 [0, 0, h, h],
								 [1, 1, 1, 1]])
		
	outputCorners = numpy.dot(tform, inputCorners)
	
	outputCorners = homogeneous(outputCorners)
		
	xmin = int(numpy.min(outputCorners[0, :]))
	xmax = int(numpy.max(outputCorners[0, :]))
	ymin = int(numpy.min(outputCorners[1, :]))
	ymax = int(numpy.max(outputCorners[1, :]))
	
	return xmin, xmax, ymin, ymax
	
def testRotate():
	'''
	a test function for the rotation function
	'''
	# handle command line arguments
	parser = optparse.OptionParser()
	parser.add_option("-f", "--filename", help="input image file")
	parser.add_option("-a", "--angle", help="rotation angle in degrees", default=0, type="float")
	options, remain = parser.parse_args()
	if options.filename is None:
		parser.print_help()
		exit(0)
	
	# load image
	cvimg = cv.LoadImage(options.filename)
	npimg = imgutil.cv2array(cvimg)
	
	# rotate image
	h,w = npimg.shape[0:2]
	A = makeCenteredRotation(options.angle, (w/2.0, h/2.0))
	nprot = transformImage(npimg, A)
	
	imgutil.imageShow(npimg, "original")
	imgutil.imageShow(nprot, "rotate")
	cv.WaitKey(0)
	
if __name__ == "__main__":
	testRotate()