'''
Created on Mar 21, 2011

Functions for finding, describing, and matching image feature points.

@authors: Elisabeth Childs
'''

import cv
import imgutil
import numpy
import math
import transform
from scipy.ndimage import filters, interpolation
from scipy.spatial import distance

class Feature():
    
    def __init__(self):
    	'''
    	Sets up the feature class with two images and three colors to use when 
    	displaying matched features
    	'''
        self.imageOne = None
        self.imageTwo = None
        
        self.featureColor = (60,60,200)
        self.matchColor = (60, 200, 60)
        self.inliersColor = (200, 100, 20)
        
    
    def harris(self, image, sigmaD=1.0, sigmaI=1.5, count=512):
        '''
        Finds Harris corner features for an input image.  The input image
        should be a 2D numpy array.  The sigmaD and sigmaI parameters define
        the differentiation and integration scales (respectively) of the
        Harris feature detector---the defaults are reasonable for many images.
        
        Returns:
        an array of locations that has the dimensions maxPoints x 3 array, 
        each row contains the (x, y) position, and Harris score of a feature point.  
        The array is sorted by decreasing score.
        '''
        image = image.astype(numpy.float32)
        h, w = image.shape[0:2]
        
        # compute image derivatives
        Ix = filters.gaussian_filter1d(image, sigmaD, 0, 0)
        Ix = filters.gaussian_filter1d(Ix, sigmaD, 1, 1)
        Iy = filters.gaussian_filter1d(image, sigmaD, 1, 0)
        Iy = filters.gaussian_filter1d(Iy, sigmaD, 0, 1)
        
        # compute elements of the structure tensor
        Ixx = filters.gaussian_filter(Ix**2, sigmaI, 0)
        Iyy = filters.gaussian_filter(Iy**2, sigmaI, 0)
        Ixy = filters.gaussian_filter(Ix * Iy, sigmaI, 0)
        
        # compute Harris feature strength, avoiding divide by zero
        imgH = (Ixx * Iyy - Ixy**2) / (Ixx + Iyy + 1e-8)
            
        # exclude points near the image border
        imgH[:16, :] = 0
        imgH[-16:, :] = 0
        imgH[:, :16] = 0
        imgH[:, -16:] = 0
        
        # non-maximum suppression in 5x5 regions
        maxH = filters.maximum_filter(imgH, (5,5))
        imgH = imgH * (imgH == maxH)
        
        # sort points by strength and find their positions
        sortIdx = numpy.argsort(imgH.flatten())[::-1]
        sortIdx = sortIdx[:count]
        yy = sortIdx / w
        xx = sortIdx % w
            
        # concatenate positions and values
        xyv = numpy.vstack((xx, yy, imgH.flatten()[sortIdx])).transpose()
        return xyv
    
    def featureDescriptor(self, image, featLoc, radius, scale):
        '''
            A function that extracts feature descriptors from an image
            given the set of feature locations. Each feature is composed of pixel 
            values in an axis-aligned square neighborhood surrounding the 
            feature point.
            
            image   - the image to extract the features from
            featLoc - the harris corner feature locations (Array num
            Radius  - the radius to sample from in each image
            Scale   - the scale to use to sample each image 
            n       - the number of feature descriptors to use from the image 
             
            The function returns a w x w x N collection of normalized feature patches.
        '''
        
        #Sets up the vars of the function 
        n = featLoc.shape[0]
        image = image.astype(numpy.float32)
        
        sigma = scale 
        smoothImg = filters.gaussian_filter(image, sigma, 0)
        
        w = 2*radius + 1
        patches = numpy.ones((w,w,n), numpy.float32)
        
        #Sets up the feature patches
        for i in range(n):
            
            centerX, centerY = featLoc[i,0], featLoc[i,1]
            
            ymin = centerY - scale * radius
            ymax = centerY + scale * radius + 1
            xmin = centerX - scale * radius
            xmax = centerX + scale * radius + 1
            
            featDescription = smoothImg[ymin: ymax:scale, xmin: xmax:scale]
            
            mean = numpy.mean(featDescription)
            stdev = numpy.std(featDescription)
            
            normDes = (featDescription-mean)/stdev
            
            patches[:,:,i] = normDes
        
        return patches
    
    def displayDescriptors(self, descriptorArray):
        '''
        Displays the first twenty five prominent features in a grid from the 
        descriptorArray 
        
        Returns the image
        '''
        wAndH = descriptorArray.shape[0]
        numDesc = descriptorArray.shape[2]
        
        #sets up the numpy image where the descriptors will be put
        img = numpy.zeros((5*wAndH,5*wAndH))
        for j in range(5):
            for i in range(5):
                #checks to make sure you dont go over the number of descriptors
                if i*5+j <= numDesc:
                    startY = i * wAndH
                    endY = (i + 1) * wAndH
                    
                    startX = j * wAndH
                    endX = (j + 1) * wAndH
                    
                    img[startY:endY, startX:endX] = descriptorArray[:,:,i*5+j]
        
        imgutil.imageShow(img, "descriptor")
        
        return img
    
    def matchDescriptors(self, imgOne, imgTwo, tolerance):
        '''
            imgOne are the descriptors from image one shape N x w squared
            imgTwo are the descriptors from image two  M x w squared
            
             Perform an initial feature matching for a pair of images by 
             computing the Euclidean distance between all potential descriptor 
             matches. You may find the function scipy.spatial.distance.cdist() 
             incredibly helpful for distance computations. Compute the ratios of 
             closest distance for each feature to the average second closest 
             distance for all features. Threshold the ratios to exclude unlikely 
             matches.
        '''
        M = imgOne.shape[0]
        dists = distance.cdist(imgOne, imgTwo)
        
        sortIdx = numpy.argsort(dists, 1)
        
        bestIdx = sortIdx[:, 0]
        nextbestIdx = sortIdx[:, 1] 
        
        indexes = numpy.r_[0:M]
        bestDist = dists[indexes, bestIdx]
        nextbestMean = numpy.mean(dists[indexes, nextbestIdx])
        
        ratio = bestDist/nextbestMean
        
        matches = numpy.zeros((ratio[ratio < tolerance].shape[0], 2), dtype = numpy.int32)
        matches[:,0] = indexes[ratio < tolerance]
        matches[:,1] = bestIdx[ratio < tolerance]
        
        return matches
   
    def RANSAC(self, correspondence, tolerence):
        '''
        this function finds the best homography to transform one image to the other with 
        the most matches between the features of each of the images.
        
        correspondence - a list of features from each image that seems to have a 
        				 correspondence in the other image. The list is the number of 
        				 corresponding features, by 4 where each row x1 and y1 in image one 
        				 and x2 y2 in image two
        tolerance - is the how close the matches need to be to count at a close model 
        '''
        
        confidence = 0.90           #the desired confidence in the model
        min = 4						#the minimin number of feature matches
        
        numPoints = correspondence.shape[0] #the number of matches
        bestFit = numpy.zeros((3,3))        #sets best fit model to an empty matrix 
        inliers = None                      #inlires are the matches that fit in the bestFit 
        inlierCount = 0                     #number of inliers
        pts = correspondence[:,:2]          #set of first points
        pts2 = correspondence[:,2:]         #set of second points
        indices = numpy.r_[0:numPoints]     #list of indices to choose the random points to find the holography  
        maxIterations = 1                   #a variable for the maximum iterations to test
        iterations = 0                      #current number of iterations 
        
        while iterations < maxIterations:
            #Choose four random point
            numpy.random.shuffle(indices)   
            randPts = correspondence[indices[:4]]
            
            p1 = randPts[:,:2]
            p2 = randPts[:,2:]
            
            #Create a possible model
            model = transform.homography(p2, p1)
            
            transPts = numpy.ones((numPoints,3))
            transPts[:,:2] = pts2
            
            #find the matches
            transformMatches = numpy.dot(model, transPts.transpose())
            transformMatches = transform.homogeneous(transformMatches).transpose()
        
            secondPoints = transformMatches[:,:2]
     		
     		#Test how well the model fits
            dist = numpy.sqrt(((pts - secondPoints)**2).sum(1))
            num = dist[dist<tolerence].shape[0]
            
            
            if num > inlierCount:
                inliers = indices[dist<tolerence]
                inlierCount = num
                bestFit = model

            	percentFit = float(inlierCount)/float(numPoints)
            
            	if percentFit == 0:
                	maxIterations = iterations + 2
            	else:
                	maxIterations = int((math.log(1-confidence))/(math.log(1-math.pow(percentFit, min))))
            
            iterations += 1
            if iterations > 5000:
            	if inlierCount > 4:
                	return bestFit, inliers
            	else:
                	print "There isn't a good model for your images"
                	exit() 
        
        return bestFit, inliers
              
    def displayMatches(self, matchPts, inliers):
        '''
        a display is shown of the matches that connect the two images
        '''
        matchOne = matchPts[:,:2]
        matchTwo = matchPts[:,2:]
        
        self.drawSquare(self.imageOne, matchOne, self.matchColor)
        self.drawSquare(self.imageTwo, matchTwo, self.matchColor)
        
        inlierMatchOne = matchOne[inliers]
        inlierMatchTwo = matchTwo[inliers]
        
        self.drawSquare(self.imageOne, inlierMatchOne, self.inliersColor)
        self.drawSquare(self.imageTwo, inlierMatchTwo, self.inliersColor)
        
        npimg = imgutil.cv2array(self.imageOne)
        npimg2 = imgutil.cv2array(self.imageTwo)
        
        displayPic = numpy.hstack((npimg,npimg2))
        
        #add the width of image one to image two
        #inlierMatchTwo[:,0] =  inlierMatchTwo[:,0] + npimg.shape[1]
        
        cvimg = imgutil.array2cv(displayPic)
        
        
        for i in range(inlierMatchOne.shape[0]):
            
            x1, y1 = int(inlierMatchOne[i,0]), int(inlierMatchOne[i,1])
            x2, y2 = int(inlierMatchTwo[i,0]+ npimg.shape[1]), int(inlierMatchTwo[i,1])
            
            cv.Line(cvimg, (x1,y1), ( x2, y2), self.inliersColor )
        
        imgutil.imageShow(cvimg, "matchImg")
        cv.WaitKey(0)
        
    def getMatchPoints(self, matches, ptsOne, ptsTwo):
        '''
        returns the matched points from the matches based on the two different sets of points
        '''
        firstPts = ptsOne[matches[:,0],:2]
        secondPts = ptsTwo[matches[:,1],:2]
        
        matchPts = numpy.hstack((firstPts, secondPts))
        
        return matchPts
    
    def drawSquare(self, cvimg, pts, color):
    	'''
    	draws squares around the points past
    	cvimg - the image on wich to draw the squares
    	pts - the points around witch the squares should be drawn
    	color - the color for the squares
    	'''
        for pxy in pts:
            cv.Rectangle(cvimg, (int(pxy[0] - 5), int(pxy[1] - 5)), (int(pxy[0] + 5), int(pxy[1] + 5)), color)
    
    def setupImages(self,cvimg, display = False):
    	'''
    	sets up the images to find the harris corners and patches in the image passed 
    	optionally displays some of the features found  
    	'''
        blue = .114 
        green = .587
        red = .299
        npimg = imgutil.cv2array(cvimg)
        
        
        grey = npimg[:,:,0]*blue + npimg[:,:,1]*green + npimg[:,:,2]*red
        
        harrisPoints = self.harris(grey)
  
        self.drawSquare(cvimg, harrisPoints, self.featureColor)
        
        patches = self.featureDescriptor(grey, harrisPoints, 6, 1)
        
        #displays the descriptos if the user chooses to`
        if display == True:
       		self.displayDescriptors(patches)
        
        w = patches.shape[0]
        N = patches.shape[2]
        #reshapes the discriptors
        reshapePatches = patches.reshape((w**2,N)).transpose() 
        
        return reshapePatches, harrisPoints
        
    def getBestHomography(self, imgOne, imgTwo, display = False):
        '''
        	Finds the best homagraphy that transforms imgTwo to imgOne's image plane and
        	optionally displays the results 
        '''
        
        self.imageOne = imgOne
        self.imageTwo = imgTwo

        
        patchesOne, harrisPtsOne = self.setupImages(self.imageOne)
        patchesTwo, harrisPtsTwo = self.setupImages(self.imageTwo)
        
        matches = self.matchDescriptors(patchesOne, patchesTwo, 0.6)
        
        if matches.shape[0] < 4:
            print "There aren't enough matches"
            exit()
        
        matchPts = self.getMatchPoints(matches, harrisPtsOne, harrisPtsTwo)
        
        matchOne = matchPts[:,:2]
        matchTwo = matchPts[:,2:]
        
        self.drawSquare(self.imageOne, matchOne, self.matchColor)
        self.drawSquare(self.imageTwo, matchTwo, self.matchColor)
        
        
        bestfit, inliers = self.RANSAC(matchPts, 0.5)
        
        #displays the matches if display is true
        if display == True:
        	self.displayMatches(matchPts, inliers)
        
        return bestfit


if __name__ == "__main__":
    cvimg = cv.LoadImage("../SmallPics/pics/2/image-0003.jpg")
    cvimg2 = cv.LoadImage("../SmallPics/pics/2/image-0004.jpg")
    feature = Feature()
    feature.getBestHomography(cvimg,cvimg2)
    
    
    cv.WaitKey(0)
    
    
    
    
    
    
    
    
    
    