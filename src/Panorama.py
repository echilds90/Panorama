import cv
import glob
import numpy
import os.path
import imgutil
import optparse

import ginput 
import transform

import Feature


'''
Submit Mar 30, 2010

@author: Elisabeth Childs
'''

class Panorama:
    '''
        initializes the panorama class with a path and middle picture number
    '''
    def __init__(self, path = None, middlePic = 0 , pick = False):
        
        '''
            The first part of the init constructs a list of the files in the 
            directory passed in
        '''
        
        self.pick = pick
        
        # find image files in the directory
        exts = ["*.jpg", "*.JPG"]
        self.files = []
        for ext in exts:
            self.files += glob.glob(os.path.join(path, ext))
        self.files.sort()
        print "numFiles",len(self.files)
        
        #An enumirated type type object to be used for window names
        self.enumImageNames = ["First", "Second", "Third", "Fourth", "Fifth"]
        
        #The middle image number 
        self.middlePic = middlePic
        
        #the Identity matrix, used to tranform the main image 
        self.identity = numpy.array([[1, 0, 0], 
                                      [0, 1, 0],
                                      [0, 0, 1]])
        
        self.automaticFeatures = Feature.Feature()

        # a lits of the panorama objects 
        self.panImages = []
        
        #A list of the original homographies
        self.homographies = []
    
    def testRectifying(self):
        '''
        This function tests the rectification of an image 
        it is structured so you have to manually manipulate 
        the four corners that the points you choose to go to
    	'''
        cv.NamedWindow("grab", cv.CV_WINDOW_NORMAL)
        grab = ginput.Grab("grab", 4)
        
        key = None
        while key != 27:
            # grab frame
            cvimg = cv.LoadImage("../perspective.jpg")
              
            # draw points
            pts = grab.getPoints()
            for pxy in pts:
                cv.Rectangle(cvimg, (pxy[0]-15, pxy[1]-15), (pxy[0]+15, pxy[1]+15), (0, 0, 255))   
            
            # display
            cv.ShowImage("grab", cvimg)
                
            # handle keys
            key = cv.WaitKey(100)
            if key >= 0 and chr(key) == 'c':
                grab.clear(4)

        if pts.shape[0] != 4:
            return 
        npimg = imgutil.cv2array(cvimg)
        
        #The points that the point you choose are transformed to
        squarePts = numpy.array([[0,0],
                                 [100, 0],
                                 [0,200],
                                 [100,200]])
        
        #Finds the homography based on the points 
        homography = transform.homography(pts, squarePts)
        
        output = transform.transformImage(npimg, homography, "auto")
            
        imgutil.imageShow(output, "Homography")
        # handle keys
        cv.WaitKey(0)
    
    def constPanObjects(self):
    	'''
    	constructing the pan objects including a window for each frame
   		'''
        for i in range(len(self.files)):
            cvimg = cv.LoadImage(self.files[i])
            if self.pick == False:
                panImg = PanObj(i, None, cvimg)
            else:
                cv.NamedWindow(self.enumImageNames[i], cv.CV_WINDOW_NORMAL)
                grab = ginput.Grab(self.enumImageNames[i], 6)
                panImg = PanObj(i, grab, cvimg)
            panImg.setNpImg(imgutil.cv2array(cvimg))
            self.panImages.append(panImg)
        print "panImage", len(self.panImages)

    def setHomographies(self, display = False):
    	'''
    	gets the best homography for all the images that are a part of the panorama
    	'''
        for j in range(len(self.panImages) - 1):
            imageOne = self.panImages[j].getNpImg()
            imageTwo = self.panImages[j+1].getNpImg()
            
            #Displays the image info if display is true
            if display == True:
            	imgutil.imageInfo(imageOne, "One")
            	imgutil.imageInfo(imageTwo, "Two")
            
            imageOne = self.panImages[j].getImg()
            imageTwo = self.panImages[j+1].getImg()
            
            #points from the second image are being warped to the first
            homography = self.automaticFeatures.getBestHomography(imageOne, imageTwo)
            self.removeSquares()
            
            self.homographies.append(homography)
            self.panImages[j + 1].setHomography(homography)
    
    def pickPoints(self):
        '''
        Allows the user to select points in the images one at a time instead of 
        automatically building the panorama  
        NOTE: the images should be ordered so that 2 over laps with 1
              and 3 overlaps with 2 and so on
        '''
        for j in range(len(self.panImages)-1):
            key = None
            while key != 27:
                
                self.panImages[j].setNpImg( imgutil.cv2array(self.panImages[j].getImg()))
                  
                # draw points
                imgOneInfo = self.panImages[j]
                pts1 = imgOneInfo.getGrab().getPoints()
                for pxy in pts1:
                    cv.Rectangle(imgOneInfo.getImg(), (pxy[0]-15, pxy[1]-15), (pxy[0]+15, pxy[1]+15), (0, 0, 255)) 
                
                # draw points
                imgTwoInfo = self.panImages[j+1]
                pts2 = imgTwoInfo.getGrab().getPoints()
                for pxy in pts2:
                    cv.Rectangle(imgTwoInfo.getImg(), (pxy[0]-15, pxy[1]-15), (pxy[0]+15, pxy[1]+15), (0, 0, 255))   
                
                
                cv.ShowImage(self.enumImageNames[j], self.panImages[j].getImg())
                cv.ShowImage(self.enumImageNames[j+1], self.panImages[j+1].getImg())
               
                # handle keys
                key = cv.WaitKey(100)
                if key >= 0 and chr(key) == 'c':
                    # display
                    for i in range(len(self.files)):
                        self.panImages[i].getGrab().clear(6)
            
            #clears the grab so it can be used with the next image
            self.panImages[j+1].getGrab().clear(6)
            
            #returns if there are to few points
            if pts1.shape[0] < 4 or pts2.shape[0]  < 4:
                return 
    
            #points from the second image are being warped to the first
            homography = transform.homography(pts2, pts1)
            self.homographies.append(homography)
            self.panImages[j+1].setHomography(homography)
        
        #setting the np Image of the last picture
        self.panImages[-1].setNpImg( imgutil.cv2array(self.panImages[-1].getImg()))
 
    def removeSquares(self):
    	'''
    	Removes the red squares from the images
    	'''
        for i in range(len(self.panImages)):
            cvimg = cv.LoadImage(self.files[i])
            self.panImages[i].setImg(cvimg)
            self.panImages[i].setNpImg(imgutil.cv2array(self.panImages[i].getImg()))

    def MapHomography(self):
        '''
        the function maps all the images in panImages to the plane of the middle image
        '''
        #Maps each image to the plane of the first image
        for i in range(1, len(self.panImages)):
            newHom = numpy.dot(self.panImages[i - 1].getHomography(), self.homographies[i])
            self.panImages[i].setHomography(newHom)
        
        if self.middlePic != 0:
            #Then maps it to any other image if needed, I choose to
            #implement it this way, because it made sense to me to spit
            #up transforming everything to one plane and then from there shifting it
            inverse = numpy.linalg.inv(self.panImages[self.middlePic].getHomography())
            for i in range(len(self.panImages)):
                newHom = numpy.dot(inverse, self.panImages[i].getHomography())
                self.panImages[i].setHomography(newHom)

    def combineImage(self):
    	'''
    	takes each image that should be used in the panorama, and builds the final panorama
    	'''
        #creates the average picture for the finnal output
        output1 = self.panImages[0].getOutput()
        averageTop =  numpy.zeros( output1.shape, dtype = numpy.float32)
        averageBot = numpy.zeros(output1.shape, dtype = numpy.float32)
        average = numpy.zeros(output1.shape, dtype = numpy.float32)
        
        for panImg in self.panImages:
            
            npimg = panImg.getOutput() 
            
            averageTop[:,:,0] += npimg[:,:,3]*npimg[:,:,0]
            averageTop[:,:,1] += npimg[:,:,3]*npimg[:,:,1]
            averageTop[:,:,2] += npimg[:,:,3]*npimg[:,:,2]
                
            averageBot[:,:,0] += npimg[:,:,3]
            averageBot[:,:,1] += npimg[:,:,3]
            averageBot[:,:,2] += npimg[:,:,3]
        
        average[averageBot > 0] = averageTop[averageBot > 0]/averageBot[averageBot > 0]
        
        return average
      
    def panCorners(self):
    	'''
        returns the corners for composite panorama image based on the transformations
        of all the individual images
    	''' 
        xmin, xmax, ymin, ymax = 0,0,0,0
        for i in range(len(self.panImages)):
            h, w = self.panImages[i].getNpImg().shape[0:2] 
            homography = self.panImages[i].getHomography()
            xminPic, xmaxPic, yminPic, ymaxPic = transform.getCorners(w, h, homography)
            
            if xmin > xminPic:
                xmin = xminPic
            if xmax < xmaxPic:
                xmax = xmaxPic
            if ymin > yminPic:
                ymin = yminPic
            if ymax < ymaxPic:
                ymax = ymaxPic
        
        return numpy.array([[xmin, ymin],[xmax, ymax]])
    
    def createPanorama(self, display = False):
        '''
        builds the entire panorama 
        '''
        #adding the identity matrix as the first matrix
        self.homographies.append(self.identity)
        
        self.constPanObjects()
        
        if self.pick == False:
            self.setHomographies()
        else:
            self.pickPoints()
        
        if len(self.panImages) == 0:
        	print "You don't have any images"
        	exit()
        
        #setting the np Image of the last picture
        self.panImages[-1].setNpImg( imgutil.cv2array(self.panImages[-1].getImg()))
        
        self.removeSquares()
        
        self.panImages[0].setHomography(self.identity)
        self.MapHomography()
        
        #gets all the corners of the Panorama    
        corners = self.panCorners()
        
        #transforms all the images based on there homographies
        for panImg in self.panImages:
            npimg = panImg.getNpImg()
            hom = panImg.getHomography()
            output = transform.transformImage(npimg, hom, corners)
            panImg.setOutput(output)
        
        if display == True:
        	for panImg in self.panImages:
        		imgutil.imageShow(panImg.getOutput(), "Panorama")
        
        average =  self.combineImage() 

        imgutil.imageShow(average)
        cv.WaitKey(0)

'''
    An object used to store all the information about an image for the panarama
'''
class PanObj():
    def __init__(self, id, grab, img ):
        self.ID = id
        self.grab = grab
        self.img = img
        self.npimg = None
        self.homography = None
        self.output = None 
    
    def setImg(self, img):
        self.img = img
    
    def getImg(self):
        return self.img
    
    def setNpImg(self, img):
        self.npimg = img
    
    def getNpImg(self):
        return self.npimg
    
    def setGrab(self, grab):
        self.grab = grab
    
    def getGrab(self):
        return self.grab
    
    def setHomography(self, hom):
        self.homography = hom
    
    def getHomography(self):
        return self.homography
    
    def setOutput(self, out):
        self.output = out
    
    def getOutput(self):
        return self.output

if __name__ == "__main__":
    '''
    Main Function for building the panorama
    '''
    # parse command line parameters
    parser = optparse.OptionParser()
    parser.add_option("-d", "--dir", help="A directory containing nearly identical images", default=".")
    parser.add_option("-m", "--mid", help="The Middle image for the panorama", default=0, type = "int")
    parser.add_option("-p", "--pick", help="Do you want to match your own Features? Y or N", default="N")
    options, remain = parser.parse_args()
    print "Use the esc key to escape any imaged displayed"
    if options.pick == "y" or options.pick == "Y":
        pick = True
    else:
        pick = False
    
    pan = Panorama(options.dir, int(options.mid), pick)
    pan.createPanorama()
    
        