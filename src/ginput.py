'''
Created on Jan 25, 2011

@author: bseastwo
'''

import cv
import numpy

class Grab(object):
    '''
    Grabs mouse click positions from cv windows, like matlab's ginput.
    The set of positions, as an N x 2 array, can be retrieved with 
    getPoints().  Each row in the array is a point (x, y).
    '''
    def __init__(self, window, maxpts=0):
        '''
        Create a new point grabbing object bound to a cv window (by name).
        '''
        self.clear(maxpts)
        cv.SetMouseCallback(window, self.callback)
        self.currentPosition = (0, 0)
        
    def callback(self, event, x, y, flags, param):
        '''
        The callback function that stores click locations.
        '''
        self.currentPosition = (x, y)
        if event == cv.CV_EVENT_LBUTTONDOWN:
            if self.maxpts == 0 or self.maxpts > self.points.shape[0]:
                self.points = numpy.vstack( (self.points, numpy.array([(x, y)])) ) 
#                print self.points
            
    def getPoints(self):
        '''
        Returns the N x 2 set of points that this grabber has collected.
        '''
        return self.points
    
    def getCurrentPosition(self):
        '''
        Returns the current position of the mouse.
        '''
        return self.currentPosition
    
    def clear(self, maxpts=0):
        '''
        Clears the point set.  Optionally, maxpts sets the maximum number of
        points that will be recorded.
        '''
        self.points = numpy.zeros((0, 2), dtype=numpy.uint16)
        self.maxpts = maxpts
    
def testGrab():
    
    cv.NamedWindow("grab", cv.CV_WINDOW_NORMAL)
    grab = Grab("grab", 4)
    
    key = None
    while key != 27:
        # grab frame
        cvimg = cv.LoadImage("../pic.JPG")
          
        # draw points
        pts = grab.getPoints()
        for pxy in pts:
            cv.Rectangle(cvimg, (pxy[0]-5, pxy[1]-5), (pxy[0]+5, pxy[1]+5), (0, 0, 255))
        
        # display
        cv.ShowImage("grab", cvimg)
            
        # handle keys
        key = cv.WaitKey(100)
        if key >= 0 and chr(key) == 'c':
            grab.clear(4)
            
        
if __name__ == "__main__":
    testGrab()
    
        