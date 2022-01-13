#!/usr/bin/env python

"""MKTreader.py: Read MKT (Cellvizio) raw files."""
__author__ = "Marc Aubreville"
__license__ = "GPL"
__version__ = "0.0.1"

# MKT is the binary file format for Cellvizio files. The container format contains
# 16 bit raw image files that can be extracted using this module.

import numpy as np
import struct
import cv2
from os import stat
from auxfiles import circularMask
from matplotlib import pyplot

# fileinfo is a class for basic information about an opened file

class fileinfo:
    offset = 16 # fixed: 16 byte header
    gapBetweenImages = 32 # fixed: 32 byte gap between images
    size = 0 # size of file
    width = 0 # width of images
    height = 0 # height of images
    nImages = 0 # number of images
    circMask = 0; # circular mask (round shape)

###########################################################
# MKTreader: class for handling MKT files
###########################################################
# use: MKTreader('myfile.mkt', verbose)
#         verbose: 1 if you want to have some debug output
#
#  Returns an object with the following fields:
#    fileName:   fileName of the MKT fileName
#    fps:        Frames per Second of MKT file
#    fi:         fileInfo structure (see above)
#
###########################################################

class MKTreader:
    fileName = ''
    fileHandle = 0
    verbose=1
    fi = fileinfo()

    ###########################################################
    ## Constructor
    ###########################################################
    def __init__(self,filename, verbose=1):
       self.fileName = filename;
       self.fileHandle = open(filename, 'rb');
       self.verbose = verbose
       self.fileHandle.seek(5) # we find the FPS at position 05
       fFPSByte = self.fileHandle.read(4)
       self.fps = struct.unpack('>f', fFPSByte)[0]

       if (self.verbose):
           print('FPS: '+str(self.fps))

       self.fileHandle.seek(10) # we find the image size at position 10
       fSizeByte = self.fileHandle.read(4)
       self.fi.size = int.from_bytes(fSizeByte, byteorder='big', signed=True)
       self.fi.nImages=1000

       if (self.verbose):
           print("Image size: "+str(self.fi.size)+ " bytes")

       self.fi.width = 576
       if ((self.fi.size/(2*self.fi.width))%2!=0):
            self.fi.width=512
            self.fi.height=int(self.fi.size/(2*self.fi.width))
       else:
            self.fi.height=int(self.fi.size/(2*self.fi.width))

       if (self.verbose):
            print("Resolution is " +str(self.fi.width) + " x " + str(self.fi.height))

       self.filestats = stat(self.fileName)
       self.fi.nImages = int((self.filestats.st_size-self.fi.offset) / (self.fi.size+self.fi.gapBetweenImages))

       if (self.verbose):
           print("Number of images "+str(self.fi.nImages))

       # generate circular mask for this file
       self.circMask = circularMask.circularMask(self.fi.width,self.fi.height, self.fi.width-2).mask

    ###########################################################
    ## readImage: Read image at a certain position in the file
    ##   returns an image of type int16 in a numpy masked array
    ###########################################################

    def readImage(self, position=0):

       self.fileHandle.seek(self.fi.offset + self.fi.size*position + self.fi.gapBetweenImages*position)

       image = np.fromfile(self.fileHandle, dtype=np.int16, count=int(self.fi.size/2))

       image=np.reshape(image, newshape=(self.fi.height, self.fi.width))

       image = np.ma.masked_array(image, 1-self.circMask) # apply mask to image
       return image

    #######################################################################
    ## readImageUINT8: Read image at a certain position in the file
    ##    This is essentially a wrapper for readImage and scaleImageUINT8.
    #######################################################################

    def readImageUINT8(self, position=0):
       # read image and scale to uint8 [0;255] format
       image=self.readImage(position)

       image = self.scaleImageUINT8(image)
       return image

    #######################################################################
    ## readImageUINT16: Read image at a certain position in the file
    ##    This is the primary function for reading images from an MKT file
    #######################################################################

    def readImageUINT16(self, position=0):

       self.fileHandle.seek(self.fi.offset + self.fi.size*position + self.fi.gapBetweenImages*position)

       image = np.fromfile(self.fileHandle, dtype=np.int16, count=int(self.fi.size/2))
       if (image.shape[0] != int(self.fi.size/2)):
           print('Error occured when reading from MKT file. Read '+str(image.shape[0])+
                 ', requested '+str(self.fi.size/2)+' 16 bit values')
           print('Seeked position '+str(self.fi.offset + self.fi.size*position + self.fi.gapBetweenImages*position))
           print('File size is '+str(self.filestats.st_size))
       image = np.uint16(image+32768)
       image=np.reshape(image, newshape=(self.fi.height, self.fi.width))
       return image




    #######################################################################
    ## scaleImageUINT8: Scale a CLE image to uint8 data format
    ##    Scale the image according to the 0.5% and 99.5% percentiles
    ##    within the circular mask in the middle. Returns an uint8 image.
    #######################################################################

    def scaleImageUINT8(self, image, mask = None):
       # read image and scale to uint8 [0;255] format

       if (mask is None):
           mask = self.circMask

       maskedImage = image[mask]

       cmin,cmax = np.percentile(maskedImage,0.5), np.percentile(maskedImage,99.5)
       if (cmax>5000):
           cmax=5000
       dyn=cmax-cmin

       # compress
       compr=255/dyn
       image = image-cmin
       image = image*compr

       # limit to 0
       image = np.clip(np.round(image),0,255)
       image=np.uint8(image)

       return image

    class fileBatch:
        filename = ''
        frames = []

    #######################################################################
    ## readImageBatch(fileBatches)
    ##       read images in batches. To be used like:
    ##          keys=[77885, 71588, 73583, 68259]
    ##          arr=(CLEdB.getFileBatchFromFrameList(keys))
    ##          batch=MKTreader.MKTreader.readImageBatch(arr)
    ##       This script is very handy when reading a lot of images. They
    ##       are stored into a single int16 type array of size <N,Y,X>
    ##       where Y is the y-size, X is the x-size and N is the number of images
    ##
    ##       NOTE: To reduce complexity, two sortings are used:
    ##             First, the keys are sorted. Then, the sequences are sorted.
    ##             So in order to have the same order, make sure that you fill
    ##             pre-sorted keys.
    #######################################################################

    def readImageBatch(fileBatches, target_size_x=576, target_size_y=576):

         #preallocate
         nImages=0
         for fBatch in fileBatches:
             nImages += len(fBatch.frames)

         images = np.zeros((nImages,target_size_x,target_size_y),np.int16)
         nImage=0
         for fBatch in fileBatches:
               with open(fBatch.filename, 'rb') as fileHandle:

                    fileHandle.seek(10) # we find the image size at position 10
                    fSizeByte = fileHandle.read(4)
                    size = int.from_bytes(fSizeByte, byteorder='big', signed=True)
                    width = 576
                    if ((size/(2*width))%2!=0):
                            width=512
                            height=int(size/(2*width))
                    else:
                            height=int(size/(2*width))
                    cropx = np.int16(np.floor((width-target_size_x)/2))
                    cropy = np.int16(np.floor((height-target_size_y)/2))
                        

                    for position in fBatch.frames:
                        fileHandle.seek(16 + size*position + 32*position)


                        image = np.fromfile(fileHandle, dtype=np.int16, count=int(size/2))

                        image=np.reshape(image, newshape=(height, width))

#                        if ((width<=514) and (target_size_x==576)) or: # resize to 576x576
#                        image = cv2.resize(image, dsize=(target_size_x, target_size_y))

                        if (width==512): # resize to 576x576
                            image = cv2.resize(image, dsize=(target_size_x, target_size_y))
                            
                        if (cropy>0) or (cropx>0):
                            images[nImage,:,:]=image[cropy:height-cropy,cropx:width-cropx]
                        else:
                            images[nImage,:,:]=image

                        nImage += 1
                    fileHandle.close()


         return images
