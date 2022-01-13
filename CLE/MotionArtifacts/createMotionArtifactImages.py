import matplotlib.pyplot as plt
import scipy.signal
import cv2
from auxfiles import circularMask
from inout import MKTreader, CLEdB
import numpy as np

# Extend the image in polar space
def extendPolar(inpt, radius=276):
    M = radius
    pol = cv2.linearPolar(inpt, center= (round(inpt.shape[0]/2),round(inpt.shape[1]/2)),maxRadius=M, flags=cv2.WARP_FILL_OUTLIERS)

    pol2 = np.hstack([pol, cv2.flip(pol, 1)])
    pol2 = cv2.resize(pol2, dsize=(inpt.shape[0]*2,inpt.shape[1]*2))
    extd= cv2.linearPolar(pol2, center= (round(inpt.shape[0]),round(inpt.shape[1])), maxRadius=2*M, flags=(cv2.WARP_FILL_OUTLIERS | cv2.WARP_INVERSE_MAP))
    return extd,pol2,pol


# Pick a random image from the database
def pickRandomImages(DB, whereclause='', count=10):
    query = 'SELECT CLEframes.id FROM CLEframes LEFT JOIN CLEsequences on CLEsequences.id == CLEframes.sequenceID '+whereclause+' ORDER BY RANDOM() LIMIT '+str(count)
    DB.c.execute(query)
    images=np.reshape(np.asarray(DB.c.fetchall()),(-1))
    images.sort()
    arr=(DB.getFileBatchFromFrameList(images))
    batch=MKTreader.MKTreader.readImageBatch(arr)

    return batch, images

DB = CLEdB.CLEdB()

# First, pick a random image
whereclause = 'WHERE sequenceId <= 116 AND motionArtifactClass<2 AND gaussianNoiseClass<2 AND cellStructure==0';

randomImage,id = pickRandomImages(DB, whereclause, count=1)

randomImage = randomImage[0, :,:]

# scale (percentile-scale) to uint8 data type
mask=circularMask.circularMask(576,576,576).mask
scaledImage = (MKTreader.MKTreader.scaleImageUINT8(None, randomImage, mask))


# Length of the overall acceleration is random between [60,576]
accelerationLength = np.int16(np.random.rand(1)*506)[0]+60


# First part of the acceleration is random between [0.5-1]*accelerationLength
accelerationPart1Length = np.int16((np.random.rand(1)*0.5+0.5)*accelerationLength)[0]

maxAcceleration= 2

# Motion process:
acceleration = np.random.randn(2)*maxAcceleration
acceleration = np.append(acceleration, -np.trapz(acceleration))
acceleration = np.append(acceleration,0)

# Upsample acceleration
accelerationUp = scipy.signal.resample(acceleration,accelerationPart1Length)

# only take acceleration up to first zero crossing
firstZC = np.where(np.abs(np.diff(np.sign(accelerationUp)))==2)[0][0]

# Add zero
accelerationPart1 = np.append(accelerationUp[0:firstZC+1],0)


# Part 2 is inverse, but scaled differently
accelerationPart2Length = accelerationLength - len(accelerationPart1)

accelerationPart2 = scipy.signal.resample(-accelerationPart1,accelerationPart2Length)

# normalize integral
accelerationPart2 = accelerationPart2 / np.trapz(accelerationPart2) * -1 * np.trapz(accelerationPart1)

acceleration = np.append(accelerationPart1, accelerationPart2[-1:0:-1])

velocity = scipy.integrate.cumtrapz(acceleration)
positionOffset = scipy.integrate.cumtrapz(velocity)

maxAngularVelocity = 2*np.pi
angularVelocity = np.ones((accelerationLength-3))*np.random.rand(1)*maxAngularVelocity

maxAngularAcceleration = 0.001
angularAcceleration = np.ones((accelerationLength-2))* np.random.rand(1)*maxAngularAcceleration




# convert image to grayscale
img=cv2.cvtColor(scaledImage, cv2.COLOR_GRAY2RGB)
mask=circularMask.circularMask(576,img.shape[0],img.shape[1])
for dim in range(3):
    img[:,:,dim]=img[:,:,dim]*mask.mask
extendedImage, pol, pol2=extendPolar(img)


angle = angularVelocity + scipy.integrate.cumtrapz(angularAcceleration)


maxOffset = (np.random.rand(1)[0]*2+0.5)*140
positionOffset= np.abs(positionOffset) / np.max(np.abs(positionOffset)) * maxOffset

coordinate_x = (positionOffset * np.sin(angle))
coordinate_y = (positionOffset * np.cos(angle))

# artifact start coordinate is the y coordinate where the artifact starts
artifactStartYCoordinate = np.int16(np.random.rand(1)[0] * (576 - len(coordinate_x)))

# prepare arrays
motionArtifactImage = np.zeros(img.shape, np.uint8)
cxt=np.zeros(580)
cyt=np.zeros(580)
cy=0
cx=0

print('Coordinate x len',len(coordinate_x))

if (np.min(coordinate_x) < -100):
    coordinate_x=coordinate_x/np.min(coordinate_x)*-80.0

if (np.max(coordinate_x) > 100):
    coordinate_x=coordinate_x/np.max(coordinate_x)*80.0

if (np.max(coordinate_y) > 80):
    coordinate_y=coordinate_y/np.max(coordinate_y)*80.0

# coveredImage is the image
coveredImage = np.copy(extendedImage)


# Generate motion artifact image
for y in range(img.shape[0]):
    if (y>artifactStartYCoordinate):
        if (y-artifactStartYCoordinate<len(coordinate_x)):
            cx = coordinate_x[y - artifactStartYCoordinate]
            cy = coordinate_y[y - artifactStartYCoordinate]

    # OpenCV getRectSubPix is used here - it provides an interpolated rectangle from the original image
    motionArtifactImage[y, :, :] = np.reshape(cv2.getRectSubPix(extendedImage, patchSize=(576, 1), center=(576 + cx, 288 + cy + y)), (576, 3))

    cy = np.int16(cy)
    cx = np.int16(cx)
    coveredImage[288 + cy + y, 288 + cx:576 + 288 + cx, 0] *= (np.uint8(1) - mask.mask[y, :])
    cyt[y] = cy
    cxt[y] = cx

# Mask out motion artifact image
for k in range(3):
    motionArtifactImage[:, :, k] = motionArtifactImage[:, :, k] * mask.mask


coveredImage = cv2.circle(coveredImage, (580, 576), radius=288, color=(255, 0, 0), thickness=2)
cx=0
cy=0

# Draw covered Image

for y in np.arange(0,img.shape[0],10):
    if (y>artifactStartYCoordinate):
        if (y-artifactStartYCoordinate<len(coordinate_x)):
            cx = coordinate_x[y - artifactStartYCoordinate]
            cy = coordinate_y[y - artifactStartYCoordinate]
    coveredImage = cv2.circle(coveredImage, (576 + np.int16(cx), 288 + np.int16(cy + y)), radius=4, color=(0, 0, 255), thickness=2)


# Plot and save motion profile
plt.figure()
plt.subplot(3,1,1)
plt.plot(cxt, label='X offset')
plt.plot(cyt, label='Y offset')
plt.legend()
plt.subplot(3,1,2)
plt.plot(np.hstack((np.zeros(artifactStartYCoordinate), angularVelocity, np.zeros(576 - len(coordinate_y) - artifactStartYCoordinate))), label='angular velocity')
plt.plot(np.hstack((np.zeros(artifactStartYCoordinate), velocity, np.zeros(576 - len(coordinate_y) - artifactStartYCoordinate))), label='abs velocity')
plt.legend()
plt.subplot(3,1,3)
plt.plot(np.hstack((np.zeros(artifactStartYCoordinate), angularAcceleration, np.zeros(576 - len(coordinate_y) - artifactStartYCoordinate))), label='angular acceleration')
plt.plot(np.hstack((np.zeros(artifactStartYCoordinate), np.append(accelerationPart1, accelerationPart2[-1:0:-1]), np.zeros(576 - len(coordinate_y) - artifactStartYCoordinate))), label='abs acceleration')
plt.legend()

# save some intermediate results and some images
num='28'
cv2.imwrite('original'+num+'.jpg',img)
cv2.imwrite('elp' + num +'.jpg', extendedImage)
cv2.imwrite('pol'+num+'.jpg',pol)
cv2.imwrite('pol2'+num+'.jpg',pol2)
cv2.imwrite('coveredImage' + num +'.jpg', coveredImage)
cv2.imwrite('fakeMotion' + num +'.jpg', motionArtifactImage)
plt.savefig('motionProfile'+num+'.pdf')
plt.figure()
