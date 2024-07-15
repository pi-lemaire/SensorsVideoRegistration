# Python implementation of the following paper:
# Lemaire, Pierre, et al. "Registering unmanned aerial vehicle videos in the long term." Sensors 21.2 (2021): 513.



import numpy as np
import cv2



# functions below are to manipulate and translate to and from homography matrices

def applyScalingToPerspectiveTransformMat(mat, scaleFactor):
    # perspective transforms are linked to a resolution
    # the ratio is the ratio we want to apply to mat (src img -> dst img)
    scaledMat = mat.copy()
    scaledMat[0][2] *= scaleFactor
    scaledMat[1][2] *= scaleFactor
    scaledMat[2][0] *= 1./scaleFactor
    scaledMat[2][1] *= 1./scaleFactor
    return scaledMat


def extractCornersFromHomography(H, imShape):
    # given an image size, provide the relative displacements of corners after applying homography H
    w = imShape[1]
    h = imShape[0]
    CornerPts = np.array([[0., 0.], [w, 0.], [w, h], [0., h]], dtype=np.float32)
    CornersRelativePositions = cv2.perspectiveTransform(CornerPts.reshape(-1, 1, 2), H).squeeze()
    CornersRelativePositions -= CornerPts    
    return CornersRelativePositions


def generateHomographyFromCorners(imShape, cornersRelativePos):
    # given an image size, compute the corresponding homography matrix from corners displacement
    w = imShape[1]
    h = imShape[0]
    CornerPts = np.array([[0., 0.], [w, 0.], [w, h], [0., h]], dtype=np.float32)
    CornerPtsOutput = cornersRelativePos + CornerPts
    H = cv2.getPerspectiveTransform(CornerPts, CornerPtsOutput)
    return H







# this class is designed to combine homographies, in order to compute homography trajectories
class TrajectoryHomographyFromCorners:
    def __init__(self, size=None, initial_h=None):
        self.CornersRelativePositions = np.array([[0., 0.] for _ in range(4)], dtype=np.float64)
        self.HMat = np.eye(3, 3, dtype=np.float64)

        if size is not None:
            self.imSize = size
        else:
            self.imSize = (0, 0)

        if initial_h is not None:
            self.multiplyByL(initial_h)
            
    def multiplyByL(self, HMatrix, coeff=1.):
        mulCornerPositions = extractCornersFromHomography(HMatrix, self.imSize)
        
        mulCornerPositions = mulCornerPositions * coeff

        mulHomography = generateHomographyFromCorners(self.imSize, mulCornerPositions)
        self.HMat = np.matmul(mulHomography, self.HMat)

        self.CornersRelativePositions = extractCornersFromHomography(self.HMat, self.imSize)
        
    def multiplyByR(self, HMatrix, coeff=1.):
        mulCornerPositions = extractCornersFromHomography(HMatrix, self.imSize)
        
        mulCornerPositions = mulCornerPositions * coeff

        mulHomography = generateHomographyFromCorners(self.imSize, mulCornerPositions)
        self.HMat = np.matmul(self.HMat, mulHomography)

        self.CornersRelativePositions = extractCornersFromHomography(self.HMat, self.imSize)

    def getResultingMatrix(self):
        return self.HMat.copy()

    def getResultingInvertedMatrix(self):
        return np.linalg.inv(self.HMat)
    
    def setInitialHomography(self, initial_h):
        self.HMat = initial_h

    def setImageSize(self, size):
        self.imSize = size

    def copyFrom(self, other):
        self.CornersRelativePositions = other.CornersRelativePositions.copy()
        self.imSize = other.imSize
        self.HMat = other.HMat.copy()

    def clone(self):
        cloned_instance = TrajectoryHomographyFromCorners()
        cloned_instance.copyFrom(self)
        return cloned_instance


    
    
# The following class is the core of the method. It computes trajectories from comparing images
# one is the short-term trajectory (registering an incoming frame to the previous one)
# the issue with doing just this is that just inverting it leads to drifting, especially in the presence of
# mobile objects within the scene. So we correct it by applying a filtered registration to the initial view

# here, image to image correspondences are performed with a very basic punctual optical flow approach
# thus, this can work using only CPU on a relatively basic computer. This keypoint matching may be replaced
# by any method depending on the situation and/or the configuration



_VideoStab_flag_reset = 0x01
_VideoStab_flag_badly_registered = 0x02
_VideoStab_flag_too_much_foreground = 0x04




# some hyper parameters - actually the method is not too sensitive to most of them

_StabilizerHomographyTrajCorrected_default_workingImgWidth = 550           # how do we resize the image before we process it

_StabilizerHomographyTrajCorrected_default_FF_gFTTPointsNumber = 200       # number of feature points that we match from frame to frame
_StabilizerHomographyTrajCorrected_default_FF_gFTTqualityLevel = 0.02      # qualityLevel in the goodFeaturesToTrack method
_StabilizerHomographyTrajCorrected_default_FF_gFTTminDistance = 20.        # minDistance in the goodFeaturesToTrack method
_StabilizerHomographyTrajCorrected_default_FF_bestCandidatesNumber = 100   # minDistance in the goodFeaturesToTrack method
_StabilizerHomographyTrajCorrected_default_FF_boundaryMargins = 5.
_StabilizerHomographyTrajCorrected_default_FF_RansacReprojThreshold = 2.
_StabilizerHomographyTrajCorrected_default_FF_minPointsNumberForRegistration = 6

_StabilizerHomographyTrajCorrected_default_consec_correctionCoeff = .01

_StabilizerHomographyTrajCorrected_default_consec_gFTTPointsNumber = 200    # number of feature points that we match from frame to frame
_StabilizerHomographyTrajCorrected_default_consec_gFTTqualityLevel = 0.02   # qualityLevel in the goodFeaturesToTrack method
_StabilizerHomographyTrajCorrected_default_consec_gFTTminDistance = 20.     # minDistance in the goodFeaturesToTrack method
_StabilizerHomographyTrajCorrected_default_consec_minPointsNumberForStabilization = 20


_StabilizerHomographyTrajCorrected_default_maxErrorConsecForStabilization = 20.
_StabilizerHomographyTrajCorrected_default_maxErrorRegisterForStabilization = 30.
_StabilizerHomographyTrajCorrected_default_maxStabilizationFreezeLength = 500
_StabilizerHomographyTrajCorrected_default_allowFreeze = 0

_StabilizerHomographyTrajCorrected_default_kalmanProcessNoiseCovVar = 1e-6
_StabilizerHomographyTrajCorrected_default_kalmanMeasureNoiseCovVar = 5.
_StabilizerHomographyTrajCorrected_default_kalmanErrorCovPostVar = 0.01







class StabilizerHomographyTrajCorrected:
    def __init__(self):
        # just setting the default parameters
        self.resizeMaxWidth = _StabilizerHomographyTrajCorrected_default_workingImgWidth

        self.FF_gFTTPointsNumber = _StabilizerHomographyTrajCorrected_default_FF_gFTTPointsNumber
        self.FF_gFTTqualityLevel = _StabilizerHomographyTrajCorrected_default_FF_gFTTqualityLevel
        self.FF_gFTTminDistance = _StabilizerHomographyTrajCorrected_default_FF_gFTTminDistance
        self.FF_boundaryMargins = _StabilizerHomographyTrajCorrected_default_FF_boundaryMargins
        self.FF_bestCandidatesNumber = _StabilizerHomographyTrajCorrected_default_FF_bestCandidatesNumber
        self.FF_RansacReprojThreshold = _StabilizerHomographyTrajCorrected_default_FF_RansacReprojThreshold
        self.FF_minPointsNumberForRegistration = _StabilizerHomographyTrajCorrected_default_FF_minPointsNumberForRegistration

        self.correctionCoeff = _StabilizerHomographyTrajCorrected_default_consec_correctionCoeff

        self.consec_gFTTPointsNumber = _StabilizerHomographyTrajCorrected_default_consec_gFTTPointsNumber
        self.consec_gFTTqualityLevel = _StabilizerHomographyTrajCorrected_default_consec_gFTTqualityLevel
        self.consec_gFTTminDistance = _StabilizerHomographyTrajCorrected_default_consec_gFTTminDistance

        self.consec_minPointsNumberForStabilization = _StabilizerHomographyTrajCorrected_default_consec_minPointsNumberForStabilization

        self.maxErrorConsecForStabilization = _StabilizerHomographyTrajCorrected_default_maxErrorConsecForStabilization
        self.maxErrorRegisterForStabilization = _StabilizerHomographyTrajCorrected_default_maxErrorRegisterForStabilization
        self.maxStabilizationFreezeLength = _StabilizerHomographyTrajCorrected_default_maxStabilizationFreezeLength
        self.allowFreeze = _StabilizerHomographyTrajCorrected_default_allowFreeze

        self.kalmanProcessNoiseCovVar = _StabilizerHomographyTrajCorrected_default_kalmanProcessNoiseCovVar
        self.kalmanMeasureNoiseCovVar = _StabilizerHomographyTrajCorrected_default_kalmanMeasureNoiseCovVar
        self.kalmanErrorCovPostVar = _StabilizerHomographyTrajCorrected_default_kalmanErrorCovPostVar
    
        self.init()
    
    
    
    def init(self):
        self.firstFrame = None
        self.prevFrame = None
        self.resizeFactor = 0.3
        
        # optical flow definition
        self.FF_optFlow = cv2.SparsePyrLKOpticalFlow_create(
            winSize=(21, 21),
            maxLevel=1,
            crit=(cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 30, 0.01),
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW
        )
        
        self.Consec_optFlow = cv2.SparsePyrLKOpticalFlow_create()
        self.freezeCounter = 0
        self.badRegistrationsCounter = 0
        
        # initialization of the kalman filter
        # the kalman filter here is applied to the (x,y) coordinates of each 4 corners of the homography trajectory
        self.KF = cv2.KalmanFilter(16, 8, 0, cv2.CV_32F)
        
        # apparently, in the python implementation, we need to allocate explicitely all of the state variables....
        self.KF.statePre = np.zeros((16, 1), dtype=np.float32)
 
        # transition matrix allows us to handle velocity
        self.KF.transitionMatrix = np.eye(16, dtype=np.float32)
        
        # the measurement matrix however will store only the positions
        self.KF.measurementMatrix = np.zeros((8,16), dtype=np.float32)
        
        for i in range(8):
            # applying the filter to both position and speed of trajectory corner coordinates
            self.KF.transitionMatrix[i, 8 + i] = 1.
            self.KF.measurementMatrix[i, i] = 1.

        # filter parameters
        self.KF.processNoiseCov = np.identity(16, dtype=np.float32) * self.kalmanProcessNoiseCovVar
        self.KF.measurementNoiseCov = np.identity(8, dtype=np.float32) * self.kalmanMeasureNoiseCovVar
        self.KF.errorCovPost = np.identity(16, dtype=np.float32) * self.kalmanErrorCovPostVar
        
        
        self.KFMeasurement = np.zeros((8, 1), dtype=np.float32)

        self.currVidPos = 0

        
        
        
        
    def setReferenceFrame(self, f_frame):
        self.firstFrame = f_frame.copy()
        self.prevFrame = f_frame.copy()
        
        self.imSize = f_frame.shape

        self.firstFrameCorners = cv2.goodFeaturesToTrack(
            self.firstFrame, self.FF_gFTTPointsNumber, self.FF_gFTTqualityLevel, self.FF_gFTTminDistance
        )

        self.TrajAccumulated = TrajectoryHomographyFromCorners(self.firstFrame.shape)
        self.TrajConsecOnly = TrajectoryHomographyFromCorners(self.firstFrame.shape)


        

    def stabilize(self, im):
        warpMat = np.eye(3, dtype=np.float64)

        returnFlag = 0

        
        if im is None or im.size == 0:
            return returnFlag

        
        if self.firstFrame is None or self.firstFrame.size == 0:
            # initialization: setting the first frame to the desired size
            self.resizeFactor = min(float(self.resizeMaxWidth) / float(im.shape[1]), 1.0)

            FFrame = cv2.resize(im, None, fx=self.resizeFactor, fy=self.resizeFactor)
            if len(FFrame.shape) > 2:
                FFrame = cv2.cvtColor(FFrame, cv2.COLOR_BGR2GRAY)

            self.setReferenceFrame(FFrame)

            tmpTComputed = np.eye(3, dtype=np.float64)
            KFPred = np.zeros((8, 1), dtype=np.float32)
            KFEst  = np.zeros((8, 1), dtype=np.float32)
            KFMeas = np.zeros((8, 1), dtype=np.float32)

            self.currVidPos += 1

            return tmpTComputed, _VideoStab_flag_reset

        
        # we already have a first frame.
        # first compute the short-term motion compensation
        
        appliedTransform = np.eye(3, dtype=np.float32)
        correctToFirstFrame = np.eye(3, dtype=np.float32)
        currTConsec = np.eye(3, dtype=np.float32)
        KFPred8x1 = np.zeros((8, 1), dtype=np.float32)
        KFEst8x1 = np.zeros((8, 1), dtype=np.float32)


        cur_frame = cv2.resize(im, None, fx=self.resizeFactor, fy=self.resizeFactor)
        if len(cur_frame.shape) > 2:
            cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)



        currCorners = cv2.goodFeaturesToTrack(cur_frame, self.consec_gFTTPointsNumber, self.consec_gFTTqualityLevel, self.consec_gFTTminDistance);
        prevCorners, status, consecErrors = self.Consec_optFlow.calc(cur_frame, self.prevFrame, currCorners, None)


        currCorners = currCorners[status == 1]
        prevCorners = prevCorners[status == 1]
        consecErrors = consecErrors[status == 1]

        # using a robust estimator - here, LMEDS makes the assumption that at least half of the matched points are inliers
        currTConsec, inliers = cv2.findHomography(currCorners, prevCorners, cv2.LMEDS)

        
        # storing the current frame for the next iteration
        self.prevFrame = cur_frame.copy()
        
        
        # trajAccumulated and TrajConsec are the motion compensation values if there was no drifting element in currTConsec
        self.TrajAccumulated.multiplyByR(currTConsec)
        self.TrajConsecOnly.multiplyByR(currTConsec)
        
        
        # predict the next correction, first from the corners
        pred = self.KF.predict()
        for k in range(8):
            KFPred8x1[k] = pred[k]
            
        # then the corresponding matrix
        predicted3x3 = generateHomographyFromCorners(self.imSize, KFPred8x1.reshape((4,2)))
        
        
        # predict the correction to the first frame based on this
        TrajToFFPrioriPredicted = self.TrajAccumulated.clone()
        TrajToFFPrioriPredicted.multiplyByL(predicted3x3)
        
        # and calculate it to apply it to the current frame so that it's already aligned with the FF as much as possible
        invertedPrioriToFFMat = TrajToFFPrioriPredicted.getResultingInvertedMatrix()
        
        
        # set the tracked corners in the original reversed trajectory position (before the "priori" correction)
        queryCornersPrioriPositions = cv2.perspectiveTransform(self.firstFrameCorners, invertedPrioriToFFMat)
        

        # warp the image according to the theoretical position at which we expect the image to be
        # this is done here mostly because the optical flow tracking is not robust to heavy rotation or warping
        prioriWarpedImg = cv2.warpPerspective(cur_frame, TrajToFFPrioriPredicted.getResultingMatrix(), (self.imSize[1],self.imSize[0]))
        
        
        # then compute the optical flow and keep track of errors (we will use them later to sort them to keep only relevant points)
        queryCornersWarped = self.firstFrameCorners.copy()
        
        ffTrackedCorners, status, ffErrors = self.FF_optFlow.calc(self.firstFrame, prioriWarpedImg, self.firstFrameCorners, queryCornersWarped)
        
        # this is pretty simplified compared to the original algorithm (a lot more selection of the points according to their matching scores)
        ffCorners = self.firstFrameCorners[status==1]
        ffTrackedCorners = ffTrackedCorners[status==1]
        
        # the robust estimator here is the very efficient MAGSAC because we cannot suppose that most points were well registered here.
        correctToFirstFrame, _ = cv2.findHomography(ffTrackedCorners, ffCorners, cv2.USAC_MAGSAC)
        
        # apply it to the prediction
        TrajToFFPrioriMeasured = TrajToFFPrioriPredicted.clone()
        TrajToFFPrioriMeasured.multiplyByL(correctToFirstFrame)
        
        # remove the original trajectory component
        TrajToFFPrioriMeasured.multiplyByR(self.TrajAccumulated.getResultingInvertedMatrix())
        
        # now update the corner positions in the Kalman Filter
        actualizedTrajCorrectCornerPositions = extractCornersFromHomography(TrajToFFPrioriMeasured.getResultingMatrix(), self.imSize)
        
        atcp = actualizedTrajCorrectCornerPositions.reshape((8,1))
        for k in range(8):
            self.KFMeasurement[k] = atcp[k]
        
        # estimated is a smoothed version of what we have measured
        estimated = self.KF.correct(self.KFMeasurement)
        
        # revert to an homography
        for i in range(8):
            KFEst8x1[i] = estimated[i]
        estimated3x3 = generateHomographyFromCorners(self.imSize, KFEst8x1.reshape((4,2)))

        # Apply it to the trajectory
        TrajToFFEstimated = self.TrajAccumulated.clone()
        TrajToFFEstimated.multiplyByL(estimated3x3)

        # Finally, update the global trajectory with our estimation
        # the coefficient here is used for 2 reasons:
        # 1. in case of a lot of noise in the registration measurement,
        #    prevent the kalman filter from oscillating because of a strong impulse
        # 2. prevent the correction from diverging infinitvely,
        #    and risking a degraded case where 3 corners are colinear
        # Since the trajectory from origin is corrected by this, its effect is cumulative over time, so even a small
        # coefficient is useful to prevent the trajectory from diverging infinitely
        self.TrajAccumulated.multiplyByL(estimated3x3, self.correctionCoeff)

        # Fill in the transform matrix
        appliedTransform = TrajToFFEstimated.getResultingMatrix()

        # revert to the original image size
        warpMat = applyScalingToPerspectiveTransformMat(appliedTransform, 1.0 / self.resizeFactor)
        
        # UNCOMMENT THE LINE BELOW TO SEE THE SHORT-TERM TRAJECTORY INVERSION WITHOUT CORRECTING THE DRIFTING
        # DIFFERENCES ARE VISIBLE MOSTLY ON LENGTHY VIDEOS
        # warpMat = applyScalingToPerspectiveTransformMat(self.TrajConsecOnly.getResultingMatrix(), 1.0 / self.resizeFactor)

        self.currVidPos += 1

        return warpMat, returnFlag
    
    

if __name__ == '__main__':
    
    # load the video
    video = cv2.VideoCapture( 'example_videos/mocopo.mp4' )

    # initialize the stabilization
    VStab = StabilizerHomographyTrajCorrected()

    # loop through the video
    try:
        while True:
            # read the video
            ret, frame = video.read()
            if not ret:
                break

            # calculate the registration
            H, flag = VStab.stabilize(frame)

            # register accordingly to the output
            warpedF = cv2.warpPerspective(frame, H, (frame.shape[1],frame.shape[0]))
            
            cv2.imshow('Frame', warpedF)
        
            # Press Q on keyboard to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        video.release()