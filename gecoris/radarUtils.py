#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""radarUtils

Module containing radar utility functions

Copyright (C) 2020 by R.Czikhardt

Email: czikhardt.richard@gmail.com
Last edit: 28.7.2020

This file is part of GECORIS - Geodetic Corner Reflector (In)SAR Toolbox.

    GECORIS is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    GECORIS is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with GECORIS. If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
import math
import scipy.linalg
from gecoris import geoUtils, atmoUtils, s1Utils
speedOfLight = 299792458.0
from functions import ioUtilsPaolo as io
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


def xyz2t(xyzAll,metadata):
    """Return a (azimuthTime,rangeTime,satelliteVector)
    
    Inverse range-Doppler solution from ECEF coordinates to azimuth and range 
    time.
    
    input: numpy.array([x,y,z]), 'metadata' dict as read by 'readMetadata'.
    """
    maxiter = 10
    criter = 1e-10
    
    
    t0 = metadata['orbitFit']['t0']
    cX = metadata['orbitFit']['cX']
    cY = metadata['orbitFit']['cY']
    cZ = metadata['orbitFit']['cZ']
    cVX = metadata['orbitFit']['cVX']
    cVY = metadata['orbitFit']['cVY']
    cVZ = metadata['orbitFit']['cVZ']
    cAX = metadata['orbitFit']['cAX']
    cAY = metadata['orbitFit']['cAY']
    cAZ = metadata['orbitFit']['cAZ']
    
    # initial value for azimuth time (first in swath is sufficient):
    tAz = (metadata['centerAzimuth']-1)/metadata['PRF'] + metadata['azimuth0time'];
    # tAz_initial = tAz
    
    # loop through points:
    idx = 0
    if xyzAll.ndim < 2:
        xyzAll = np.nditer(xyzAll,flags = ['external_loop'], order = 'C')
        tA = np.empty(1)
        tR = np.empty(1)
        satVec = np.empty((1,3))
    else:
        # allocate:
        tA = np.empty((xyzAll.shape[0]))
        tR = np.empty((xyzAll.shape[0]))
        satVec = np.empty((xyzAll.shape[0],3))
    for xyz in xyzAll:
        i = 1
        #DEBUG
        # ITER = []
        # DIST = []
        # DOT = []
        #
        while i < maxiter:
            dx = xyz[0] - np.polynomial.chebyshev.chebval(tAz-t0,cX)
            dy = xyz[1] - np.polynomial.chebyshev.chebval(tAz-t0,cY)
            dz = xyz[2] - np.polynomial.chebyshev.chebval(tAz-t0,cZ)
            vx = np.polynomial.chebyshev.chebval(tAz-t0,cVX)
            vy = np.polynomial.chebyshev.chebval(tAz-t0,cVY)
            vz = np.polynomial.chebyshev.chebval(tAz-t0,cVZ)
            ax = np.polynomial.chebyshev.chebval(tAz-t0,cAX)
            ay = np.polynomial.chebyshev.chebval(tAz-t0,cAY)
            az = np.polynomial.chebyshev.chebval(tAz-t0,cAZ)
            # inverse range-Doppler solution:
            dt = -(vx*dx + vy*dy + vz*dz)/(ax*dx + ay*dy + az*dz - vx**2 -vy**2 -vz**2)        
            tAz = tAz + dt
            
            if np.abs(dt) < criter:
                break
            elif i >= maxiter:
                print("Warning, range-Doppler solution didn't converge!")
            # DEBUG
            # satVecP = np.array([np.polynomial.chebyshev.chebval(tAz-t0,cX),np.polynomial.chebyshev.chebval(tAz-t0,cY),np.polynomial.chebyshev.chebval(tAz-t0,cZ)])
            # dist = np.sqrt(np.sum(np.power(xyz - satVecP,2)))*1e-3
            #DEBUG
            # vxSat = np.polynomial.chebyshev.chebval(tAz-t0,cVX)
            # vySat = np.polynomial.chebyshev.chebval(tAz-t0,cVY)
            # vzSat = np.polynomial.chebyshev.chebval(tAz-t0,cVZ)
            # satVel = np.array([vxSat,vySat,vzSat])
            # dott = np.dot((satVecP-xyz)/np.linalg.norm((satVecP-xyz)),satVel/np.linalg.norm(satVel))
            # #
            # ITER.append(i)
            # DIST.append(dist)
            # DOT.append(np.abs(dott))
            #
            i += 1
        # compute corresponding range time and satellite position:
        xSat = np.polynomial.chebyshev.chebval(tAz-t0,cX)
        ySat = np.polynomial.chebyshev.chebval(tAz-t0,cY)
        zSat = np.polynomial.chebyshev.chebval(tAz-t0,cZ)
        satVec[idx] = np.array([xSat,ySat,zSat])
        # DEBUG
        # vxSat = np.polynomial.chebyshev.chebval(tAz-t0,cVX)
        # vySat = np.polynomial.chebyshev.chebval(tAz-t0,cVY)
        # vzSat = np.polynomial.chebyshev.chebval(tAz-t0,cVZ)
        # satVel = np.array([vxSat,vySat,vzSat])
        
#         DOTglob = []
#         DT = .01
#         tTest = np.linspace(tAz-DT,tAz+DT,2000)
#         for tt in tTest:
#             vxSat = np.polynomial.chebyshev.chebval(tt-t0,cVX)
#             vySat = np.polynomial.chebyshev.chebval(tt-t0,cVY)
#             vzSat = np.polynomial.chebyshev.chebval(tt-t0,cVZ)
#             satVel = np.array([vxSat,vySat,vzSat])
#             satVecP = np.array([np.polynomial.chebyshev.chebval(tt-t0,cX),np.polynomial.chebyshev.chebval(tt-t0,cY),np.polynomial.chebyshev.chebval(tt-t0,cZ)])
#             dott = np.dot((satVecP-xyz)/np.linalg.norm((satVecP-xyz)),satVel/np.linalg.norm(satVel))
#             DOTglob.append(np.abs(dott))
#         #  

        
#         print('START DEBUG (@radarUtils.py line 101)\n----------------------------------------------------------------')
#         print('xyz=' + str(xyz*1e-3) + ' [km]')
#         print('satVec=' + str(satVec[idx]*1e-3) + ' [km]')
#         print('tAz= ' + str(tAz) + ' [seconds of day]')
#         dt = np.abs(dt)*1e6
#         print('dt= ' + str(dt) + ' [seconds ]')
#         print('LOS distance @tAz+00= ' + str(np.sqrt(np.sum(np.power(xyz - satVec[idx],2)))*1e-3)+' [km]')
#         satVecP = np.array([np.polynomial.chebyshev.chebval(tAz+dt-t0,cX),np.polynomial.chebyshev.chebval(tAz+dt-t0,cY),np.polynomial.chebyshev.chebval(tAz+dt-t0,cZ)])
#         print('LOS distance @tAz+dt= ' + str(np.sqrt(np.sum(np.power(xyz - satVecP,2)))*1e-3)+' [km]')
#         satVecM = np.array([np.polynomial.chebyshev.chebval(tAz-dt-t0,cX),np.polynomial.chebyshev.chebval(tAz-dt-t0,cY),np.polynomial.chebyshev.chebval(tAz-dt-t0,cZ)])
#         print('LOS distance @tAz-dt= ' + str(np.sqrt(np.sum(np.power(xyz - satVecM,2)))*1e-3)+' [km]')
#         print('DIST[-1] - DIST[0] = ' + str((DIST[-1] - DIST[0])) + ' [km]')
#         print('tAz-tAz_initial = '+str(tAz-tAz_initial))
#         print('Range doppler solution converged'+ str( (tAz-tAz_initial)*metadata["PRF"]) + ' azimuth lines distant from the initial guess')
#         print('Dot product between geom range and sat velocity (satVec-xyz)*satVel = ' + str(np.dot((satVec[idx]-xyz),satVel)))
#         fig, dbgP = plt.subplots(1,1)
#         dbgP.semilogy(ITER,DIST)
#         fig, dbgB = plt.subplots(1,1)
#         dbgB.plot(ITER,DOT)
#         fig, dbgC = plt.subplots(1,1)
#         dbgC.plot(tTest,DOTglob)
#         dbgC.vlines(tAz,-0.0001,0.0001,color='g')
#         # dbgC.vlines(tAz_initial,-0.0001,0.0001,color='r')
#         print('-------------------------------------------------------------------------------------------------------')

        
        tR[idx] = np.sqrt(np.sum(np.power(xyz - satVec[idx],2)))/speedOfLight
        
        tA[idx] = tAz
        idx += 1
    
    return tA,tR,satVec

def xyz2satvec(xyz,metadata):
    """Return a (9,1) array with (Xs,Ys,Zs,VXs,VYs,VZs,AXs,AYs,AZs)
    where X are coordinates, VX are velocities and AX are acceleration of 
    the satellite corresponding to position of target XYZ using acquisition
    metadata   
    """
    maxiter = 10
    criter = 1e-12    
    t0 = metadata['orbitFit']['t0']
    cX = metadata['orbitFit']['cX']
    cY = metadata['orbitFit']['cY']
    cZ = metadata['orbitFit']['cZ']
    cVX = metadata['orbitFit']['cVX']
    cVY = metadata['orbitFit']['cVY']
    cVZ = metadata['orbitFit']['cVZ']
    cAX = metadata['orbitFit']['cAX']
    cAY = metadata['orbitFit']['cAY']
    cAZ = metadata['orbitFit']['cAZ']
    # initial value for azimuth time (first in swath is sufficient):
    tAz = (metadata['centerAzimuth']-1)/metadata['PRF'] + metadata['azimuth0time'];
    # iterative solution:
    i = 1
    while i < maxiter:
        dx = xyz[0] - np.polynomial.chebyshev.chebval(tAz-t0,cX)
        dy = xyz[1] - np.polynomial.chebyshev.chebval(tAz-t0,cY)
        dz = xyz[2] - np.polynomial.chebyshev.chebval(tAz-t0,cZ)
        vx = np.polynomial.chebyshev.chebval(tAz-t0,cVX)
        vy = np.polynomial.chebyshev.chebval(tAz-t0,cVY)
        vz = np.polynomial.chebyshev.chebval(tAz-t0,cVZ)
        ax = np.polynomial.chebyshev.chebval(tAz-t0,cAX)
        ay = np.polynomial.chebyshev.chebval(tAz-t0,cAY)
        az = np.polynomial.chebyshev.chebval(tAz-t0,cAZ)
        # inverse range-Doppler solution:
        dt = -(vx*dx + vy*dy + vz*dz)/(ax*dx + ay*dy + az*dz - vx**2 -vy**2 -vz**2)        
        tAz = tAz + dt
        
        if np.abs(dt) < criter:
            break
        elif i >= maxiter:
            print("Warning, range-Doppler solution didn't converge!")
        
        i += 1
    # compute corresponding range time and satellite position:
    xSat = np.polynomial.chebyshev.chebval(tAz-t0,cX)
    ySat = np.polynomial.chebyshev.chebval(tAz-t0,cY)
    zSat = np.polynomial.chebyshev.chebval(tAz-t0,cZ)
    vx = np.polynomial.chebyshev.chebval(tAz-t0,cVX)
    vy = np.polynomial.chebyshev.chebval(tAz-t0,cVY)
    vz = np.polynomial.chebyshev.chebval(tAz-t0,cVZ)
    ax = np.polynomial.chebyshev.chebval(tAz-t0,cAX)
    ay = np.polynomial.chebyshev.chebval(tAz-t0,cAY)
    az = np.polynomial.chebyshev.chebval(tAz-t0,cAZ)
    satVec = np.array([xSat,ySat,zSat,vx,vy,vz,ax,ay,az])
    return satVec


def radar2xyz(Azimuth,Range,elev,metadata, returnSat = False):
    """Return a XYZ cartesian geo-coordinates
    
    Transformation from radar coordinates to geo coordinates.
    
    input: Azimuth & Range in [pix], 'metadata' dict as read by 'readMetadata'.
    """
    
    a = 6378137.0
    b = 6356752.3141                       # WGS84 ellipsoid
    e2 = (a**2-b**2)/a**2
    maxiter = 10                             # Max number of iterations
    criter = 1e-6                         # Stop criterion for iterations
    
    # initial values for scene center:
    centerLat = metadata['centerLat']*np.pi/180    
    centerLon = metadata['centerLon']*np.pi/180
    centerN = a/np.sqrt(1-e2*(np.sin(centerLat)**2))
    centerX = (centerN+np.mean(elev))*np.cos(centerLat)*np.cos(centerLon);
    centerY = (centerN+np.mean(elev))*np.cos(centerLat)*np.sin(centerLon);
    centerZ = (centerN+np.mean(elev)-e2*centerN)*np.sin(centerLat);
    
    tAzimuth, tRange = radar2time(Azimuth,Range,metadata)
    
    satVec,satVel = geoUtils.orbitVal(metadata['orbitFit'], tAzimuth)
    
    # allocate:
    if satVec.ndim < 2:
        satVec = [satVec]
        satVel = [satVel]
        elev = [elev]
        tRange = [tRange]
        xyzAll = np.empty((1,3))
    else:
        xyzAll = np.empty((satVec.T.shape))
        satVec = satVec.T
        satVel = satVel.T
    
    # loop through points:
    idx = 0
    for sat,vel,h,tR in zip(satVec,satVel,elev,tRange):
        x = centerX.copy()
        y = centerY.copy()
        z = centerZ.copy()
        i = 1
        while i < maxiter:
            dx = x - sat[0]
            dy = y - sat[1]
            dz = z - sat[2]
            eq = -1*np.array([vel[0]*dx+vel[1]*dy+vel[2]*dz,
                              dx**2+dy**2+dz**2 - (speedOfLight*tR)**2,
                              (x**2+y**2)/((a+h)**2)+(z/(b+h))**2-1])
            design = np.array([vel,
                                2*np.array([dx,dy,dz]),
                                np.array([2*x/((a+h)**2),2*y/((a+h)**2),2*z/((b+h)**2)])])
            
            sol = np.linalg.solve(design,eq)
            x += sol[0]
            y += sol[1]
            z += sol[2]
            
            if np.any(np.abs(sol) > criter):
                if i < maxiter:
                    i += 1
                    continue
                else:
                    print('radar2xyz did not converge after 10 iterations!')
            else:
                break
        #xyz = np.array([x,y,z])
        xyzAll[idx] = np.array([x,y,z])
        idx += 1
    if returnSat:
        return xyzAll, satVec
    else:
        return xyzAll


def time2radar(tAzimuth,tRange,metadata):
    """Return a (azimuth,range) radar coordinates
    
    Transformation from time to pixel radar coordinates.
    
    input: tAzimuth & tRange in [s], 'metadata' dict as read by 'readMetadata'.
    """

    if metadata['nBursts'] < 2:
        # zero-based indexing
        Azimuth = metadata['PRF']*(tAzimuth - metadata['azimuth0time'])
    else:
        burstAz = metadata['burstInfo']['firstAzTime']
        # modify by "usable" samples:
        burstAzValid = burstAz + 25/metadata['PRF']
        burstStep = metadata['burstInfo']['nAzBurst']
        burstIdx = np.nonzero(tAzimuth > burstAzValid)[0][-1]
        Azimuth = (metadata['PRF']*(tAzimuth - burstAz[burstIdx]) 
                    + burstStep*burstIdx)
    
    Range = metadata['RSR']*(tRange - metadata['range0time'])
    return Azimuth, Range


def radar2time(Azimuth,Range,metadata):
    """Return a (azimuth,range) time radar coordinates from pixels
    
    Transformation from pixel to time radar coordinates.
    
    input: Azimuth & Range in [pix], 'metadata' dict as read by 'readMetadata'.
    """
   
    if metadata['nBursts'] < 2:
        # zero-based indexing
        tAzimuth = (Azimuth)/metadata['PRF'] + metadata['azimuth0time']
    else:
        burstAz = metadata['burstInfo']['firstAzTime']
        burstStep = metadata['burstInfo']['nAzBurst']    
        burstIdx = int(np.floor(np.mean(Azimuth)/burstStep))
        tAzimuth = ((Azimuth - burstStep*burstIdx)/metadata['PRF'] 
                    + burstAz[burstIdx])
    
    tRange = Range/metadata['RSR'] + metadata['range0time']  
    return tAzimuth, tRange


def beta2RCSdB(beta0,metadata):
    RCS = beta0*metadata['rangeResolution']*metadata['azimuthResolution']
    return 10*np.log10(RCS)
    

def RCSdB2beta(RCSdB,metadata):
    RCS = np.power(10,RCSdB/10)
    return RCS/metadata['rangeResolution']/metadata['azimuthResolution']


def amp2RCSdB(amp,metadata):
    # NOTE: use only if already radiometrically calibrated!!!
    beta0 = np.power(amp,2)
    RCS = beta0*metadata['rangeResolution']*metadata['azimuthResolution']
    return 10*np.log10(RCS)
    

def RCSdB2amp(RCSdB,metadata):
    RCS = np.power(10,RCSdB/10)
    beta0 = RCS/metadata['rangeResolution']/metadata['azimuthResolution']
    return np.sqrt(beta0)


def ph2mm(ph,wavelength = 0.05546576):
    # HARDCODED Sentinel-1 C-band by default   
    mm = ph/(-4*np.pi)*wavelength*1e3
    return mm


def mm2ph(mm,wavelength = 0.05546576):
    # HARDCODED Sentinel-1 C-band by default   
    ph = mm/1e3/wavelength*-4*np.pi
    return ph


def oversample(slc, up_factor,metadata):
    """Return slc image oversampled by up_factor
    
    FFT-based oversampling of SLC data by zero-padding.
    
    input: slc data grid as np.array
    """    
    # parse useful metadata
    acqDate = metadata["acqDate"]
    rangeSpacing = metadata["rangeSpacing"]
    azimuthSpacing = metadata["azimuthSpacing"]

    
    # coordinates of the crop's origin
    firstR = slc.range.data[0]
    firstAz = slc.azimuth.data[0]
    
    
    slc = np.array(slc[:,:])
    [rows, cols] = slc.shape


    slc = np.fft.fftshift(np.fft.fft2(slc))
    min_row = math.ceil(rows * up_factor / 2 - rows / 2)
    max_row = min_row + rows
    min_col = math.ceil(cols * up_factor / 2 - cols / 2)
    max_col = min_col + cols
    
    slc_padding = np.zeros((rows * up_factor, cols * up_factor), dtype=np.complex64)    
    slc_padding[min_row:max_row,min_col:max_col] = slc
    slc = np.fft.fftshift(slc_padding)
    
    # define the axes: to the first pixel and line is added the grid of the oversampled image
    # The range and Az Spacing are considered to express the image in meters
    xaxis = firstR*rangeSpacing + np.linspace(0,(cols-1)*rangeSpacing,up_factor*cols)
    yaxis = firstAz*azimuthSpacing + np.linspace(0,(rows-1)*azimuthSpacing,up_factor*rows)

    axes = (xaxis,yaxis)
    
    
    slc = np.fft.ifft2(slc) * up_factor * up_factor
    
    # io.plotSLC(slc,acqDate,axes=axes)
    
    return slc, axes


def interp2d(grid,x,y):
    """Return interpolated grid
    
    Bilinear grid interpolation
    
    input: real valued grid as np.array and x,y coordinates as np.array
    """   
    from scipy import interpolate
    xVec = np.arange(0,grid.shape[0])
    yVec = np.arange(0,grid.shape[1])
    f = interpolate.RegularGridInterpolator((xVec, yVec), grid)
    result = f((x,y))
    return result


def getBoundingBox(Azimuth,Range,metadata,n):
    """Prepare bounding box around reflector
    
    input: n - n-times resolution cell of bounding box size
    """     

    azCrop = math.ceil(n*metadata["azimuthResolution"]/metadata["azimuthSpacing"])
    rCrop = math.ceil(n*metadata["rangeResolution"]/metadata["rangeSpacing"])
    
    minAz = int(round(Azimuth)-azCrop ) - metadata["1stAzimuth"] #python index
    maxAz = int(round(Azimuth)+azCrop+1 ) - metadata["1stAzimuth"] #
    minR = int(round(Range)-rCrop) - metadata["1stRange"]
    maxR = int(round(Range)+rCrop+1) - metadata["1stRange"]
    
    return ((minAz,maxAz),(minR,maxR))


def getCrop(Azimuth,Range,metadata,cropSize):
    """Get crop bounding box 
    in pixels for specified cropSize
    """     
    
    azCrop = math.ceil(cropSize/2/metadata["azimuthSpacing"])
    rCrop = math.ceil(cropSize/2/metadata["rangeSpacing"])
    
    minAz = int(round(Azimuth)-azCrop ) #python index
    maxAz = int(round(Azimuth)+azCrop+1 ) #
    minR = int(round(Range)-rCrop)
    maxR = int(round(Range)+rCrop+1)
    
    return ((minAz,maxAz),(minR,maxR))


def getAOIbox(stack, min_lon, max_lon, min_lat, max_lat, aver_h):
    plh = np.array([[min_lat, min_lon],
                [min_lat, max_lon],
                [max_lat, max_lon],
                [max_lat, min_lon]])/180*np.pi
    plh = np.hstack((plh, np.tile(aver_h,(4,1))))
    
    # get Crop BoundingBox
    (Azimuth,Range) = plh2radar(plh, stack.masterMetadata)
    return ((int(np.min(Azimuth)), int(np.max(Azimuth))), 
            (int(np.min(Range)), int(np.max(Range))))


def radarcode(plh,metadata,**kwargs):
    """Precise transformation of ellipsoidal geodetic coordinates
    to radar coordinates (Azimuth,Range)
    
    input: plh as np.array; metadata structure
    """  
    
    # from ellipsoidal to ECEF:
    xyz = geoUtils.plh2xyz(plh)
    
    # DEBUG
    #print('WARNING: xyz position in ETRF2000 epoch 2010.5 has been hardcoded in radarUtils.py line 508\n(see ToEditInGecoris for more informations)')
    #xyz = np.array([3908910.3663, 330932.7742, 5012262.5786])
    #
    
    # transform CRS:
    acqDate = geoUtils.decimalYear(metadata['acqDate'])
    if 'crs' in kwargs:
        if kwargs['crs'] != 'ITRS':
            xyz = geoUtils.etrf2itrf(xyz,acqDate)
        else:
            xyz = geoUtils.itrf2itrf(xyz,acqDate)
    else:
        # transform from ETRS-89 to ITRS (ITRF2014), t = acqEpoch
        xyz = geoUtils.etrf2itrf(xyz,acqDate)
    
    # get SET:
    decimalHour = (metadata['acqDate'].hour
                    + metadata['acqDate'].minute/60 
                    + metadata['acqDate'].second/3600)
    acqDate = metadata['acqDate'].strftime("%Y%m%d")
    dxyz_set = geoUtils.getSET(xyz,acqDate,decimalHour)
    xyz += dxyz_set
    # get radar time coordinates:
    (tAzimuth,tRange,satVec) = xyz2t(xyz,metadata) 
               
    # get tropo delay:
    if 'atmoDir' in kwargs:
        if 'atmoFile' in kwargs:
            tropoDelay = atmoUtils.parse_supplied_tropodelay(acqDate, 
                                                 kwargs['atmoFile']
                                                 )/speedOfLight
        else:
            tropoDelay = atmoUtils.getTropoDelay(xyz, satVec, acqDate, 
                               kwargs['atmoDir'], method='Jolviet'
                               )/speedOfLight
        ionoDelay = atmoUtils.getIonoDelay(xyz, satVec[0], 
               metadata['acqDate'], atmoDir=kwargs['atmoDir'])/speedOfLight
    else:
        tropoDelay = geoUtils.tropoDelay(xyz,satVec)
        ionoDelay = 0.1/speedOfLight # approx.
    slantDelay = tropoDelay + ionoDelay
    tRange = tRange + slantDelay    # + because GEO -> IPF
    if 'geometry' in kwargs:
        return time2radar(tAzimuth,tRange,metadata)
        
    # S1 residual bistatic correction:
    bistaticAz = s1Utils.bistaticCorrection(tRange,metadata)
    tAzimuth = tAzimuth - bistaticAz # - because zeroDoppler -> IPF
    # S1 Doppler shift
    doppler = s1Utils.dopplerRgCorrection(tRange,tAzimuth,metadata)
    tRga = tRange # for DEBUG it's tRange without doppler correction
    tRange -= doppler # - because from corr -> IPF  
    # S1 FM mismash
    FM = s1Utils.FMmismatchCorrection(xyz,tRange,tAzimuth,metadata)
    tAzimuth += FM # + because from GEO -> IPF 
    # # DEBUG
    # print('------DEBUG start in radarUtils.py line 553 -----------')
    # print('geometrical range= ' + str(1e-3 * tRg*speedOfLight) +' [km]')
    # print('atmoDelay= ' + str(1e-3 * slantDelay*speedOfLight) +' [km]')
    # print('tDoppler= ' + str(1e-3 * doppler*speedOfLight) +' [km]\n')
    # print('bistaticAz= ' + str(-bistaticAz*speedOfLight) +' [m]')
    # print('FM= ' + str(FM*speedOfLight) +' [m]')  
    # print('Az - corrections on = ' + str(tAzimuth*speedOfLight) +' [m]') 
    # print('-------------------------------------------------------')
    # convert to pixels:
    
    (Azimuth,Range) = time2radar(tAzimuth,tRange,metadata)


    return (Azimuth,Range)


def plh2radar(plh,metadata,**kwargs):
    """Transformation of ellipsoidal geodetic coordinates
    to radar coordinates (Azimuth,Range), without corrections
    
    input: plh as np.array; metadata structure
    """  
    # from ellipsoidal to ECEF:
    xyz = geoUtils.plh2xyz(plh)
    # get time coords:
    (tAzimuth,tRange,satVec) = xyz2t(xyz,metadata)  
    # convert to pixels:
    (Azimuth,Range) = time2radar(tAzimuth,tRange,metadata) 
    return (Azimuth,Range)


def estimatePeak(beta0,boundingBox,metadata,ovsFactor,method='max'):
    """Return precise IRF peak coordinates (azimuth,range) and beta0
    
    Two methods:
        (a) Simple maximum search in 1x1 resolution cell around apriori 
        coordinates
        (b) Peak fitting using elliptic paraboloid fit around small 
        3x3 up-samples around maximum from (a)
    
    input: beta0 np.array grid; boundingBox, metadata structure
    """  
    
    if method == 'max':
        az = beta0.shape[0]/2
        r = beta0.shape[1]/2
        azCrop = math.ceil(2*metadata["azimuthResolution"]/metadata["azimuthSpacing"]*ovsFactor)
        rCrop = math.ceil(2*metadata["rangeResolution"]/metadata["rangeSpacing"]*ovsFactor)
        minAz = int(round(az)-azCrop ) #python index
        maxAz = int(round(az)+azCrop ) #
        minR = int(round(r)-rCrop)
        maxR = int(round(r)+rCrop)
        beta0crop = beta0[minAz:maxAz,minR:maxR]
        idx = np.unravel_index(np.argmax(beta0crop, axis=None), beta0crop.shape)
        peakAz = (boundingBox[0][0]*ovsFactor + minAz + idx[0])/ovsFactor
        peakR = (boundingBox[1][0]*ovsFactor + minR + idx[1])/ovsFactor
        peakBeta0 = beta0crop[idx]
        localAz = idx[0]
        localR = idx[1]
    
    elif method == 'fit':
        # work on 1 x 1 resolution crop
        az = beta0.shape[0]/2
        r = beta0.shape[1]/2
        azCrop = math.ceil(1*metadata["azimuthResolution"]/metadata["azimuthSpacing"]*ovsFactor)
        rCrop = math.ceil(1*metadata["rangeResolution"]/metadata["rangeSpacing"]*ovsFactor)
        minAzloc = int(round(az)-azCrop)
        maxAzloc = int(round(az)+azCrop)
        minRloc = int(round(r)-rCrop)
        maxRloc = int(round(r)+rCrop)
        beta0crop = beta0[minAzloc:maxAzloc,minRloc:maxRloc]
        idx = np.unravel_index(np.argmax(beta0crop, axis=None), beta0crop.shape)
        minAz = minAzloc + idx[0]-3
        maxAz = minAzloc + idx[0]+3
        minR = minRloc + idx[1]-3
        maxR = minRloc + idx[1]+3
        # Print bot (1)beta0 and (2)beta0Crop DEBUG
        # (1) 
        # nAz = boundingBox[0][1] - boundingBox[0][0]
        # nR = boundingBox[1][1] - boundingBox[1][0]
        # yaxis = np.linspace(boundingBox[0][0],boundingBox[0][1],nAz*ovsFactor)
        # xaxis = np.linspace(boundingBox[1][0],boundingBox[1][1],nR*ovsFactor)
        # axes = (xaxis,yaxis)
        # io.plotSLC(beta0,'beta0'+str(metadata["acqDate"]),axes = axes,units='pixel')
        # #(2)
        # nAz = maxAzloc-minAzloc
        # nR = maxRloc-minRloc
        # print([minAzloc,maxAzloc,minRloc,maxRloc])
        # yaxis = boundingBox[0][0] + np.linspace(minAzloc/ovsFactor,maxAzloc/ovsFactor,nAz)
        # xaxis = boundingBox[1][0] + np.linspace(minRloc/ovsFactor,maxRloc/ovsFactor,nR)
        # axes = (xaxis,yaxis)
        # io.plotSLC(beta0crop,'beta0crop'+str(metadata["acqDate"]),axes = axes,units='pixel',bars=[(minAz-13)/32+boundingBox[0][0],(maxAz+13)/32+boundingBox[0][0],minR-13,maxR+13])
        # fit data:
        beta0crop = beta0[minAz:maxAz+1,minR:maxR+1]
        # estimation grid:
        X,Y = np.meshgrid(np.arange(minAz,maxAz+1),np.arange(minR,maxR+1))
        XX = X.flatten()
        YY = Y.flatten()
        XY = np.c_[XX,YY]
        data = np.sqrt(beta0crop).flatten()
        # best-fit paraboloid
        A = np.c_[np.ones(data.size),XY, np.prod(XY, axis=1), XY**2]
        C,_,_,_ = scipy.linalg.lstsq(A, data)
        # derivaties to find extrema:
        #maxX = C[1]+C[3]*YY+2*C[4]*XX
        #maxY = C[2]+C[3]*XX+2*C[5]*YY
        # find X and Y above such that both equals 0
        Q = np.array([[2*C[4],C[3]], [C[3],2*C[5]]])
        rhs = np.array([-C[1],-C[2]])
        sol = np.linalg.inv(Q).dot(rhs)
        # solution must be within estimation bounds!:
        if sol[0] < minAz:
            sol[0] = minAz
        elif sol[0] > maxAz:
             sol[0] = maxAz
        if sol[1] < minR:
            sol[1] = minR
        elif sol[1] > maxR:
             sol[1] = maxR     
            
        peakBeta0 = np.power(np.dot(np.c_[1, sol[0], sol[1], sol[0]*sol[1], 
                                        sol[0]**2, sol[1]**2], C),2)
        peakAz = ((boundingBox[0][0]+metadata["1stAzimuth"])*ovsFactor + sol[0])/ovsFactor
        peakR = ((boundingBox[1][0]+metadata["1stRange"])*ovsFactor + sol[1])/ovsFactor
        localAz = sol[0]
        localR = sol[1]
    
    return peakAz,peakR,peakBeta0,localAz,localR


def RCSintegral(SLCderamp,metadata,ovsFactor):
    """Return RCS and SCR of CR SLC measurement
    
    Integral method to estimate RCS & SCR of point scatterer on SLC crop
    
    input: deramped SLC grid as np.array, metadata structure
    """ 
    
    SLCovs = oversample(SLCderamp,ovsFactor)
    beta0 = np.power(np.abs(SLCovs),2)/(metadata['beta0']**2)
    azFactor = metadata["azimuthResolution"]/metadata["azimuthSpacing"]*ovsFactor
    rgFactor = metadata["rangeResolution"]/metadata["rangeSpacing"]*ovsFactor
    
    # get mainlobe power:
    az = beta0.shape[0]/2
    r = beta0.shape[1]/2
    azCrop = np.ceil(1*azFactor)
    rCrop = np.ceil(1*rgFactor)
    minAz = int(round(az)-azCrop )
    maxAz = int(round(az)+azCrop )
    minR = int(round(r)-rCrop)
    maxR = int(round(r)+rCrop)
    beta0crop = beta0[minAz:maxAz,minR:maxR]
    idx = np.unravel_index(np.argmax(beta0crop, axis=None), beta0crop.shape)
    peakAz = (minAz + idx[0])
    peakR = (minR + idx[1])
    # re-center the crop:
    minAz = int(peakAz-azCrop )
    maxAz = int(peakAz+azCrop )
    minR = int(peakR-rCrop)
    maxR = int(peakR+rCrop)
    beta0mainlobe = beta0[minAz:maxAz,minR:maxR]
    
    #% estimate clutter:
    clutAzFactor = int(2*azFactor)
    clutRgFactor = int(2*rgFactor)
    clutterSample = np.zeros(4)
    clutterCrop = beta0[0:clutAzFactor,0:clutRgFactor]
    clutterSample[0] = np.sum(clutterCrop,axis=None)/clutterCrop.size
    clutterCrop = beta0[0:clutAzFactor,-clutRgFactor:-1]
    clutterSample[1] = np.sum(clutterCrop,axis=None)/clutterCrop.size
    clutterCrop = beta0[-clutAzFactor:-1,0:clutRgFactor]
    clutterSample[2] = np.sum(clutterCrop,axis=None)/clutterCrop.size
    clutterCrop = beta0[-clutAzFactor:-1,-clutRgFactor:-1]
    clutterSample[3] = np.sum(clutterCrop,axis=None)/clutterCrop.size
    # remove samples more powerful than mainlobe:
    cluttIdx = np.nonzero(clutterSample < np.mean(beta0mainlobe,axis=None))[0]
    clutterPower = np.mean(clutterSample[cluttIdx])
    
    #% mainlobe power
    mainlobePower = np.sum(beta0mainlobe-clutterPower,axis=None)
    SCR = 10*np.log10(mainlobePower/(clutterPower*beta0mainlobe.size))
    
    #% relative power in sidelobes:
    sidelobePower = 0
    sidelobeCrop = beta0[minAz:maxAz,0:clutRgFactor]
    sidelobePower += np.sum(sidelobeCrop-clutterPower,axis=None)
    sidelobeCrop = beta0[minAz:maxAz,-clutRgFactor:-1]
    sidelobePower += np.sum(sidelobeCrop-clutterPower,axis=None)
    sidelobeCrop = beta0[0:clutAzFactor,minR:maxR]
    sidelobePower += np.sum(sidelobeCrop-clutterPower,axis=None)
    sidelobeCrop = beta0[-clutAzFactor:-1,minR:maxR]
    sidelobePower += np.sum(sidelobeCrop-clutterPower,axis=None)
    #ISLR:
    ISLR = sidelobePower/mainlobePower
    C_F = 1/(1+ISLR)
    
    #% RCS:
    pixA = metadata['azimuthSpacing']*metadata['rangeSpacing']/ovsFactor/ovsFactor
    RCS = mainlobePower*pixA/C_F
    RCSdB = 10*np.log10(RCS)
    
    return RCSdB,SCR
