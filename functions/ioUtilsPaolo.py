import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
from gecoris import radarUtils
from functions import s1UtilsPaolo
import time
import csv

# Constants
azResolution=21.89
rangeResolution=3.1034415942028986
wavelength=0.05546576
aCR= 0.9 # [m] inner leg of IGRS, from the msc thesis on the design
RCStheor = 4*np.pi*aCR**4 / (3*wavelength**2)
myDpi = 100
cropsDir = '/home/caroline-pbazzocchi/algorithm/gecoris/RC/'



def datetimeToMJD(dt):
    # Reference datetime for MJD calculation (November 17, 1858)
    referenceDatetime = datetime(1858, 11, 17)

    # Calculate the time difference in days
    timeDifference = dt - referenceDatetime

    # Calculate the total number of days (including fractions)
    totalDays = timeDifference.days + timeDifference.seconds / (24 * 3600)


    # Calculate the Modified Julian Date (MJD)
    MJD = totalDays #- 2400000.5

    return MJD

def warning(strMsg):
    # print a warning message
    # in case you want to spot the line of the code where the warning is, you simply have to raise an exception here with the message indicated in the warning
    # This can be done modifying the variable stopAt
    
    if isinstance(strMsg,str):
        print('WARNING: '+ strMsg)
        msg = strMsg
    elif isinstance(strMsg,list):
        for Msg in strMsg:
            print('WARNING: '+ Msg)
        msg = strMsg[0]
            
    
    stopAt = 'forExample'
    if msg == stopAt:
        raise Exception('This is the line where the WARNING is')
    
    return

def plotSLC(SLC,acqDate,**kwargs):
    

    

    title = str(acqDate)
    elev = 37.3
    if isinstance(SLC, np.ndarray):
        amplitude_i = np.abs(SLC)
                    
        # calibrate the image for apparent RCS - peak estimation method
        amplitude_i = amplitude_i * azResolution * rangeResolution
            
        
        
        if ('axes' in kwargs) and ('units' in kwargs) :
            
            if (kwargs["units"] == 'meters') or (kwargs["units"] == 'both'):
                # Create figure
                fig,ax = plt.subplots(1,1)
            
                # parse input axes and convert slant range 2 ground range
                x = kwargs['axes'][0] / np.sin(elev*np.pi/180)
                y = kwargs['axes'][1]
            
                # meshgrid
                X,Y = np.meshgrid(x,y)
                # colormesh
                im = ax.pcolormesh(X, Y, amplitude_i, cmap='jet')#, norm=plt.matplotlib.colors.SymLogNorm(linthresh=0.01, linscale=1))
            
                # coordinates of the center
                centerAz = 0.5*(y[-1]+y[0])
                centerR = 0.5*(x[-1]+x[0])
            
                # set titles and info
                fig.suptitle(title + ' - Apparent RCS $[m^2]$')
                ax.set_title('Center point Az='+str(int(centerAz))+', R='+str(int(centerR))+'\n$to \ be \ compared \ with \ google \ maps, \ North \ as \ per \ arrow$\n$Theoretical \ RCS \ = \ '+str(round(RCStheor,2)) + ' \ m^2$',fontsize=9)
                ax.set_xlabel('Ground Range [m] (elevation='+str(elev)+' deg)')
                ax.set_ylabel('Azimuth [m]')
                ax.invert_yaxis()
                ax.invert_xaxis()
            
                # Normalize the arrow position
                arrow_x = 0.1
                arrow_y = 0.8
                arrow_dx = 0.05*np.cos(106*np.pi/180)
                arrow_dy = 0.05*np.sin(106*np.pi/180)*((x[-1]-x[0]))/((y[-1]-y[0]))

                # Draw an arrow using normalized coordinates
                arrow = ax.arrow(arrow_x, arrow_y, arrow_dx, arrow_dy, width=0.005, color='green',
                             transform=ax.transAxes)
            
                # draw the scope shape
                ax.axvline(x=centerR, color='green', linestyle='-',linewidth=.3)
                ax.axhline(y=centerAz, color='green', linestyle='-',linewidth=.3)
            
                im.set_clim(np.percentile(amplitude_i, 20), np.percentile(amplitude_i, 99))
                # im.set_clim(0, 10**(np.ceil(np.log10(RCStheor))))
                plt.axis('scaled')
            
                # Show colorbar
                cbar = plt.colorbar(im, ax=ax,shrink=0.3)
            
                # RCS
            
            
                # save it
                plt.savefig(cropsDir+title+'-meters.png',dpi=myDpi, bbox_inches="tight", pad_inches = 0)
                
                
            if (kwargs["units"] == 'pixel') or (kwargs["units"] == 'both'):
                # Create figure
                fig,ax = plt.subplots(1,1)
            
                # parse input axes 
                x = kwargs['axes'][0]
                y = kwargs['axes'][1]
            
                # meshgrid
                X,Y = np.meshgrid(x,y)
            
                # colormesh
                im = ax.pcolormesh(X, Y, amplitude_i, cmap='jet')#, norm=plt.matplotlib.colors.SymLogNorm(linthresh=0.01, linscale=1))
            
                # coordinates of the center
                centerAz = 0.5*(y[-1]+y[0])
                centerR = 0.5*(x[-1]+x[0])
            
                # set titles and info
                fig.suptitle(title + ' - Apparent RCS $[m^2]$')
                ax.set_title('Center point Az='+str(int(centerAz))+', R='+str(int(centerR))+'$Theoretical \ RCS \ = \ '+str(round(RCStheor,2)) + ' \ m^2$',fontsize=9)
                ax.set_xlabel('Range [pixel]')
                ax.set_ylabel('Azimuth [pixel]')
                ax.invert_yaxis()
                ax.invert_xaxis()
                print('###############')
            
                # draw the scope shape
                ax.axvline(x=centerR, color='green', linestyle='-',linewidth=.3)
                ax.axhline(y=centerAz, color='green', linestyle='-',linewidth=.3)
            
                # im.set_clim(np.percentile(amplitude_i, 20), np.percentile(amplitude_i, 90))
                # im.set_clim(0, 10**(np.ceil(np.log10(RCStheor))))
                plt.axis('scaled')
            
                # Show colorbar
                cbar = plt.colorbar(im, ax=ax,shrink=0.3)
            
                # bars
                if 'bars' in kwargs:
                    ax.axhline(y=kwargs["bars"][0], color='red', linestyle='-',linewidth=.5)
                    ax.axhline(y=kwargs["bars"][1], color='red', linestyle='-',linewidth=.5)
                    ax.axvline(x=kwargs["bars"][2], color='red', linestyle='-',linewidth=.5)
                    ax.axvline(x=kwargs["bars"][3], color='red', linestyle='-',linewidth=.5)
                    print('------------------- BETA0 CROP BARS -----------------------\n')
                    print(kwargs["bars"])
            
            
                # save it
                plt.savefig(cropsDir+title+'-pixel.png',dpi=myDpi, bbox_inches="tight", pad_inches = 0)
            
            
        # show both image with scaled axis and not
        if (not 'axes' in kwargs) or ('both' in kwargs):
            
            # creat figure
            fig,ax = plt.subplots(1,1)
            im = ax.imshow(amplitude_i, cmap='gray',origin='lower')#, norm=plt.matplotlib.colors.SymLogNorm(linthresh=0.01, linscale=0.1))
            ax.set_title(title + ' - Apparent RCS\ntheoretical RCS = '+str(round(RCStheor,2))+' m2')
            ax.set_xlabel('Range [pixel]')
            ax.set_ylabel('Azimuth [pixel]')
        

            # im.set_clim(np.percentile(amplitude_i, 20), np.percentile(amplitude_i, 90))
            # im.set_clim(0, 10**(np.ceil(np.log10(RCStheor))))
            plt.axis('scaled')
            
            # Show colorbar
            cbar = plt.colorbar(im, ax=ax,shrink=0.3)
            
            # Invert the y axis (because descending geometry) and adjust the tick positions and labels
            ax.invert_yaxis()
            ax.invert_xaxis()
            x_min, x_max = plt.xlim()
            y_min, y_max = plt.ylim()
            
            # Draw a horizontal line centered at y = 0.5 (50% of the y-axis range)
            ax.axhline(y=(y_min+y_max)/2, color='green', linestyle='-',linewidth=.3)
            # Draw a vertical line centered at x = 0.5 (50% of the x-axis range)
            ax.axvline(x=(x_min+x_max)/2, color='green', linestyle='-',linewidth=.3)
            
            if 'bars' in kwargs:
                ax.axhline(y=kwargs["bars"][0], color='red', linestyle='-',linewidth=.5)
                ax.axhline(y=kwargs["bars"][1], color='red', linestyle='-',linewidth=.5)
                ax.axvline(x=kwargs["bars"][2], color='red', linestyle='-',linewidth=.5)
                ax.axvline(x=kwargs["bars"][3], color='red', linestyle='-',linewidth=.5)
                print('------------------- BETA0 CROP BARS -----------------------\n')
                print(kwargs["bars"])
   
            # Optionally, adjust other properties of the plot as needed
            plt.savefig(cropsDir+title+'-pixel.png',dpi=myDpi, bbox_inches="tight", pad_inches = 0)
    else:
        # amplitude_i = SLC.amplitude.isel(time=0)
        amplitude_i = SLC.amplitude
        
        fig,ax = plt.subplots(1,1)

        ax.imshow(amplitude_i)
        amplitude_i.plot(robust=True, ax=ax, cmap='gray')  # cmap='jet'
        ax.invert_yaxis()
    
        print('saving image...')
        plt.savefig(cropsDir+title+'-sxarray.png',dpi=myDpi, bbox_inches='tight')

    

    plt.close()
    return

def rx2apex(plhRx):
    
    # convert in degrees
    plhRx = np.array([plhRx[0]*180/np.pi,plhRx[1]*180/np.pi,plhRx[2]])
                      
    # Variables
    h1 = 146.1
    h2 = 63.2
    w1 = 28.7
    w2 = 28.7
    d = 8
    t = 0.2
    l1 = 52.5
    l2 = 64
    l3 = 63.4
    l4 = 63.4
    tilt = 5.1
    
    # Geometric formulas
    b = h2 * np.tan(np.radians(5.1))
    alpha = np.arcsin((l1+d+l2)/(l3+l4))
    
    Beta = 90 - np.degrees(alpha)
    h3 = np.sin(np.radians(Beta))*b
    
    # display box
    # print('-------------------- rx2apex --------------------')
    # print(f'\nplh GNSS: {plhRx}')
    # Distances between apex and bottom center of GNSS antenna
    diffUp = h1+h2
    # print(f'Difference in up direction: {diffUp:.2f} cm')
    
    diffEast = (w1 + w2 + d + 2*t) / 2
    # print(f'Difference in east direction: {diffEast:.2f} cm')
    
    diffNorth = ((l1+d+l2)/2 - h3) - (l1+d/2)
    # print(f'Difference in north direction: {diffNorth:.2f} cm')
    
    # Difference in coordinates
    lat = 50.7
    tanB = 0.99664719 * np.tan(np.deg2rad(lat))
    c1 = 6378137*np.pi/180*np.cos(np.arctan(tanB))
    diffLon = diffEast/100/c1
    
    # print(f'The difference in longitudinal coordinates between the apexes will be {diffLon:.8f} degrees')
    
    c4 = 111132.954 - 559.882*np.cos(2*np.deg2rad(lat)) + 1.175*np.cos(4*np.deg2rad(lat))
    diffLat = diffNorth/100/c4
    
    # print(f'The difference in latitudinal coordinates between the apexes will be {diffLat:.8f} degrees')
    
    # plh apex correction
    # print(f'diffLon: {diffLon} degrees')
    plhAsc = np.array([plhRx[0]+diffLat,plhRx[1]-diffLon,plhRx[2]-diffUp*0.01])
    plhDsc = np.array([plhRx[0]+diffLat,plhRx[1]+diffLon,plhRx[2]-diffUp*0.01])
    # print(f'\nplh ascending apex: Lat={plhAsc[0]:.9f}, Lon={plhAsc[1]:.9f}, h={plhAsc[2]:.3f}')
    # print(f'\nplh descending apex: Lat={plhDsc[0]:.9f}, Lon={plhDsc[1]:.9f}, h={plhDsc[2]:.3f}')
    # print('--------------------------------------------------')
    
    return plhAsc,plhDsc

def RCexport(stacks,outDir):
    
   
    
    
    for stack in stacks:
        
        tableHead = ['stationID','Range','Azimuth']
        fileHead = [f'This file contains the RADAR coordinates of the selected stations, specified for each coregistered stack. Created with GECORIS on {time.asctime()}','','']
        
        csvpath = outDir + f'{stack.id}_RadarCoordinates.csv'
        matrix = stack.stationsMatrix
        
        stationsIW = stack.stationsIW
        stations = stationsIW[0]+stationsIW[1]+stationsIW[2]
        
        datesColumns = ['' for col in matrix[1,:]]
        datesColumnsDash = ['----' for col in matrix[1,2:]]
        
        fileHead = fileHead+datesColumns
        acqDates = stack.acqDates
        tableHead = tableHead+acqDates
        
        with open(csvpath, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fileHead)
            csvwriter.writerow(['----','----','----']+datesColumnsDash)
            csvwriter.writerow(['stack:',stack.id,'']+datesColumns)
            csvwriter.writerow(tableHead)
            for i in range(len(stations)):
                ID = stations[i].id
                peakR = matrix[i,0]
                peakAz = matrix[i,1]
                status = [matrix[i,2+k] for k in range(len(acqDates))]
                row = [ID,
                       peakR,
                       peakAz]+status
                csvwriter.writerow(row)
            csvwriter.writerow(['----','----','----']+datesColumnsDash)
            
    return       
                
            
        
    
#     for i in range(0,len(stations)):
        
#         if stations[0].stacks[0]["type"] == 'coreg':
#             stacks = [['stack: ',stack["id"],''] for stack in stations[0].stacks]
#             fileHead = [f'This file contains the RADAR coordinates of the selected stations, specified for each coregistered stack. Created with GECORIS on {time.asctime()}','','']
            
            
#         elif stations[0].stacks[0]["type"] == 'raw':
#             stacks = [[stack["id"],f'swath_{stack["metadata"]["swath"]}_burst_{stack["metadata"]["burstInfo"]}',''] for stack in stations[0].stacks]
        
#             fileHead = [f'This file contains the selected stations\' RADAR coordinates, specified for each acquisition date and each stack. Created with GECORIS on {time.asctime()}','','']
            
       
        
        
        
        
        
        
#             for i in range(0,len(stacks)):
#                 csvwriter.writerow(['----','----','----'])
#                 csvwriter.writerow(stacks[i])
#                 csvwriter.writerow(tableHead)
#                 for dx in [0]:#range(0,len(stations[0].stacks[i]["data"])):
#                     # csvwriter.writerow(['','',''])
#                     acqDate = stations[0].stacks[i]["data"][0]
#                     # csvwriter.writerow([acqDate,'',''])
#                     for station in stations:
#                         if acqDate < station.startDate:
#                             continue
#                         row = [station.id,
#                                round(station.stacks[i]["data"][5]),
#                                round(station.stacks[i]["data"][6])]
#                         csvwriter.writerow(row)
                    
                    
#     return


def plotMatch(stationID,peakAz,peakR,rcAz,rcR,SLC,metadata,outDir):

    # Observed Pixel coordinates
    lineObs = round(peakAz)
    pixelObs = round(peakR)
    
    # Computed Pixel coordinates
    lineCom = round(rcAz[0])
    pixelCom = round(rcR[0])
    
    # acqDate
    acqDate = metadata["acqDate"].strftime("%Y%m%d")
    
    # define a bounding box
    boundingBox = radarUtils.getBoundingBox(lineObs,pixelObs,metadata,1)
    
    # read SLC
    # SLC = s1UtilsPaolo.readSLC(file,metadata,boundingBox,method='raw',deramp = True) 
    amplitude = np.abs(SLC)
    
    # calibrate: apparent RCS
    amplitude = np.power(np.abs(amplitude),2)/(metadata['beta0']**2)
    amplitude = amplitude*metadata["azimuthResolution"]*metadata["rangeResolution"]
    
    
    
    # plot the SLC crop, centered in the pixel containing the reflector
    fig,ax = plt.subplots(1,1)

    ax.imshow(amplitude)
    amplitude.plot(robust=True, ax=ax, cmap='gray')  # cmap='jet'
    ax.invert_yaxis()
    ax.scatter(peakR-metadata["1stRange"],peakAz-metadata["1stAzimuth"],label='Observed',color='black', marker='v',s=30)
    
    # draw scope shape
    ax.axvline(x=rcR-metadata["1stRange"], color='green', linestyle='-',linewidth=.3,label='Computed')
    ax.axhline(y=rcAz-metadata["1stAzimuth"], color='green', linestyle='-',linewidth=.3)
    
    ax.set_ylabel(r'Azimuth (local) [pixel]',fontsize=14)
    ax.set_xlabel(r'Range (local) [pixel]',fontsize=14)
    ax.legend(fontsize = 16)
    ax.set_title(stationID+'_'+acqDate+'_Match result\nObserved in ('+str(lineObs)+','+str(pixelObs)+'),Computed to ('+str(lineCom)+','+str(pixelCom)+')')
    
        
    if (lineObs == lineCom) and (pixelObs == pixelCom):
        matchCard = 'match'
    else:
        matchCard = 'mismatch'

    saveDir = outDir+os.sep+matchCard+os.sep
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
            
    plt.savefig(saveDir+stationID+'_'+acqDate+'_match.png',dpi=200, bbox_inches='tight')
    plt.close()
    
def plotRC(SLC,stack,outDir):
    
    matrix = stack.stationsMatrix
        
    stationsIW = stack.stationsIW
    stations = stationsIW[0]+stationsIW[1]+stationsIW[2]
    
    R = matrix[:,0]
    Az = matrix[:,1]
    amplitude = np.abs(SLC)
    
    fig,ax = plt.subplots(1,1)

    print('plotting SLC...')
    ax.imshow(amplitude)
    amplitude.plot(robust=True, ax=ax, cmap='gray')  # cmap='jet'
    ax.invert_yaxis()
    ax.scatter(R-stack.masterMetadata["1stRange"],Az-stack.masterMetadata["1stAzimuth"],label='stations',color='yellow', marker='v',s=30)
    
    # draw scope shape    
    ax.set_ylabel(r'Azimuth (local) [pixel]',fontsize=14)
    ax.set_xlabel(r'Range (local) [pixel]',fontsize=14)
    ax.legend(fontsize = 16)
    ax.set_title(stack.id+'_RadarCoordinates')
    
    
    print('saving SLC...')     
    plt.savefig(outDir+stack.id+'_RadarCoordinates.png',dpi=200, bbox_inches='tight')
    plt.close()
    
    
    
    