import os
import numpy as np
import geopandas as gpd
from pathlib import Path
import shapely.geometry as sg
from time import asctime as timenow
import csv

def swathBurst(stackDir,stations,mode='slave'):
    
    '''
    The function finds the burst in which the stations are located according to the structure of the database on spider
    
    To locate them the following steps are performed:
    
    (1) The shapeFile present in the datatake directory is read, 
        and the polygons describing the coverage of each burst are loaded
        
    (2) The stations are loaded and for each one the polygon in which they are included is found.
        For each station the IW number and the burst number are saved
        
    (3) A check is performed wether a single burst common to all the stations exists, if not an Exception is thrown.
        A check wether a common burst exists is performed, if not multiple bursts in the same swath are returned.
        The two checks have the following purpose:
        - if a reflector is located in the overlapping region between two swaths,
        the swath non-common to all the other stations is discarded. A single swath is possible due to the structure 
        of the metadata dictionary.
        - if the all the stations are not covered by a single swath, multiple swaths are loaded in order to 
        avoid radarcoding errors.
    (4) The path of the metadata (.res) file is generated with the common burst
        
    '''
    
    # Load geometry data from swathburst_coverage shapefile
    shpPath = [p for p in Path(stackDir).rglob('../stackburst_coverage.shp')] 
    data = gpd.read_file(shpPath[0]) # Read the Shapefile using geopandas
    geometry = data.geometry
    

    # Iterate over bursts and stations
    swathburst = [[] for st in stations]
    for sb in range(len(data)):
        # Area of Coverage of current iteration
        AoC = geometry[sb]
            
        # for each target
        for loop,station in zip(range(0,len(stations)),stations):
            
            if station.descending:
                stationPoint = sg.Point(station.descending.longitude*180/np.pi, station.descending.latitude*180/np.pi, station.descending.elevation)
            elif station.ascending:
                stationPoint = sg.Point(station.ascending.longitude*180/np.pi, station.ascending.latitude*180/np.pi, station.ascending.elevation)
            else:
                raise Exception('Both ascending and descending are not active...')

            if AoC.contains(stationPoint):
                # print(station.id+": "+data.name[sb]) # DEBUG
                swathburst[loop].append(data.name[sb])
                            
                            
    # compute the centerpoint, will be used to select the correct original product
    centerPoint = AoC.centroid
    
    print(f'stationPoint: {stationPoint}')
    print(f'centerPoint: {centerPoint}')
    # group the swathIDs for each station
    swathIDs = [np.array([int(swathburst[q][i][6]) for i in range(0,len(swathburst[q]))]) for q in range(0,len(stations))] 



    # Find the swath common to every reflector
    commonSwath = swathIDs[0]
    for i in range(0,len(swathIDs)-1):
        commonSwath = np.intersect1d(swathIDs[i+1],commonSwath)
    commonSwath = np.unique(commonSwath)
    
    # message/exception about intersection
    if len(commonSwath)>1:
        print('All stations have more than one swath in common, selecting the first available...')
    elif len(commonSwath)<1:
        raise Exception('Selected stations are located on different Interferometric swaths (IW). \nPlease sort them per IW number and run the code for each of them.')
                    
    burstIDs = [np.array([int(swathburst[q][i][14:]) for i in range(0,len(swathburst[q]))]) for q in range(0,len(stations))]
    # print(f'commonSwath = {commonSwath}, burstIDs = {burstIDs}')
    # Find the burst common to every reflector
    commonBurst = burstIDs[0]

    for i in range(0,len(burstIDs)-1):
        commonBurst = np.intersect1d(burstIDs[i+1],commonBurst)


    
    # generate .res metadata path          
    if commonBurst.size == 1:
        # there is a burst common for every reflector
        burstID = str(commonBurst[0])
        swathID = commonSwath[0]
        slavePath =  '/swath_'+str(swathID)+'/burst_'+burstID+'/'+mode+'.res'
        print('Common burst Found...')
    elif commonBurst.size>1:
        # there is a burst common for every reflector
        burstID = str(commonBurst[1])
        swathID = commonSwath[0]
        slavePath =  '/swath_'+str(swathID)+'/burst_'+burstID+'/'+mode+'.res'
        print('Multiple bursts contain the same stations... Selecting one.')
    else:        
        raise Exception('Reflectors on different bursts... \n a burst-wise analysis is desired please select stations on the same burst.\nOtherwise please run the analysis on the merged SLC image.')

        
    
    return slavePath,str(commonSwath[0]),burstID,stationPoint

from datetime import datetime

def createMatrix(stations,acqDates,stackID,plotFlag=0):
    
    """
    This function creates a N-by-(2+M) matrix, where:
    
    N: number of stations to be processed in the stack
    M: number of acquisition dates
    
    the single element is 1 if the station is active on that date, otherwise 0.
    In the first two columns, the Range and Azimuth coordinates will be stored once computed.
    
    The second output of the function is a dateIdx. If any, it is the index of the date where all the stations
    are active, otherwise a False is returned. This allow to load just the image corresponding to 
    the common date, saving time and resources.
    
    """

    
    
    matrix = np.zeros((len(stations),2+len(acqDates)))
    
    for i in range(len(stations)):
        
        station = stations[i]
        readCoordinates
        startDate = station.startDate
        endDate = station.endDate
        startIdx = next(x for x, val in enumerate(acqDates) if val > startDate)
        if endDate == '99999999':
            endIdx = len(acqDates)-1
        else:
            endIdx = next(x for x, val in enumerate(acqDates) if val > endDate)

        matrix[i,2+startIdx:2+endIdx+1] = np.ones((1,endIdx-startIdx+1))
        
    if plotFlag > 0:
        import matplotlib.pyplot as plt
        
        fig,ax = plt.subplots(1,1)
        
        ax.imshow(matrix)
        plt.xticks(range(len(acqDates)+2), ['Range','Azimuth']+acqDates)
        plt.yticks(range(len(stations)),[st.id for st in stations])
        ax.xaxis.set_tick_params(rotation=90)
        plt.show()
    

    return matrix

def splitIW(stations,stackDir):
    
    """
    This functions check which stations are contained in the current stack, and sort them
    according to the subswath where they are contained (IWx), by:
    
    (i)  Checking for each station which is the subswath containing it. 
         In case of overlap, swath in which the station has the biggest distance from the borders is selected.
         An additional check is made on the swathID. Since the coverage shapefile provide each burst's footprint,
         the check on the distance with the borders is performed only between different swaths, and not
         between different bursts.
        
    (ii) Creating a list, which contains three sublists, one for each subswath.
    
    This function is thought to be used in a processing of more than one stack. 
    In that case, the stations provided to the Radar-coding algorithm are the one in the area of interest, 
    but it is not guaranteed that all the stacks processed contain all the stations in the area of interest.
    stationsIN is the list containing all the stations inside the stacks.
    
    """
    # Extracting geometry features
    shpPath = [p for p in Path(stackDir).rglob('../stackburst_coverage.shp')] 
    data = gpd.read_file(shpPath[0]) # Read the Shapefile using geopandas
    geometry = data.geometry
    
    # associate to each station its swathID, in case of overlap picking the one which borders are furter
    swathIDs = []
    stationsIW = [[],[],[]]
    for i in range(len(stations)):
        # station of the current iteration
        station = stations[i]
        # extract stationPoint
        if station.descending:
            stationPoint = sg.Point(station.descending.longitude*180/np.pi, station.descending.latitude*180/np.pi, station.descending.elevation)
        elif station.ascending:
            stationPoint = sg.Point(station.ascending.longitude*180/np.pi, station.ascending.latitude*180/np.pi, station.ascending.elevation)
        else:
            raise Exception('Both ascending and descending are not active...')
        # Iterate over swathburst
        swathID = 0
        dist = 0
        for sb in range(len(data)):
            # update geometry
            AoC = geometry[sb]
            # check if contains station, if the distance is larger than the previous checked one, and if the swath is different
            if AoC.contains(stationPoint) and stationPoint.distance(AoC.exterior) > dist and int(data.name[sb][6]) != swathID:
                dist = stationPoint.distance(AoC.exterior)
                swathID = int(data.name[sb][6])
        # append swathID to the swathIDs vector
        swathIDs.append(swathID)
        # append the station to the corresponding sublist
        if swathID != 0:
            stationsIW[swathID-1].append(station)

    stationsIN = stationsIW[0]+stationsIW[1]+stationsIW[2]
    print(f'{len(stationsIN)} stations on {int(len(stationsIW[0])>0)+int(len(stationsIW[1])>0)+int(len(stationsIW[2])>0)} subswaths to process..')
            
    return stationsIW,stationsIN    
        
import matplotlib.pyplot as plt      
from gecoris import ioUtils
from collections import Counter

def RCexport(stations,stacks,outDir,plotFlag=1):
    
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    
    
    for stack in stacks:
        
        acqDates = stack.acqDates
        print('start')

        stMatrix = []
        RCSanalysis = []
        
        for station in stations:

            # find dates where the station was active
            startDate = station.startDate
            endDate = station.endDate
            dateList = sorted([str(p).split('/')[-1] 
                                   for p in Path(stack.stackDir).glob('[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]')])
            
            startIdx = next(x for x, val in enumerate(dateList) if val > startDate)
            if endDate == '99999999':
                endIdx = len(dateList)-1
            else:
                endIdx = next(x for x, val in enumerate(dateList) if val > endDate)
                    

            
            timeRow = [0 for i in range(startIdx)]+[1 for i in range(endIdx-startIdx)]+[0 for i in range(endIdx-(len(dateList)-1))]
                                                                                                                    
                              
            
            # extract data
            stackIdx = station.getStackIdx(stack.id)
            data = station.stacks[stackIdx]["data"]
            active = np.array(data["status"])
            R = [data["range"][i] for i in range(len(acqDates)) if data["status"][i]]
            Az = [data["azimuth"][i] for i in range(len(acqDates)) if data["status"][i]]
            RCS = station.stacks[stackIdx]["reflRCS"]
            sigRCS = station.stacks[stackIdx]["sigRCS"]
            RCS0 = station.stacks[stackIdx]["RCS0"]-1
                              
            
            # Extract the most common coordinates pair
            pairs = [(R[i],Az[i]) for i in range(len(R))]
            pCount = Counter(pairs)
            mode = pCount.most_common()
            modeRC = mode[0][0]
                              
            
            
            # check if the predicted RCS falls into the 3-sigma confidence interval
            cLev = 5
            RCSint = (RCS-cLev*sigRCS,RCS+cLev*sigRCS)
            if (RCS0 <= RCSint[1]):
                if (RCS0 >= RCSint[0]):
                    vFlag = 0
                else:
                    vFlag = +1
            else:
                vFlag = -1
            
            RCSanalysis.append([station.id,RCS0,RCS,vFlag,cLev*sigRCS])
                
            # write coordinates and RCS check
            stRow = [station.id,vFlag,modeRC[0],modeRC[1]]+timeRow
            
            # fill matrix
            stMatrix.append(stRow)
                              
          
            # plot coordinates timeseries, if required
            if plotFlag > 0:
                fig,axR = plt.subplots(figsize=(20,3))
                plt.rcParams.update({'font.size': 25})
                xlabels = [data["acqDate"][i] for i in range(len(acqDates)) if data["status"][i]]
                axR.plot(range(len(R)),R,color='C0',label='Range',linewidth=0.5,linestyle='-',marker='o')
                axAz = axR.twinx()
                axAz.plot(range(len(Az)),Az,color='C2',label='Azimuth',linestyle='-',marker='*',linewidth=0.5)
                plt.title(f'{station.id} Radar Coordinates - {stack.id}\n\nmode=({int(modeRC[0])},{int(modeRC[1])})\nRCS0={np.round(RCS0,2)} $\in$({np.round(RCSint[0],2)},{np.round(RCSint[1],2)}) dBm2: {vFlag}')
                plt.xlabel('Dates')
                axR.set_xticks(range(len(R)), xlabels, rotation=90)
                axR.grid(axis='x')
                axR.set_ylabel('Range [px]',color='C0')
                axR.set_xlabel('Dates')
                axAz.set_ylabel('Azimuth [px]',color='C2')
                plt.savefig(outDir+station.id+'_'+stack.id+'_RC.png',dpi=100,bbox_inches='tight')
                plt.close()
                
            
            
            
        # export radar coordinates to csv
        csvPath = outDir+stack.id+'_RC.csv'
        bSpaces = ['','','',''] + ['' for i in range(len(dateList))]
        with open(csvPath, 'w', newline='') as csvFile:
            csvW = csv.writer(csvFile)
            csvW.writerow(['********']+bSpaces)
            csvW.writerow(['This file contains the Radar Coordinates (RC) of the selected reflectors. RCS test is nominally 0. In case 1 or -1 are displayed, please check manually RCS plots.']+bSpaces)
            csvW.writerow(['stack: '+stack.id,'slc path example: '+stack.files[0]]+bSpaces[0:-1])
            csvW.writerow(['Generated on '+timenow()]+bSpaces)
            csvW.writerow(['********']+bSpaces)
            csvW.writerow(['ID','RCS test','Range','Azimuth']+dateList)
            csvW.writerows(stMatrix)
            
        print(f'Radar Coordinates exported to: {csvPath}')
        
        
        # export plot of RCS analysis
        fig,ax = plt.subplots(figsize=(20,4))
        plt.rcParams.update({'font.size': 27})  # Set the font size to 12
        stIDs = [ID[0] for ID in RCSanalysis]
        RCSmeas = [ID[2] for ID in RCSanalysis]
        RCS0vec = [ID[1] for ID in RCSanalysis]
        RCSnominal = [v[3] for v in RCSanalysis if v[3]==0]
        RCSnonominal = [v[3] for v in RCSanalysis if v[3]!=0]
        x = np.arange(len(stIDs))
        ax.scatter(x, RCSmeas,100,marker='o', color='C7')
        ax.scatter(x, RCS0vec,100,marker='*', color='k',label='RCS0')
        bar_width=0.2
        for i, (stID) in enumerate(stIDs):
               plt.errorbar(x[i], RCSanalysis[i][2],yerr=[[RCSanalysis[i][4]], [RCSanalysis[i][4]]], color='C7', linewidth=2, capsize=4)
        i = 0
        plt.errorbar(x[i], RCSanalysis[i][1], yerr=[[RCSanalysis[i][4]], [RCSanalysis[i][4]]], color='C7',label='$RCS\pm 3 \sigma$', linewidth=2, capsize=3)
        bar_width=0.2
        
        ax.set_ylabel('$\mathbf{RCS_{app}}$ [dBm2]')
        ax.set_title(f'RCS_analysis_RC_{stack.id}')
        ax.set_xticks(x)
        ax.set_xticklabels(stIDs)
        ax.set_ylim(RCS0-10,RCS0+10)
        plt.grid(axis='x')
        plt.xticks(rotation=90)
        plt.text(0.01,0.7,f'nominal: {len(RCSnominal)}\nnon-nominal: {len(RCSnonominal)}',color='k',transform=ax.transAxes,
                 bbox=dict(facecolor='white', edgecolor='white', boxstyle='square'))
    
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.3), ncol=3)
        plt.savefig(outDir+'RCSanalysis_'+stack.id+'.png',dpi=100,bbox_inches='tight')
        plt.close()
        

        
    return 
    
        
    
def footPrint(stacks,stations,outDir):
    
    '''
    The function prints the geographic footprint of the radar image, together with the stations' points 
    
    '''
    
    allColors = ['C0','C1','C2','C3','C4','C5','C6','C7']
    
    figA,axA = plt.subplots()
    axA.set_title('All')
    
    for stack,color in zip(stacks,allColors):
        
        stackDir = stack.stackDir
        
        # Load geometry data from swathburst_coverage shapefile
        shpPath = [p for p in Path(stackDir).rglob('../stackburst_coverage.shp')] 
        data = gpd.read_file(shpPath[0]) # Read the Shapefile using geopandas
        geometry = data.geometry
    

        # Iterate over bursts and stations
        stackName = stackDir.split('/')[-3]
        figS,axS = plt.subplots()
        
        axS.set_title(stackName)

        for sb in range(len(data)):
            # Area of Coverage of current iteration
            AoC = geometry[sb]
            x,y = AoC.exterior.xy
            axS.plot(x, y,color='k',linewidth=.5)
            
            
            if sb==0:
                axA.plot(x, y,color=color,linewidth=.5,label=stackName)
            else:
                axA.plot(x, y,color=color,linewidth=.5)
                
            # plt.plot(x, y, color='blue', alpha=0.7, linewidth=2, solid_capstyle='round', zorder=2)
            
            # for each target
            for loop,station in zip(range(0,len(stations)),stations):
            
                if station.descending:
                    stationPoint = sg.Point(station.descending.longitude*180/np.pi, station.descending.latitude*180/np.pi, station.descending.elevation)
                elif station.ascending:
                    stationPoint = sg.Point(station.ascending.longitude*180/np.pi, station.ascending.latitude*180/np.pi, station.ascending.elevation)
                else:
                    raise Exception('Both ascending and descending are not active...')

                if sb==0:
                    axS.scatter(stationPoint.x,stationPoint.y)
                    axS.text(stationPoint.x,stationPoint.y,station.id,fontsize=8)
                    if color=='C0':
                        axA.scatter(stationPoint.x,stationPoint.y)
                        axA.text(stationPoint.x,stationPoint.y,station.id,fontsize=8)
                        
        
        axS.set_xlabel('Longitude [deg]')
        axS.set_ylabel('Latitude [deg]')
        figS.savefig(outDir+os.sep+stackName+'.png',bbox_inches='tight',dpi=100)
        print(f'image exported to {outDir+os.sep+stackName+".png"}')
        plt.close(figS)
    
    
    axA.set_xlabel('Longitude [deg]')
    axA.set_ylabel('Latitude [deg]')
    axA.legend()
    figA.savefig(outDir+os.sep+'All.png',bbox_inches='tight',dpi=100)
    print(f'image exported to {outDir+os.sep+"All.png"}')
    plt.close()
    
    return
                            
                
            
        

        
    
    
    
    


