print('\n')
print('Caroline Radar-Coding Toolbox (based on GECORIS v.1.0)')
# print('Copyright (c) 2021 Richard Czikhardt, czikhardt.richard@gmail.com')
print('Dept. of Geosciences and Remote Sensing, Delft University of Technology')
print('-----------------------------------------------------------------------')
print('License: GNU GPL v3+')
print('-----------------------------------------------------------------------')


import sys
import os
import glob
import numpy as np
from gecoris import ioUtils, plotUtils, atmoUtils, dorisUtils
from functions import s1UtilsPaolo as s1p
import time
from tqdm import tqdm

stackType = np.dtype({'names':('acqDate','acqTime','status','RCS','SLC','azimuth','range',
                               'dAz','dR','dist','IFG'),
                      'formats':('U10','U10','?','f8','c16','f8','f8',
                                 'f8','f8','f8','c16')})



def main(parms):
    

    # unpack:
    stationLog = parms['stationLog']
    stacksLog = parms['stackLog']
    outDir = parms['outDir']
    posFlag = parms['precisePosFlag']
    plotFlag = parms['plotFlag']
    fullStack = parms['fullStack']
    cropFlag = parms['cropFlag']
    #
    
    # --- Reflectors Are Loaded
    print('Initializing reflectors...')
    if stationLog.lower().endswith('.json'):
        stations = ioUtils.load_station_log(stationLog)
    elif stationLog.endswith('.csv'):
        stations = ioUtils.fromCSV(stationLog)
    else:
        print('Unknown station log file format. Use .json or .csv.')
        raise

    print('Loading SAR data stacks...')
    if stacksLog.lower().endswith('.json'):
        stacks = ioUtils.load_stacks(stacksLog)
    elif stacksLog.lower().endswith('.csv'):
        stacks = ioUtils.stacks_fromCSV(stacksLog)
    else:
        print('Unknown stacks log file format. Use .json or .csv.')

    # load data:
    SLCstacks = []
    for stack in stacks: 
        stack.readData(stations,fullStack=fullStack,crop=cropFlag) # decide if you want to iterate over two or if you want to import metadata specifying bursts. For the moment we select only the first reflector of the list
        ioUtils.toJSON(stack, parms['outDir']) # save to JSON
    dorisUtils.footPrint(stacks,stations,outDir)


    # check if output directory exists and load already processed:
    if not os.path.exists(outDir):
        os.makedirs(outDir)
        logs = []
    else:
        logs = glob.glob(outDir + os.sep +"*.json")
    #
    
    print(str(len(stations))+ ' reflectors on ' +str(len(stacks))
          + ' stacks to process.')
    # iterate over list indices to modify objects:
    
    for stack in stacks:
        print(f'****** stack ID: {stack.id}')
        slcIdx = -1
        
        for acqDate in tqdm(['00000000']+stack.acqDates+['99999999']):
            
            if acqDate != '00000000' and acqDate != '99999999':
                dateIdx = stack.acqDates.index(acqDate)
                file = stack.files[dateIdx]
                # print(f'Reading SLC for date {acqDate}')
                if cropFlag:
                    SLC = s1p.readSLC(file,stack.masterMetadata,0,method='coregCrop',
                                              deramp = True)
                else:    
                    SLC = s1p.readSLC(file,stack.masterMetadata,0,method='coregSingle',
                                              deramp = True)
                
            for i in range(len(stations)):
                
                # perform only one tie at the beginning
                if acqDate == '00000000':
                    stations[i].data = np.zeros(len(stack.acqDates), dtype=stackType)
                    continue
                
                # perform only one time at the end
                if acqDate == '99999999':
                    ts = {'id': stack.id, 'data': stations[i].data, 'metadata': stack.masterMetadata,
                          'stack': stack,'type':'coreg','zas':zas}
                    stations[i].stacks.append(ts)
                    
                    continue
                    
                    
                # check if analysis already performed:
                inJSON = [q for q in logs 
                          if stations[i].id+'.json' in q.split(os.sep)[-1]]
                if inJSON:
                    print('Station '+stations[i].id+ ' already analyzed, updating.')
                    stations[i] = ioUtils.fromJSON(inJSON[0])
                    stations[i].updateStack(stack, ovsFactor=1,
                                            posFlag=posFlag)
                else:
                

                    zas = stations[i].addStack(stack, ovsFactor=1, 
                                             posFlag=posFlag, plotFlag=plotFlag,
                                             outDir=outDir,SLC=SLC,acqDate=acqDate,slcIdx=slcIdx)

            slcIdx += 1
                    
        
    
    for i in range(len(stations)):
        stations[i].print_all_timeseries(outDir)
        print('Removing outliers.')
        stations[i].detectOutliers()
        
        print('Performing RCS and SCR analysis.')
        stations[i].RCSanalysis()
        
        if posFlag and plotFlag > 1:
            # print('Plotting ALE.')
            for stack in stacks:
                stations[i].plotALE(stack.id,outDir)
            print(f'Current station: {stations[i].id}')
            stations[i].plotALE_TS(outDir)
        if plotFlag > 0:
            # print('Plotting RCS time series.')
            stations[i].plotRCS(outDir)
        
        # print('Exporting to JSON dumps.')
        stations[i].toJSON(outDir)
        stations[i].statsToJSON(outDir)
    print('Exporting Radar Coordinates to CSV...')
    dorisUtils.RCexport(stations,stacks,outDir+os.sep+'RadarCoordinates'+os.sep)

    
    if len(stations) > 2 and plotFlag > 0:
        print('Generating network plots.')
        # plotUtils.plotNetworkALE(stations, outDir+os.sep+'network_ALE.png')
        #plotUtils.plotNetworkRCS(stations, outDir+os.sep+'network_RCS.png')
        # plotUtils.plotNetworkSCR_hz(stations, outDir+os.sep+'network_SCR.png')
    #
    print('Done.')
    print('In case of any inquries, please contact bazzocchip@gmail.com')


def parse_parms(parms_file):
    print('looking for a .parms file at: '+parms_file)
    try:
        with open(parms_file,'r') as inp:
            try:
                parms = eval(inp.read())
            except:
                print("Something wrong with parameters file.")
                raise
        return parms
    except:
        print("Specified parameters file not found.")
        raise
        
        
if __name__ == "__main__":
    # load parameters:
    if len(sys.argv) > 1:
        parms = parse_parms(sys.argv[1])
        main(parms)
    else:
        print('Not enough input arguments!')