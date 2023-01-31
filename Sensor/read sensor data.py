#import packages
import pandas as pd
import numpy as np
import serial
import serial
import time
import datetime
import threading
import matplotlib.pyplot as plt


TouchBoardData = []
countTimeEvent = threading.Event()

# generate array [256, 512]. Set 1 in place where sensor X and sensor Y were touched within 0.1 sec  
def create_scratch(df):

        maxDifference = 0.1
        scratch = np.zeros(( 256, 512))

        dfX = df[df['Board'] == 'X']
        dfY = df[df['Board'] == 'Y']

        if len(dfX) == len(dfY):
                loopIndex = len(dfY)
        elif  len(dfX) > len(dfY):
                loopIndex = len(dfY)
        else: 
                loopIndex = len(dfX)

        electrodeX = 0
        electrodeY = 0

        for i in range(1, loopIndex):
                absDifference =  abs(float(dfX['Time'].iloc[i]) - float(dfY['Time'].iloc[i]))
                if absDifference > maxDifference:
                        continue
                if str(dfX['Value'].iloc[i]).count('1') == 1:
                        electrodeX = str(dfX['Value'].iloc[i]).find('1')
                if str(dfX['Value'].iloc[i]).count('1') == 1:
                        electrodeY = str(dfY['Value'].iloc[i]).find('1')
                if electrodeX > 0 and electrodeY > 0:
                        scratch[(electrodeY * 19 )][((electrodeX * 19) + 255)] = 1
                        elctroodeX  = 0 
                        electrodeY = 0

        return scratch

# time count for sensors ( 30 second )
def countTime():
        countTimeEvent.set()
        print('Event started')
        time.sleep(30)
        countTimeEvent.clear()
        print('Event finished')

#read sensor data
def readSensor(usbPort, label, sensorData):
        countTimeEvent.wait()
        if countTimeEvent.is_set():
                readBool = True                
                try:
                        ser = serial.Serial(
                        port = usbPort, 
                        baudrate = 9600)
                except serial.SerialException:
                        print('Problem z portem' + label)
                        readBool = False
        while readBool:
                if countTimeEvent.is_set():
                        line = str(ser.readline(), 'utf-8')
                        print(line)
                        TouchedTime = str(time.time())
                        sensorData.append( [TouchedTime, line , label ])
                        
                        # print(TouchBoardData)
                else:
                        return None

# Thread for reading sensor X
TouchBoardX = threading.Thread( name='AxisX',
                                target=readSensor,
                                args=('/dev/tty.usbmodem1421', 'X', TouchBoardData ))
# Thread for reading sensor Y
TouchBoardY = threading.Thread( name='AxisY',
                                target=readSensor,
                                args=('/dev/tty.usbmodem1411', 'Y', TouchBoardData )) 

#starting main execution to save output to data and display touched points
countTimeThread = threading.Thread( target=countTime )
countTimeThread.start()
TouchBoardX.start()
TouchBoardY.start()
countTimeThread.join()
print(TouchBoardData)
df2 = pd.DataFrame(np.array(TouchBoardData) , columns=['Time', 'Value', 'Board' ])
print(df2)
nazwa_pliku = 'plik_' + str(time.time()) + '.csv'
df2.to_csv(nazwa_pliku)
scratch_dots = create_scratch(df2)
print(scratch_dots)
plt.imshow(scratch_dots)
plt.show() 
TouchBoardX.join()
TouchBoardY.join()
