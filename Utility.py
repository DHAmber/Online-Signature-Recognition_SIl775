import numpy as np
import glob as g
from scipy.ndimage import gaussian_filter1d

'''we are dividing data into two parts positive signatures and negative signature(UsersPositiveSignature[],UsersNegativeSignature[]).
Each User has 20 Positive Signature and 20 Negative Signature. We are temporarly saving this 20 Signature in Signature[] Array.
Each Sinature has multiple rows(Feature Vectors), We are keeping this multiple rows in temp[] array.
So Sequence is like that
UserPositiveSignature=[Signature[Temp[]],Signature[],Signature[],Signature[],Signature[]]
'''

'''Feature Sequence is as follow
  X-coordinate  - scaled cursor position along the x-axis
  Y-coordinate  - scaled cursor position along the y-axis
  Time stamp    - system time at which the event was posted
  Button status - current button status (0 for pen-up and
                  1 for pen-down)
  Azimuth       - clockwise rotation of cursor about the z-axis
  Altitude      - angle upward toward the positive z-axis
  Pressure      - adjusted state of the normal pressure
'''

def load_Signature():
    UsersSignature=[]
    for i in range(1,6):
        imagePath = "SignatureDB/Pos/USER"+str(i)+"_*.txt"
        path=g.glob(imagePath)
        Signatures=[]
        for p in path:
            temp=[]
            with open(p,"r") as f:
                rows=f.readlines()
                for r in rows:
                    t=r.split()
                    temp.append(t)
            Signatures.append(temp)
        UsersSignature.append(Signatures)
    return UsersSignature

def SeprateFeatrures(Sign):
    UsersXCordinate = []
    UsersYCordinate = []
    UsersTimeStamp = []
    UsersButtonStatus = []
    UsersAzimuth = []
    UsersAltitude = []
    UsersPressure = []
    for user in Sign:
        XCordinate = []
        YCordinate = []
        TimeStamp = []
        ButtonStatus = []
        Azimuth = []
        Altitude = []
        Pressure = []
        for Sign in user:
            X = []
            Y = []
            T = []
            B = []
            Az = []
            Al = []
            Pr = []
            for eachPoint in Sign:
                if len(eachPoint)==7:
                    X.append(eachPoint[0])
                    Y.append(eachPoint[1])
                    T.append(eachPoint[2])
                    B.append(eachPoint[3])
                    Az.append(eachPoint[4])
                    Al.append(eachPoint[5])
                    Pr.append(eachPoint[6])
            if(len(X)!=0):
                XCordinate.append(X)
                YCordinate.append(Y)
                TimeStamp.append(T)
                ButtonStatus.append(B)
                Azimuth.append(Az)
                Altitude.append(Al)
                Pressure.append(Pr)
        UsersXCordinate.append(XCordinate)
        UsersYCordinate.append(YCordinate)
        UsersTimeStamp.append(TimeStamp)
        UsersButtonStatus.append(ButtonStatus)
        UsersAzimuth.append(Azimuth)
        UsersAltitude.append(Altitude)
        UsersPressure.append(Pressure)
    return UsersXCordinate,UsersYCordinate,UsersTimeStamp,UsersPressure,UsersAzimuth,UsersAltitude

def NormalizeAndSmmothingData(X,TimeStamp):
    for i in range(len(X)):
        for j in range(len(X[i])):
            X[i][j]=Normalization(X[i][j],TimeStamp[i][j])
            # time=np.array(TimeStamp[i][j],dtype=int)
            # T=time.max()-time.min()
            # Xar=np.array(X[i][j],dtype=int)
            # Xg=Xar.sum()/T
            # X[i][j]=Smoothing(Xar-Xg)
    return X

def Normalization(x,t):
    time = np.array(t, dtype=int)
    T = time.max() - time.min()
    Xar = np.array(x, dtype=int)
    Xg = Xar.sum() / T
    return Smoothing(Xar-Xg)


def Smoothing(Ar):
    Sigma=np.std(Ar)
    result=gaussian_filter1d(Ar,Sigma)
    return result

def PartitionData(data):
   TrainX=[]
   TrainY = []
   TestX=[]
   TestY = []
   for i in range(5):
       counter=0
       testCounter=0
       for d in data:
           if (int(d[0])==i and counter<15):
               TrainX.append(d[1:])
               TrainY.append(d[0])
               counter+=1
           elif(int(d[0])==i and testCounter<4):
               TestX.append(d[1:])
               TestY.append(d[0])
               testCounter+=1
           else:
               pass
   return TrainX,TrainY,TestX,TestY

def Load_UserSign():
    imagePath = "SignatureDB/Dataset/USER1_7.txt"
    path = g.glob(imagePath)
    Signatures = []
    for p in path:
        temp = []
        with open(p, "r") as f:
            rows = f.readlines()
            for r in rows:
                t = r.split()
                temp.append(t)
        #Signatures.append(temp)
    return temp#Signatures

def FetchFeatures(Sign):
    X = []
    Y = []
    T = []
    B = []
    Az = []
    Al = []
    Pr = []
    for eachPoint in Sign:
        if len(eachPoint) == 7:
            X.append(eachPoint[0])
            Y.append(eachPoint[1])
            T.append(eachPoint[2])
            B.append(eachPoint[3])
            Az.append(eachPoint[4])
            Al.append(eachPoint[5])
            Pr.append(eachPoint[6])

    return X,Y,T,B,Az,Al,Pr


