import numpy as np
import Utility as U
import DTW as dtw
from sklearn.ensemble import AdaBoostClassifier
#import glob as g


def Match(adaBoost, SampleX, SampleY, SampleP):
    Sign = U.Load_UserSign()
    x, y, t, b, Az, Al, p = U.FetchFeatures(Sign)
    x = U.Normalization(x, t)
    y = U.Normalization(y, t)
    b = U.Normalization(b, t)
    Az = U.Normalization(Az, t)
    Al = U.Normalization(Al, t)
    p = U.Normalization(p, t)

    dtwA = np.zeros((5, 3))
    for i in range(len(SampleX)):
        X_dtw = dtw.dtw(SampleX[i], x)
        Y_dtw = dtw.dtw(SampleY[i], y)
        P_dtw = dtw.dtw(SampleP[i], p)

        dtwA[i][0] = X_dtw
        dtwA[i][1] = Y_dtw
        dtwA[i][2] = P_dtw
        Y = adaBoost.predict(dtwA)
        print(Y)


if __name__ == '__main__':

    Signature = U.load_Signature()       #loads all signatures
    UsersXCordinate, UsersYCordinate, UsersTimeStamp,UsersPressure,UsersAzimuth,UsersAltitude = U.SeprateFeatrures(Signature)

    UsersXCordinate=U.NormalizeAndSmmothingData(UsersXCordinate,UsersTimeStamp)
    UsersYCordinate = U.NormalizeAndSmmothingData(UsersYCordinate, UsersTimeStamp)
    UsersPressure = U.NormalizeAndSmmothingData(UsersPressure, UsersTimeStamp)
    UsersAzimuth = U.NormalizeAndSmmothingData(UsersAzimuth, UsersTimeStamp)
    UsersAltitude = U.NormalizeAndSmmothingData(UsersAltitude, UsersTimeStamp)

    SampleX=[]                #To take  features of first signature's  for each user
    SampleY=[]
    SampleP=[]
    SampleA = []
    SampleAl = []
    for i in range(len(UsersXCordinate)):
        SampleX.append(UsersXCordinate[i].pop())
        SampleY.append(UsersYCordinate[i].pop())
        SampleP.append(UsersPressure[i].pop())
        SampleA.append(UsersAzimuth[i].pop())
        SampleAl.append(UsersAltitude[i].pop())

    m=len(UsersXCordinate)*len(UsersXCordinate[0])
    DTWArray = np.zeros((m, 4), dtype=object)
    a=0
    for i in range(len(UsersXCordinate)):
        X_dtw=0.0
        Y_dtw=0.0
        P_dtw=0.0
        A_dtw=0.0
        Al_dtw = 0.0
        for j in range(len(UsersXCordinate[i])-1):
            X_dtw=dtw.dtw(SampleX[i],UsersXCordinate[i][j+1])
            Y_dtw=dtw.dtw(SampleY[i],UsersYCordinate[i][j+1])
            P_dtw=dtw.dtw(SampleP[i],UsersPressure[i][j+1])
            #A_dtw = dtw.dtw(UsersAzimuth[i][j], UsersAzimuth[i][j])
            #Al_dtw = dtw.dtw(UsersAltitude[i][j], UsersAltitude[i][j])

            DTWArray[a][0]=i
            DTWArray[a][1] = X_dtw
            DTWArray[a][2] = Y_dtw
            DTWArray[a][3] = P_dtw
            #DTWArray[a][4] = A_dtw
            #DTWArray[a][4] = Al_dtw
            a=a+1

    TrainX,TrainY,TestX,TestY=U.PartitionData(DTWArray)
    adaBoost=AdaBoostClassifier(n_estimators=100, base_estimator= None,learning_rate=1, random_state = 1)
    adaBoost.fit(TrainX,TrainY)
    YPred=adaBoost.predict(TestX)

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(TestY, YPred)
    accuracy = float(cm.diagonal().sum()) / len(TestY)
    #print("\nAccuracy :", accuracy)
    Match(adaBoost,SampleX,SampleY,SampleP)







