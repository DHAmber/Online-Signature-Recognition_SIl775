import numpy as np

def dtw(x,y):
    #x=[1,3,4,9,8,2,1,5,7,3]
    #y=[1,6,2,3,0,9,4,3,6,3]
    m=len(x)
    n=len(y)
    x=np.array(x,dtype=float)
    y = np.array(y, dtype=float)
    d=np.zeros((m,n),dtype=float)
    for i in range(len(x)):
        for j in range(len(y)):
            if (i==0 and j==0):
                d[i][j]=cost(x[i],y[j])
            else:
                if(i==0 and j!=0):
                    d[i][j]=cost(x[i],y[j])+d[i][j-1]
                elif(i!=0 and j==0):
                    d[i][j]=cost(x[i],y[j])+d[i-1][j]
                else:
                    d[i][j]=cost(x[i],y[j])+min(d[i-1][j],d[i][j-1],d[i-1][j-1])
    #print(d)
    return d[m-1][n-1]
    #print('Final cost: ', d[m-1][n-1])



def cost(x,y):
    return abs(x-y)

#dtw()