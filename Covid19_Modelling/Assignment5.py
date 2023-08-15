import pandas as pd
import numpy as np
import datetime
import math
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


#Global Variables

N = 70000000
eff = 0.66
alpha = 1/5.8
gamma = 1/5





def CIR(t,CIR0):
    Tt0 = np.mean(Tested[0:7])  
    if(t<195):
        Ttt = np.mean(Tested[t:t+7])
    else:
        Ttt=Tt0
    if Ttt=="nan":
        print("sd")
    return CIR0*(Tt0/Ttt)


def COVID19Cases(params):
    def SEIR_model(beta,t):
        x = beta*S[t]*I[t]/N

        try:
            delV = Vaccinated[t+1] - Vaccinated[t]
        except:
            delV = 200000
        y = eff*delV
        z = gamma*I[t]
        w = alpha*E[t]
        v = delW(t)
        dS[t] = -x -y +v
        dE[t] = x-w
        dI[t] = w-z
        dR[t] = z-v+y
    
    
    def delW(t):
        if t<=30:
            return Vaccinated[0]/30
        elif t>30 and t<180:
            return dR[t-30]
        else :
            try:
                delV = Vaccinated[t-180+1] - Vaccinated[t-180]
            except:
                delV= 200000
            return dR[t-180] + eff*delV
    
    def confirmedCases(t):
        return np.mean(Confirmed[t-7:t])/7
    
    
    
    def loss_function(params):
        
        S[0],E[0],I[0],R[0],beta,CIR0=params
        
        for t in range(1,n_days):
            SEIR_model(beta,t-1)
            S[t] = min(N,max(0,S[t-1]+dS[t-1]))
            E[t] = min(N,max(0,E[t-1]+dE[t-1]))
            I[t] = min(N/2,max(0,I[t-1]+dI[t-1]))
            R[t] = min(N/2,max(0,R[t-1]+dR[t-1]))
        i = np.zeros(n_days)
        
        for t in range(n_days-1):
            i[t] = alpha*E[t]/CIR(t,CIR0)
            
        sum_t=0
        for t in range(n_days-7):
            
            dit = math.log10(np.mean(i[t:t+7])/(7))
            dct = math.log10(np.mean(Confirmed[t:t+7])/(7))
        
            sum_t += (dit-dct)**2
        return sum_t/42
    
            
        
    def gradient_function(p):
        grad = [0]*6  #S,E,I,R,beta,CIR
        pred_p = np.zeros(6)
        mn_loss = loss_function(p)
        pr_loss= mn_loss
        for i in (-1,1):
            pred_p[0]= p[0]+i
            
            for j in (-1,1):
                pred_p[1] = p[1] +j
                
                for k in (-1,1):
                    pred_p[2] = p[2] +k
                    
                    for l in (-1,1):
                        if 0.156*N <=p[3]+l and p[3] + l <0.36*N:
                            pred_p[3] = p[3] +l
                        for m in (-0.01,0.01):
                            pred_p[4] = p[4] +m
                            
                            for n in (-0.1,0.1):
                                if 12 <=p[5]and p[5]<=30:
                                    pred_p[5]=p[5] +n
                                pr_loss = loss_function(pred_p)
                                if mn_loss > pr_loss:
                                    mn_loss = pr_loss
                                    grad = [i,j,k,l,m,n]
        return np.array(grad)
        
    n_days= (datetime.datetime(2021,4,26)-datetime.datetime(2021,3,16)).days +8
    S,E,I,R  = np.zeros(n_days),np.zeros(n_days),np.zeros(n_days),np.zeros(n_days)
    dS,dE,dI,dR =np.zeros(n_days),np.zeros(n_days),np.zeros(n_days),np.zeros(n_days)
    S[0],E[0],I[0],R[0],beta,CIR0 = params
    return gradient_function(params),loss_function(params)
        
        
        
def gradientDescent(p,mx_itr):
    
    S,E,I,R,beta,CIR=p
    if S != (N- E- I-R): 
        print("Population is not conserved")
        return
    itr=0
    loss=100
    p_final = p
    loss_final = loss
    while itr<mx_itr and loss>=0.01:
        itr+=1
        gradient,loss = COVID19Cases(p)
        p = p + gradient/itr
        if itr%10000==0:
            p[1] = int(p[1])
            p[2] = int(p[2])
            p[3] = int(p[3])
            p[0] = N - p[1]-p[2]-p[3]
            print("Loss is {} , itr is {} and p is {}".format(loss_final,itr,p))
            print("-------------------------------")
        if loss_final > loss:
            loss_final = loss
            p_final = p
    
    p_final[1] = int(p_final[1])
    p_final[2] = int(p_final[2])
    p_final[3] = int(p_final[3])
    p_final[0] = N- p_final[1]- p_final[2]- p_final[3]
    return loss_final,p_final
    

def openLoopCOntrol(params,rate=1):
    def SEIR_model(beta,t):
        x = beta*S[t]*I[t]/N

        try:
            delV = Vaccinated[t+1] - Vaccinated[t]
        except:
            delV = 200000
        y = eff*delV
        z = gamma*I[t]
        w = alpha*E[t]
        v = delW(t)
        dS[t] = -x -y +v
        dE[t] = x-w
        dI[t] = w-z
        dR[t] = z-v+y


    def delW(t):
        if t<=30:
            return Vaccinated[0]/30
        elif t>30 and t<180:
            return dR[t-30]
        else :
            try:
                delV = Vaccinated[t-180+1] - Vaccinated[t-180]
            except:
                delV= 200000
            return dR[t-180] + eff*delV
        
    n_days=  (datetime.datetime(2021,12,31)-datetime.datetime(2021,3,8)).days
    S,E,I,R  = np.zeros(n_days),np.zeros(n_days),np.zeros(n_days),np.zeros(n_days)
    dS,dE,dI,dR =np.zeros(n_days),np.zeros(n_days),np.zeros(n_days),np.zeros(n_days)
    S[0],E[0],I[0],R[0],beta,CIR0 = params
    i = np.zeros(n_days)
    beta = beta*rate
    
    
    
    
    for t in range(1,n_days-1):
        
        SEIR_model(beta,t-1)
        S[t]=max(0,S[t-1]+dS[t-1])
        E[t]=max(0,E[t-1]+dE[t-1])
        I[t]=max(0,I[t-1]+dI[t-1])
        R[t]=max(0,R[t-1]+dR[t-1])
        
        
        i[t]=alpha*E[t-1]/CIR(t-1,CIR0)
    return i,S


def ClosedLoopCOntrol(params):
    def SEIR_model(beta,t):
        x = beta*S[t]*I[t]/N

        try:
            delV = Vaccinated[t+1] - Vaccinated[t]
        except:
            delV = 200000
        y = eff*delV
        z = gamma*I[t]
        w = alpha*E[t]
        v = delW(t)
        dS[t] = -x -y +v
        dE[t] = x-w
        dI[t] = w-z
        dR[t] = z-v+y


    def delW(t):
        if t<=30:
            return Vaccinated[0]/30
        elif t>30 and t<180:
            return dR[t-30]
        else :
            try:
                delV = Vaccinated[t-180+1] - Vaccinated[t-180]
            except:
                delV= 200000
            return dR[t-180] + eff*delV
        
    
    n_days=  (datetime.datetime(2021,12,31)-datetime.datetime(2021,3,8)).days
    S,E,I,R  = np.zeros(n_days),np.zeros(n_days),np.zeros(n_days),np.zeros(n_days)
    dS,dE,dI,dR =np.zeros(n_days),np.zeros(n_days),np.zeros(n_days),np.zeros(n_days)
    S[0],E[0],I[0],R[0],beta,CIR0 = params
    i = np.zeros(n_days)
    
    
    b=beta
    for t in range(1,n_days):
        SEIR_model(b,t-1)
        S[t]=max(0,S[t-1]+dS[t-1])
        E[t]=max(0,E[t-1]+dE[t-1])
        I[t]=max(0,I[t-1]+dI[t-1])
        R[t]=max(0,R[t-1]+dR[t-1])
        i[t]=alpha*E[t-1]/CIR(t-1,CIR0)
        if i[t] < 10000:
            b = beta
        elif i[t] < 25000:
            b = 2*beta/3
        elif i[t] < 100000:
            b = beta/2
        else:
            b = beta/3
        
    return i,S
    


def plot(i,S,title="Beta=1 Covid19 Modelling"):
        
    n = i.shape[0]
    m = Confirmed.shape[0]
    ACases = np.zeros(m)
    for r in range(1,m):
        ACases[r]=Confirmed[r]-Confirmed[r-1]
    plt.figure(figsize=(15,10))
    plt.title(title)
    plt.xlabel("Days")
    plt.ylabel("NewCases ")
    plt.plot(range(n),i)
    plt.plot(ACases)
    plt.savefig(title+".png")
    
    plt.figure(figsize=(15,10))
    
    plt.plot(range(n),S)
    plt.title(title)
    plt.xlabel("Days")
    plt.ylabel("Susceptible cases Evolution " + title )
    plt.savefig("Susceptible cases Evolution " + title +".png")
    
    


if __name__ == "__main__":
    df = pd.read_csv("../COVID19_data.csv")

    Confirmed = df[(df['Date']>'2021-03-08')]['Confirmed'].values
    Vaccinated = df[(df['Date']>'2021-03-08')]['First Dose Administered'].values
    Tested   =  df[(df['Date']>'2021-03-08')]['Tested'].values

    print("\nDays_Data_Considered: {}".format(Confirmed.shape[0]))

    print("-----------------Model Training Started--------------")
    params = [0.7*N,0.01*N,0.01*N,0.28*N,0.8,20]
    print("\nInitial_parameters i.e S0,I0,E0,R0,beta,CIR0 are : {}".format(params))
    
    loss,params = gradientDescent(params,100000)
    print("\nAfer training,Loss is {}, S0,I0,E0,R0,beta,CIR0  are :{}".format(loss,params))
    print("\n-----------------Model Training Completed--------------")

    print("\n-----------------Plot for OpenControl Started--------------")

    for rate in [1,2/3,1/2,1/3]:
        i,S= openLoopCOntrol(params,rate=rate)
        plot(i,S,title="Beta={}, possible_Cases".format(rate))
        print("\nPlot generated for beta= {}".format(rate))

    print("\n-----------------Plot for OpenControl Completed--------------")


    i,S= ClosedLoopCOntrol(params)
    plot(i,S,title="Closed ControlLoop Cases")
    print("Plot generated for Closed Control Loop")




