import numpy as np
from scipy.optimize import minimize
import pandas as pd
import math

from datetime import datetime


def date_in_day(x):
    #equant0 is reference point for all other equants
    date0  =  datetime(1580,11,18,1,31)
    date1 = datetime(x[0],x[1],x[2],x[3],x[4])
    d  = date1 -date0
    return d.days + d.seconds/(24*3600)

def get_oppositions(data):
    mars_opposition_data = pd.DataFrame(data, columns= ['Year','Month','Day','Hour','Minute','ZodiacIndex','Degree','Minute.1','Second','LatDegree','LatMinute','ZodiacIndexAverageSun','DegreeMean','MinuteMean','SecondMean'])
    oppositions  = mars_opposition_data['ZodiacIndex']*30 + mars_opposition_data['Degree']  + mars_opposition_data['Minute.1']/60 + mars_opposition_data['Second']/3600 

    return oppositions

def get_times(data):
    mars_opposition_data = pd.DataFrame(data, columns= ['Year','Month','Day','Hour','Minute','ZodiacIndex','Degree','Minute.1','Second','LatDegree','LatMinute','ZodiacIndexAverageSun','DegreeMean','MinuteMean','SecondMean'])
    times = mars_opposition_data[['Year','Month','Day','Hour','Minute']].apply(date_in_day, axis=1)
    return times


def polartoXYCoordinate(r, theta_in_radian):
    return r*np.cos(theta_in_radian) , r*np.sin(theta_in_radian) 


def MarsEquantModel(c,r,e1,e2,z,s, times, oppositions):
    
    cx,cy = polartoXYCoordinate(1,np.radians(c))
    ex,ey = polartoXYCoordinate(e1,np.radians(e2))
    
    equant_dotted_line = (times*s + z) % 360
    
    X_onCir = np.array([0.0]*12)
    Y_onCir = np.array([0.0]*12)
    
    for i in range(0, 12): 

        slope_dotted_line = np.tan(np.radians(equant_dotted_line[i]))
        
        # as equation is ax^2 + bx +d =0 
        a = (1 + slope_dotted_line ** 2)
        b = -2 * cx + 2 * slope_dotted_line * (ey - cy - ex * slope_dotted_line)
        d = (ey - cy - ex *slope_dotted_line)**2 + cx**2 - r**2
        delta = np.sqrt(b**2 - 4 * a * d)
        # roots of equation
        x1 = (-b - delta) / (2 * a)
        x2 = (-b + delta) / (2 * a)
        y1 = ey + (x1 - ex) * slope_dotted_line
        y2 = ey + (x2 - ex) * slope_dotted_line

        if 0 <= equant_dotted_line[i] <= 90 or 270 <= equant_dotted_line[i] <= 360:
            X_onCir[i] = x1 if x1 >= 0 else x2
            Y_onCir[i] = y1 if x1 >= 0 else y2
        else:
            X_onCir[i] = x1 if x1 <= 0 else x2
            Y_onCir[i] = y1 if x1 <= 0 else y2
    
    pred_lng = np.degrees(np.arctan2(Y_onCir,X_onCir))
    act_lng = np.array(oppositions)
    errors = np.subtract(pred_lng , act_lng)
    errors = np.array([i if i <= 180  else 360-i for i in errors])
    errors = np.array([i+360 if i <= -180  else i for i in errors])
    
    return errors, max(abs(errors))

def BestOrbitInnerParams(r,s,times,oppositions):
    
    
    def loss_function(x):
        c, e1, e2, z = x
        _, max_error = MarsEquantModel(c, r, e1, e2, z, s, times, oppositions)
        return max_error
        
        
    #Initial_Guess
    c,e1,e2,z = 10,1,40,60
    
    for i in range(0, 5):
        deg_range,mx = np.linspace(0,360,360),360
        for i in deg_range:
            x0 = (c,e1,e2,i)
            mx_error  = loss_function(x0)
            if mx_error<mx:
                mx = mx_error
                z = i

        e2_range,mx = np.linspace(60,360,360),360
        for i in e2_range:
            x0 = (c,e1,i,z)
            mx_error  = loss_function(x0)
            if mx_error<mx:
                mx = mx_error
                e2 = i

        dist_range,mx = np.linspace(0,5,200),360
        for i in dist_range:
            x0 = (c,i,e2,z)
            mx_error  = loss_function(x0)
            if mx_error<mx:
                mx = mx_error
                e1 = i

        mx = 360
        for i in deg_range:
            x0 = (i,e1,e2,z)
            mx_error  = loss_function(x0)
            if mx_error<mx:
                mx = mx_error
                c = i

    x0=(c,e1,e2,z)
    result = minimize(loss_function,x0,method='Nelder-Mead', options={'xatol' : 1e-5 ,'disp':False, 'return_all' :False})
    
    c, e1, e2, z = result.x
    errors, max_error = MarsEquantModel(c, r, e1, e2, z, s,times, oppositions)

    return c, e1, e2, z, errors,max_error


def BestS(r,times,oppositions):
    
    lbT = 688
    ubT = 686 
    # s = 360/lbt - 360/688
    s_arr = np.linspace(360/lbT,360/ubT, 20)
    sf ,mx, errors_f = [0,360,[]]
    for s in s_arr:
        c,e1,e2,z, errors,max_error = BestOrbitInnerParams(r,s,times, oppositions)
        if mx > max_error:
            sf =s 
            mx = max_error
            errors_f = errors
    
    return sf,errors_f,mx


def BestR(s, times, oppositions):
    r_arr = np.linspace(4,10,60)
    rf,mx,error_f = [0,360,[]]
    for r in r_arr:
        c,e1,e2,z,errors,max_error = BestOrbitInnerParams(r,s,times, oppositions)
        if mx > max_error:
            rf =r 
            mx = max_error
            error_f = errors
    return rf,error_f,mx


def bestMarsOrbitParams(times,oppositions):
    r= 6
    s= 360/687
    i =0
    rf ,sf,mx,error_f = [0,0,360,[]]
    r_temp, s_temp =0,0
    while i<4 and mx >4/60 :
        i = i+1
        r,errors,mx = BestR(s,times,oppositions)
        print("MaxError {} with bestR = {} and S = {}".format(mx,r,s))
        s,errors,mx = BestS(r,times,oppositions)
        print("MaxError {} with R = {} and BestS = {}".format(mx,r,s))
        if r_temp ==r and s_temp==s : break
        r_temp ,s_temp = r,s
    if i==10 and mx>4/60: print("Error not converged with best s and best r")

    c,e1,e2,z, errors,mx = BestOrbitInnerParams(r,s,times, oppositions)
    
    return r,s,c,e1,e2,z,errors,mx 








