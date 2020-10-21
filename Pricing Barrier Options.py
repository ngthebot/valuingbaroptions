# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 18:43:36 2020

@author: Nicolas
"""
import numpy as np
import os
os.chdir('C:/Users/jcao10/Pictures/OneDrive/Documentos/UDESA/2020 1er semestre/ingeniería financiera/Pablo Macri')
from polynomial_regression import *
from scipy.stats import norm


S0 = 10
K = 10
risk_free = 0.2
sigma = 0.25
M = 10000
N = 50
T = 0.2
mu = 0 
B = 8.5
kn = "out"
di = "down"
delta_t = T/N
discountfactor = np.exp(-risk_free * delta_t)




def simulateGeometricBrownianMotionALL(M, N, S0, T, mu, sigma):
    """
     Simulate geometric Brownian motion

    INPUT:
        M  : Number of trajectories
        N  : Number of time steps
        S0 : initial value S(t0) = S0
        T  : Length of the simulation interval [t0,t0+T]
         mu,sigma : parameters of the process
    OUTPUT:
        S_plus  : Simulation of M trajectories of geometric Brownian motion
        S_minus  : Simulation of M trajectories of geometric Brownian motion with inverse sigma

    """
    ## Size of the integration step
    delta_t = T/N

    
    ## Simulate M trajectories 
    X = np.random.randn(M, N)  # Gaussian white noise
    e_minus = np.exp((mu - 0.5 * sigma**2) * delta_t - sigma * np.sqrt(delta_t) * X)
    S_minus = np.cumprod(np.c_[S0 * np.ones((M,1)), e_minus], 1)
    e_plus = np.exp((mu - 0.5 * sigma**2) * delta_t + sigma * np.sqrt(delta_t) * X)
    S_plus = np.cumprod(np.c_[S0 * np.ones((M,1)), e_plus], 1)
    return S_minus, S_plus

#############################################################################
##EUROPEAS POR MONTECARLO: PUEDE SER PLAIN VANILLA O BARRIER


def price_BarrierEuropeanCallMC(S0, K, risk_free, sigma, M, N, T, mu, B, kn, di):
    '''
    Input:
    S0: Underlying price in t=0
    K: Strike
    risk_free: tasa libre de riesgo
    sigma: varianza
    M: Number of Paths
    T: Length from 0 to t
    N: Number of partitions for T 
    B: Precio de Barrera
    kn : tipo de barrera (out, in)
    di: tipo de barrera (up, down)
    
    SI KN Y DI SON == 0, ENOTONCES ES PLAIN VANILLA
    CASO CONTRARIO PUEDE SER UNA BARRIER OUT-DOWN, OUT-UP,IN-UP, IN-DOWN
    '''
    delta_t = T/N
    S = simulateGeometricBrownianMotionALL(M, N, S0, T, mu, sigma)[1]
    sim = S[:,N]
    payoff_barrier = np.where(sim < K, 0, sim - K)
    discountfactor = np.exp(-risk_free * delta_t)
    if (kn == 'out' and di == 'down'):
        ##Europea con Barrier Knock-out-down
        for i in range(0,M,1) :    
            if (any(S[i,1:] <= B)):
                payoff_barrier[i,]=0
    elif (kn == 'in' and di == 'up'):
        ##Europea con Barrier Knock-in-up   
        for i in range(0,M,1) :
            if (all(S[i,1:] < B)):
                payoff_barrier[i,]=0
    elif (kn == 'in' and di == 'down'):
        ##Europea con Barrier Knock-in-down   
        for i in range(0,M,1) :
            if (all(S[i,1:] > B)):
                payoff_barrier[i,]=0
    elif (kn == 'out' and di == 'up'):
        ##Europea con Barrier Knock-out-up  
        for i in range(0,M,1) :    
            if (any(S[i,1:] > B)):
                payoff_barrier[i,]=0
    pricebarrier_MC = discountfactor*np.mean(payoff_barrier)
    stdev_barrierMC = discountfactor*np.std(payoff_barrier)/np.sqrt(M)
    return pricebarrier_MC, stdev_barrierMC, payoff_barrier

price_BarrierEuropeanCallMC(S0, K, risk_free, sigma, M, N, T, mu, B, kn, di)


def price_BarrierEuropeanAVCallMC(S0, K, risk_free, sigma, M, N, T, mu, B, kn, di):
    '''Input:
    S0: Underlying price in t=0
    K: Strike
    risk_free: tasa libre de riesgo
    delta_t: T/N
    sigma: varianza
    M: Number of Paths
    T: Length from 0 to t
    N: Number of partitions for T 
    B: Precio de Barrera
    kn : tipo de barrera (out, in)
    di: tipo de barrera (up, down)
    
    SI KN Y DI SON == 0, ENOTONCES ES PLAIN VANILLA
    CASO CONTRARIO PUEDE SER UNA BARRIER OUT-DOWN, OUT-UP,IN-UP, IN-DOWN
    '''
    delta_t = T/N
    S_plus = simulateGeometricBrownianMotionALL(M, N, S0, T, mu, sigma)[1]
    S_minus = simulateGeometricBrownianMotionALL(M, N, S0, T, mu, sigma)[0]
    sim_plus = S_plus[:,N]
    sim_minus = S_minus[:,N]
    payoff_barrier_plus = np.where(sim_plus < K, 0, sim_plus - K)
    payoff_barrier_minus = np.where(sim_minus < K, 0, sim_minus - K)
    discountfactor = np.exp(-risk_free * delta_t)
    if (kn == 'out' and di == 'down'):
        ##Europea con Barrier Knock-out-down
        for i in range(0,M,1) :    
            if (any(S_plus[i,1:] <= B)):
                payoff_barrier_plus[i,]=0
            if (any(S_minus[i,1:] <= B)):
                payoff_barrier_minus[i,]=0
    elif (kn == 'in' and di == 'up'):
        ##Europea con Barrier Knock-in-up   
        for i in range(0,M,1) :
            if (all(S_plus[i,1:] < B)):
                payoff_barrier_plus[i,]=0
            if (all(S_minus[i,1:] < B)):
                payoff_barrier_minus[i,]=0
    elif (kn == 'in' and di == 'down'):
        ##Europea con Barrier Knock-in-down   
        for i in range(0,M,1) :
            if (all(S_plus[i,1:] > B)):
                payoff_barrier_plus[i,]=0
            if (all(S_minus[i,1:] > B)):
                payoff_barrier_minus[i,]=0
    elif (kn == 'out' and di == 'up'):
        ##Europea con Barrier Knock-out-up  
        for i in range(0,M,1) :    
            if (any(S_plus[i,1:] > B)):
                payoff_barrier_plus[i,]=0
            if (any(S_minus[i,1:] > B)):
                payoff_barrier_minus[i,]=0
    payoff_barrierAV = 0.5*(payoff_barrier_plus + payoff_barrier_minus)
    pricebarrier_AVMC = discountfactor*np.mean(payoff_barrierAV)
    stdev_barrierAVMC = discountfactor*np.std(payoff_barrierAV)/np.sqrt(M)
    return pricebarrier_AVMC, stdev_barrierAVMC

price_BarrierEuropeanAVCallMC(S0, K, risk_free, sigma, M, N, T, mu, B, kn, di)

##Classic European Black Scholes
d1 =  (np.log ( S0 / K) + (risk_free + 0.5*(sigma**2)) * T) / (sigma * (T **(1/2)))
d2 = d1 - sigma * np.sqrt(T)
Nd1 = norm.cdf(d1)
NNd1 = norm.cdf(-d1)
Nd2 = norm.cdf(d2)
NNd2 = norm.cdf(-d2)
call_price = S0*Nd1-K*discountfactor*Nd2

##Control Variates for Barrier Options
S = simulateGeometricBrownianMotionALL(M, N, S0, T, mu, sigma)[1]
sim = S[:,N]

Bk = price_BarrierEuropeanCallMC(S0, K, risk_free, sigma, M, N, T, mu, B, kn, di)[2] #Barrier Option payoff from the Monte Carlo price_BarrierEuropeanCallMC line 62
MCk = np.where(sim < K, 0, sim - K) #Montecarlo of a plain vanilla (Control Variate)
Pk = call_price #Black Scholes from control variate
Z = (Bk-(MCk - Pk)) #New variable Z

pricebarrierCV_MC = discountfactor*np.mean(Z)
stdev_CVMC = discountfactor*np.std(Z)/np.sqrt(M)

pricebarrierCV_MC, stdev_CVMC


##############################################################################
#VALUAMOS POR MONTECARLO CASO AMERICANO
K = 10 ##Strike
S0 = 10 ##Precio en t0
r = 0.2 ##Risk-free
M = 10000  # number of paths
N = 50   # number of discretized times
T = 0.2 ##Largo del periodo 
B = 15 ##Barrier
kn = 'out' ##Tipo de Barrier puede ser out o in
di = 'up' ##Tipo de Barrier puede ser up o down
mu = 0 
sigma = 0.25


##AMERICANA POR MONTECARLO: PLAIN O CON BARRIER
def ValuacionOpcionAmericanaPut(M, N, S0, T, mu, sigma, r, K, B, kn, di):
    '''
     S0: Underlying price in t=0
    K: Strike
    r: tasa libre de riesgo
    sigma: varianza
    mu: Esperanza de S
    M: Number of Paths
    T: Length from 0 to t
    N: Number of partitions for T 
    B: Precio de Barrera
    kn : tipo de barrera (out, in)
    di: tipo de barrera (up, down)
    
    SI KN Y DI SON == 0, ENOTONCES ES PLAIN VANILLA
    CASO CONTRARIO PUEDE SER UNA BARRIER OUT-DOWN, OUT-UP,IN-UP, IN-DOWN
    '''
    dt=T/N
    df = np.exp(-r*dt)
    S = simulateGeometricBrownianMotionALL(M, N, S0, T, mu, sigma)[1]
    t = N-1
    
    i = 1
    j = 0 
    jj = j
    
    #ARMAMOS LA MATRIZ PARA LOS IN
    if kn == 'in':
        Matriz_B = np.zeros((M, N+1))

        if  di == 'up':
            for i in range (0, M, 1):
                for j in range (0, N+1, 1):
                    if S[i, j] >= B:
                         for jj in range(j, N+1, 1):
                            Matriz_B[i, jj] = 1                   
        elif di == 'down':
            for i in range (0, M, 1):
                for j in range (0, N+1, 1):
                    if S[i, j] <= B:
                        for jj in range(j, N+1, 1):
                            Matriz_B[i, jj] = 1                                             
    
    #ARMAMOS LA MATRIZ PARA LOS OUT
    elif kn == 'out':
        Matriz_B = np.zeros((M, N+1)) +1
        if  di == 'up':
            for i in range (0, M, 1):
                for j in range (0, N+1, 1):
                    if S[i, j] >= B:
                        for jj in range(j, N+1, 1):
                            Matriz_B[i, jj] = 0  
        elif  di == 'down':
            for i in range (0, M, 1):
                for j in range (0, N+1, 1):
                    if S[i, j] <= B:
                        for jj in range(j, N+1, 1):
                            Matriz_B[i, jj] = 0
                            
    #Se anula la matriz para los casos que no tengan Barrier
    else:
        Matriz_B = np.zeros((M, N+1)) +1
        
    #DESPUES DE DEFINIR EN QUE CASO ESTAMOS CON EL LOOP, OBTENEMOS LA MATRIZ MULTIPLICADORA                        
    Matriz_B = np.delete(Matriz_B, 0,1)

    CF = np.zeros((M, N))      ###ERROR
    CV = np.zeros((M, N))
    CF[:, t] = np.maximum(K-S[:, N],0) #genera cashflows en el período i  
    CF[:, t] = Matriz_B[:, t] * CF[:, t]#Descartamos los cashflows en el ultimo periodo en relacion a la barrera
    CV[:, t] = CF[:, t]
    
    i = 0
    f = t-1
    S = np.delete(S, 0,1)
    for i in range(f,-1,-1): #creo que T tendría que seruno menos que N+1 cosa que no esta pasand
        x = np.copy(S[:, i])
        y = df * CF[:, i+1]
        
        y = y[x < K]
        x = x[x < K]
                
        x = x[:, np.newaxis]
        y = y[:, np.newaxis]
    
        ##Creo coeficientes
        coefs = create_polynomial_regression_model(2, x, y)
        CV[:, i] = coefs[0] + coefs[1]*S[:, i] + coefs[2]*S[:, i]*S[:, i]
        
        mask = ((K > S[:, i]) & (Matriz_B[:,i] == 1))
        CV[~mask, i] = 0
        
        CF[:, i] = np.maximum(K-S[:, i],0) #genera cashflows en el período i-1
        mask = ((CF[:, i] > CV[:, i]) & (K > S[:, i]) & (Matriz_B[:,i] == 1))
        CF[mask, i] = K - S[mask, i]
        CF[~mask, i] = 0
        
        mask = (CF[:, i]!=0) #si el cashflow en i-1 es != le pone al i 0
        CF[mask, i+1:] = 0
        
        
    DF = np.ones((M, N))
    for e in range(t,-1,-1):
        DF[:, e] = DF[:, e]*df**e
    P = np.sum(CF*DF)/ M
    return P

ValuacionOpcionAmericanaPut(M, N, S0, T, mu, sigma, r, K, B, kn, di)



##############################################################################
#DIFERENCIAS FINITAS MÉTODO EXPLICITO PARA EUROPEAS

s_max = 20       # lenght of the rod (m)
s_min = 0        # total time (s)
sigma = 0.25            # temperature held at the left boundary (K)         
K = 10
Bar = 12
r = 0.2
T = 1
N = 160
M = 1600
kn = 'in'
di = 'down'

##Diferencias Finitas metodo explicito para un opcion Vanilla Europea##
def ExplFinDif_CallEurBarrier(s_max, s_min, sigma, K, r, T, M, N, Bar, kn, di):
    '''
    s_max: Precio maximo que puede tomar la accion
    s_min: Precio minimo que puede tomar la accion
    K: Strike
    r: tasa libre de riesgo
    sigma: varianza
    mu: Esperanza de S
    M: Number of Paths
    T: Length from 0 to t
    N: Number of partitions for T 
    Bar: Precio de Barrera
    kn : tipo de barrera (out, in)
    di: tipo de barrera (up, down)
    
    SI KN Y DI SON == 0, ENOTONCES ES PLAIN VANILLA
    CASO CONTRARIO PUEDE SER UNA BARRIER OUT-DOWN, OUT-UP,IN-UP, IN-DOWN    
    diffusion number (has to be less than 0.5 for the solution to be stable):
    '''
    dt = (T/M) # Time step
    ds = (s_max - s_min)/N # Price step
    V = np.zeros([N, M])
    
    i = 0
    if ((kn == 'in' and di == 'up') or (kn == 'out' and di == 'down')):
        for i in range(0, N, 1):
            if ((s_min + i*ds) >= Bar):
                V[i, 0] = np.maximum(s_min + i*ds - K, 0) #Limite de Izquierda
            else:
                V[i, 0] = 0
        j = 0
        for j in range(1, M, 1):
            V[N-1,j] = (((N-1)*ds + s_min) - K*np.exp(-r*j*dt)) #Limite inferior
            
    elif ((kn == 'in' and di == 'down') or (kn == 'out' and di == 'up')):
         for i in range(0, N, 1):
            if ((s_min + i*ds) <= Bar):
                V[i, 0] = np.maximum(s_min + i*ds - K, 0) #Limite de Izquierda
            else:
                V[i, 0] = 0
    else:
        for i in range(0, N, 1):
            V[i, 0] = np.maximum(s_min + i*ds - K, 0) #Limite de Izquierda
        j = 0
        for j in range(1, M, 1):
            V[N-1,j] = (((N-1)*ds + s_min) - K*np.exp(-r*j*dt)) #Limite inferior

            
    V[0, 1:M] = 0 #Limite Superior
       
    j = 0
    A = np.zeros([N-1,1])
    B = np.zeros([N-1,1])
    C = np.zeros([N-1,1])
    for j in range(0,N-1,1):
        A[j,0] = 0.5*(sigma**2*j**2-r*j)*dt
        B[j,0] = 1 - (sigma**2*j**2+r)*dt
        C[j,0] = 0.5*(sigma**2*j**2+r*j)*dt
    
    n = 0
    m = 0
    for m in range(1, M, 1): # time
        for n in range(1, N-1, 1): # space
            V[n,m] = A[n,]*V[n-1,m-1] + B[n,]*V[n,m-1] + C[n,]*V[n+1,m-1]

    return V

Vv = ExplFinDif_CallEurBarrier(s_max, s_min, sigma, K, r, T, M, N, Bar, kn, di)
V = ExplFinDif_CallEurBarrier(s_max, s_min, sigma, K, r, T, M, N, Bar, 0, 0) #Vanilla

##############################################################################
#DIFERENCIAS FINITAS MÉTODO EXPLÍCITO PARA AMERICANAS
def ExplFinDif_CallAmericanaBarrier(s_max, s_min, sigma, K, r, T, M, N, Bar, kn, di):
    '''
     s_max: Precio maximo que puede tomar la accion
    s_min: Precio minimo que puede tomar la accion
    K: Strike
    r: tasa libre de riesgo
    sigma: varianza
    mu: Esperanza de S
    M: Number of Paths
    T: Length from 0 to t
    N: Number of partitions for T 
    Bar: Precio de Barrera
    kn : tipo de barrera (out, in)
    di: tipo de barrera (up, down)
    
    SI KN Y DI SON == 0, ENOTONCES ES PLAIN VANILLA
    CASO CONTRARIO PUEDE SER UNA BARRIER OUT-DOWN, OUT-UP,IN-UP, IN-DOWN    
    diffusion number (has to be less than 0.5 for the solution to be stable):
    '''
    dt = (T/M) # Time step
    ds = (s_max - s_min)/N # Price step
    V = np.zeros([N, M])
    
    i = 0
    if ((kn == 'in' and di == 'up') or (kn == 'out' and di == 'down')):
        for i in range(0, N, 1):
            if ((s_min + i*ds) >= Bar):
                V[i, 0] = np.maximum(s_min + i*ds - K, 0) #Limite de Izquierda
            else:
                V[i, 0] = 0
                
        V[N-1,] = (s_max - K) #Limite inferior
         
    elif ((kn == 'in' and di == 'down') or (kn == 'out' and di == 'up')):
         for i in range(0, N, 1):
            if ((s_min + i*ds) <= Bar):
                V[i, 0] = np.maximum(s_min + i*ds - K, 0) #Limite de Izquierda
            else:
                V[i, 0] = 0
    else:
         for i in range(0, N, 1):
            V[i, 0] = np.maximum(s_min + i*ds - K, 0) #Limite de Izquierda
            
    V[0, 1:M] = 0 #Limite Superior
    
        
    j = 0
    A = np.zeros([N-1,1])
    B = np.zeros([N-1,1])
    C = np.zeros([N-1,1])
    for j in range(0,N-1,1):
        A[j,0] = 0.5*(sigma**2*j**2-r*j)*dt
        B[j,0] = 1 - (sigma**2*j**2+r)*dt
        C[j,0] = 0.5*(sigma**2*j**2+r*j)*dt
    
    n = 0
    m = 0
    
    if ((kn == 'in' and di == 'down') or (kn == 'out' and di == 'up')):    
        for m in range(1, M, 1): # time
            for n in range(1, N-1, 1): # space
                if(np.maximum(s_min+ds*n - K,0) == 0):
                    V[n,m] = A[n,]*V[n-1,m-1] + B[n,]*V[n,m-1] + C[n,]*V[n+1,m-1]
                elif (Bar - (s_min+ds*n) < 0):
                        V[n,m] = A[n,]*V[n-1,m-1] + B[n,]*V[n,m-1] + C[n,]*V[n+1,m-1]                 
                else:
                        V[n,m] = V[n,m-1]
                        
    elif ((kn == 'in' and di == 'up') or (kn == 'out' and di == 'down')):
        if (K >= Bar):
            for m in range(1, M, 1): # time
                for n in range(1, N-1, 1): # space
                    if(np.maximum(s_min+ds*n - K,0) == 0):
                        V[n,m] = A[n,]*V[n-1,m-1] + B[n,]*V[n,m-1] + C[n,]*V[n+1,m-1]
                    else:
                        V[n,m] = V[n,m-1]
        elif (Bar > K):
             for m in range(1, M, 1): # time
                for n in range(1, N-1, 1): # space
                    if(np.maximum(s_min+ds*n - Bar,0) == 0):
                        V[n,m] = A[n,]*V[n-1,m-1] + B[n,]*V[n,m-1] + C[n,]*V[n+1,m-1]
                    else:
                        V[n,m] = V[n,m-1]
    return V

V = ExplFinDif_CallAmericanaBarrier(s_max, s_min, sigma, K, r, T, M, N, Bar, kn, di)



##############################################################################
#DIFERENCIAS FINITAS MÉTODO IMPLÍCITO PARA EUROPEAS
from scipy.sparse import diags
def ImplFinDif_CallEur(s_max, s_min, sigma, K, r, T, M, N):
    '''
    s_max: Precio maximo que puede tomar la accion
    s_min: Precio minimo que puede tomar la accion
    K: Strike
    r: tasa libre de riesgo
    sigma: varianza
    mu: Esperanza de S
    M: Number of Paths
    T: Length from 0 to t
    N: Number of partitions for T 
    '''
    vty = 12.5
    dt = (T/M) # Time step
    ds = (s_max - s_min)/N # Price step
    d = vty*dt/ds**2     
    s = np.arange(0, s_max + ds, ds)
    t = np.arange(0, T + dt, dt)
    nt = len(t)          # number of time steps
    ns = len(s) 
    V = np.zeros((ns,))
    
    V[0] = np.maximum(s_min - K, 0) #Limite de Izquierda np.maximum(s_min + i*ds - K, 0)
    
    V[ns-1] = np.maximum((((N-1)*ds + s_min) - K*np.exp(-r*M*dt)), 0) #(((N-1)*ds + s_min) - K*np.exp(-r*M*dt)) #Limite inferior

    A = diags([-d, 1+2*d, -d], [-1, 0, 1], shape=(ns-2, ns-2)).toarray() # create coefficient matrix   
    for n in range(nt): # time is going from second time step to last
        Vn = V #.copy()
        B = Vn[1:-1] # create vector of knowns on the RHS of the equation
        B[0] = B[0] + d*np.maximum(s_min - K, 0)
        B[-1] = B[-1] + d*np.maximum((((N-1)*ds + s_min) - K*np.exp(-r*M*dt)), 0)
        V[1:-1] = np.linalg.solve(A,B) # solve the equation using num        
    return V

VimpEur = ImplFinDif_CallEur(s_max, s_min, sigma, K, r, T, M, N)