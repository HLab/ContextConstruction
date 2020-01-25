import numpy as np


def Norm_vec(vec):
    norm = np.linalg.norm(vec)
    if norm == 0: 
        return vec
    return vec / norm
    
    
def HLI_train(bipolar_train, beta, gamma):
    gamma_fc = 0.898
    time, vec_size=np.shape(bipolar_train)
    Mtf = np.identity(vec_size)*0
    Mtf = np.repeat([Mtf],3,axis=0)
    Mft = np.identity(vec_size)*(1-gamma_fc)
    Mft = np.repeat([Mft],3,axis=0)
    t = np.zeros((3,1,vec_size))
    for i in range(time):
        for l in range(3):
#             print("layer:",l)
            if l==0:
                f = [bipolar_train[i]]
            else:
                f = t_below
            t_in = np.dot(Mft[l],np.transpose(f))
            t_in = np.transpose(t_in)
            t_in = Norm_vec(t_in)
            t_tin=np.inner(t_in,t[l])
            ro = (1+beta[l]**2*(t_tin**2-1))**0.5 - beta[l]*(t_tin)
            f=np.array(f)
            t[l] = ro*t[l] + beta[l]*f
            t[l] = Norm_vec(t[l])
            Mtf[l] = Mtf[l] + np.dot(np.transpose(f),t[l])

            P = np.dot(np.transpose(f),f)/np.linalg.norm(f)**2
            B = 1/(gamma**2 + 2*gamma*np.inner(t_in, t[l]) + 1)
            A = gamma * B
            Mft[l] = Mft[l]*(1-P) + A*Mft[l]*P + B*np.dot(np.transpose(t[l]),f)
            t_below = t[l]

            

    
    return Mtf, Mft


def HLI_test(bipolar_test, Mtf, Mft,  beta, gamma):
    time, vec_size=np.shape(bipolar_test)
    t_in_arr = np.zeros((0,vec_size))
    t_arr = np.zeros((3,time,vec_size))
    t = np.zeros((3,1,vec_size))
    for i in range(time):
#         print("time",i)
        for l in range(3):
            if l==0:
                f = [bipolar_test[i]]
            else:
                f = t_below   
            t_in = np.dot(Mft[l],np.transpose(f)) #preexperimental
            t_in_arr = np.concatenate((t_in_arr,np.transpose(t_in)),0)
            t_in = np.transpose(t_in)
            t_in = Norm_vec(t_in)
            t_tin=np.inner(t_in,t[l])
            ro = (1+beta[l]**2*(t_tin**2-1))**0.5 - beta[l]*(t_tin)
            f=np.array(f)
            t[l] = ro*t[l] + beta[l]*f 
            t[l] = Norm_vec(t[l])
            t_arr[l][i] = t[l]
            t_below = t[l]

    return t_arr



