
# coding: utf-8

import numpy as np
from scipy.spatial import distance


def backprop(Input_pattern, Hidden_out_activations, Output_out_activations, Target, Learning_rate, IH_wts, HO_wts):
    s = np.shape(Input_pattern)
    Error_Raw = Output_out_activations - Target
    Error_Op = Error_Raw * tanh_deriv(Output_out_activations)
    dE_dW = np.transpose(Hidden_out_activations).dot(Error_Op)
    HO_dwts = (-1) * Learning_rate * dE_dW

    # HO_wts = HO_wts + HO_dwts

    Error_Hidden = Error_Op.dot(HO_wts.transpose()) 
    Error_Hidden = Error_Hidden* tanh_deriv(Hidden_out_activations)
    Error_Hidden = Error_Hidden[0,0:np.int((s[1]-1)/2)]
    dE_dW = np.transpose(Input_pattern).dot([Error_Hidden])
    IH_dwts = (-1) * Learning_rate * dE_dW
    IH_wts = IH_wts + IH_dwts

    HO_wts = HO_wts + HO_dwts

    return IH_wts, HO_wts

def tanh_compress(x):
    a = np.tanh(x)
    return a


def tanh_deriv(tanh_compress_output):
    Fahlman_offset = 0.1
    a = 1 - tanh_compress_output * tanh_compress_output + Fahlman_offset
    # a = 1 - tanh_compress_output * tanh_compress_output
    return a


def feedforward(Input, IH_wts, HO_wts):

    bias_node = np.array([-1])

    Hid_net_act = Input.dot(IH_wts)
    Hid_out_act = tanh_compress(Hid_net_act)

    Hid_out_act = np.concatenate((Hid_out_act, [bias_node]), axis = 1)

    Out_net_act = Hid_out_act.dot(HO_wts)
    Output_out_act = tanh_compress(Out_net_act)

    return Hid_out_act, Output_out_act


def AT_module(t, delta_lower, delta_self, Hid_lower, Hid_self, RHS_lower, RHS_self, IH_wts, HO_wts, learning_rate, tau, learning, objective_func, k, ip_noise):
    bias_node = np.array([-1])
    a = tanh_compress(k*delta_self) #k: scaling factor of the delta
    b = tanh_compress(k*delta_lower)
    Delta = 500
    Target = 0
    Output = 0
    t2 = Hid_self
    Q =np.zeros((np.shape(Hid_lower)))
    vector_size = len(RHS_self)
    input_noise = np.random.uniform(-ip_noise, ip_noise, vector_size) #add noise to the input
    if t == 0:
        t1 = (1-b)*Hid_lower + b*RHS_lower
    else:
        ctx=tau/(tau+a)
        inp=a/(tau+a)
        t2 = ctx*Hid_self + inp*RHS_self
        t1 = (1-b)*Hid_lower+ b*RHS_lower

        Input = np.concatenate((t2,t1),axis = 0)
        s = np.shape(Input)
        Input = np.concatenate(([Input], [bias_node]), axis = 1)
        Q, Output = feedforward(Input, IH_wts, HO_wts)


        Target = Input[0,0:s[0]]
        if learning == True:
            IH_wts, HO_wts = backprop(Input, Q, Output, Target, learning_rate, IH_wts, HO_wts)
            Q, Output = feedforward(Input, IH_wts, HO_wts)
        Q = Q[0, 0:int(s[0]/2)]

        # for i in range(len(Q)):
        #     Q[i] = (Q[i]/np.sum(np.abs(Q)))*len(Q) #normalization of the Hid
        norm = np.linalg.norm(Q,keepdims=True)
        Q = Q/norm + input_noise

        D_ED = distance.euclidean(Output,Target)
        if objective_func == 'max':
            Delta = np.abs(Output-Target)
            Delta = Delta.max()
        elif objective_func == 'ED':
            Delta = np.mean((Output-Target)**2)
    return {'Delta':Delta, 'IH_wts':IH_wts ,'HO_wts':HO_wts, 'Hid':Q, 'RHS':t1, 'Target':Target, 'Output':Output}

def HAT_learn(bipolar_seq, IH_wts, HO_wts, IH_wts1, HO_wts1, IH_wts2, HO_wts2, learning_rate_initial, tau_array, Modeltype, objective_func, k, ip_noise):
    s = np.shape(bipolar_seq)
    arr_size = np.shape(bipolar_seq[0])

    bias_node = np.array([-1])

    Hid, Hid1, Hid2 = (np.zeros((arr_size))for i in range(3))
    RHS1, RHS2 = (np.zeros((arr_size))for i in range(2))

    D, D1, D2 = (np.zeros((0))for i in range(3))
    D_ED0, D_ED1, D_ED2 = ([] for i in range(3))
    Hid_array, Hid1_array, Hid2_array = (np.zeros((0,s[1]))for i in range(3))

    delta1 = 500
    delta2 = 500
    tau = tau_array[0]
    tau1 = tau_array[1]
    tau2 = tau_array[2]
    k1 = k[0]
    k2 = k[1]
    k3 = k[2]
    t = 0
    decay_rate=0.005 #learning rate will decay over time
    vector_size = arr_size
    input_noise = np.random.uniform(-ip_noise[0], ip_noise[0], vector_size)

    while t+1 < s[0]:
        learning_rate = learning_rate_initial[0] / (1 + decay_rate * t)
        learning_rate1 = learning_rate_initial[1] / (1 + decay_rate * t)
        learning_rate2 = learning_rate_initial[2] / (1 + decay_rate * t)
        if t == 0:
            In_t2 = bipolar_seq[t]
        else:
            a = tanh_compress(k1*delta)
            ctx=tau/(tau+a)
            inp=a/(tau+a)
            In_t2 = ctx*Hid+ inp*In_t1
        In_t1 = bipolar_seq[t + 1]
        RHS0 = In_t1    
        Input = np.concatenate((In_t2,In_t1),axis = 0)
        Input = np.concatenate(([Input], [bias_node]), axis = 1)

        Hid, Output = feedforward(Input, IH_wts, HO_wts)


        Target = Input[0,0:2*s[1]]

        IH_wts, HO_wts = backprop(Input, Hid, Output, Target, learning_rate, IH_wts, HO_wts)

        
        Hid, Output = feedforward(Input, IH_wts, HO_wts)

        Hid = Hid[0, 0:s[1]]

        # for i in range(len(Hid)):
        #     Hid[i] = (Hid[i]/np.sum(np.abs(Hid)))*len(Hid)
        norm = np.linalg.norm(Hid,keepdims=True)
        Hid = Hid/norm + input_noise


        # Hid_array = np.concatenate((Hid_array, [Hid]), axis = 0)

        if objective_func=='max':
            delta = np.abs(Output-Target)
            delta = delta.max()
        elif objective_func=='ED':
            delta = np.mean((Output-Target)**2)

        D = np.concatenate((D,[delta]),axis = 0)
        
        if Modeltype == 'Hid_only':
            delta_lower = 0
        else:
            delta_lower=delta
        H1 = AT_module(t, delta_lower, delta1, Hid, Hid1, RHS0, RHS1, IH_wts1, HO_wts1, learning_rate1, tau1, True, objective_func, k2, ip_noise[1])
        Hid1 = H1['Hid']
        RHS1 = H1['RHS']
        IH_wts1 = H1['IH_wts']
        HO_wts1 = H1['HO_wts']
        delta1 = H1['Delta']
        # Hid1_array = np.concatenate((Hid1_array, [Hid1]), axis = 0)
        D1 = np.concatenate((D1,[H1['Delta']]),axis = 0)            
        
        if Modeltype == 'Hid_only':
            delta_lower = 0
        else:
            delta_lower=delta1
        H2 = AT_module(t, delta_lower, delta2, Hid1, Hid2, RHS1, RHS2, IH_wts2, HO_wts2, learning_rate2, tau2, True, objective_func, k3, ip_noise[2])
        Hid2 = H2['Hid']
        RHS2 = H2['RHS'] 
        IH_wts2 = H2['IH_wts']
        HO_wts2 = H2['HO_wts']
        delta2 = H2['Delta']
        # Hid2_array = np.concatenate((Hid2_array, [Hid2]), axis = 0)
        D2 = np.concatenate((D2,[H2['Delta']]),axis = 0)
            
        t = t+1

    D = D[1:t]
    D1 = D1[1:t]
    D2 = D2[1:t]

    # Hid_array = Hid_array[1:t]
    # Hid1_array = Hid1_array[1:t]
    # Hid2_array = Hid2_array[1:t]

    return {'IH_wts':IH_wts, 'HO_wts':HO_wts , 'IH_wts1':IH_wts1, 'HO_wts1':HO_wts1, 'IH_wts2':IH_wts2, 'HO_wts2':HO_wts2, 'D':D, 'D1':D1, 'D2':D2} 
    # return {'IH_wts':IH_wts, 'HO_wts':HO_wts , 'IH_wts1':IH_wts1, 'HO_wts1':HO_wts1, 'IH_wts2':IH_wts2, 'HO_wts2':HO_wts2} 


def HAT_test(seq, IH_wts, HO_wts, IH_wts1, HO_wts1, IH_wts2, HO_wts2, tau_array, Modeltype, objective_func, k, ip_noise):
    bipolar_seq = seq
    s = np.shape(bipolar_seq)
    arr_size = np.shape(bipolar_seq[0])

    bias_node = np.array([-1])
    
    Hid, Hid1, Hid2 = (np.zeros((arr_size))for i in range(3))
    RHS1, RHS2 = (np.zeros((arr_size))for i in range(2))
    D, D1, D2 = (np.zeros((0))for i in range(3))
    Hid_array, Hid1_array, Hid2_array = (np.zeros((0,s[1]))for i in range(3))
    

    delta1 = 500
    delta2 = 500
    tau = tau_array[0]
    tau1 = tau_array[1]
    tau2 = tau_array[2]
    k1 = k[0]
    k2 = k[1]
    k3 = k[2]
    a = 0
    t = 0
    vector_size = arr_size
    input_noise = np.random.uniform(-ip_noise[0], ip_noise[0], vector_size)
    while t+1 < s[0]:
        if t == 0:
            In_t2 = bipolar_seq[t]
        else:
            a = tanh_compress(k1*delta)
            ctx=tau/(tau+a)
            inp=a/(tau+a)
            In_t2 = ctx*Hid + inp*In_t1            
            
        In_t1 = bipolar_seq[t + 1]  
        RHS0 = In_t1
        Input = np.concatenate((In_t2,In_t1),axis = 0)
        Input = np.concatenate(([Input], [bias_node]), axis = 1)

        Hid, Output = feedforward(Input, IH_wts, HO_wts)

        Target = Input[0,0:2*s[1]]
        
        Hid = Hid[0, 0:s[1]]
        # for i in range(len(Hid)):
        #     Hid[i] = (Hid[i]/np.sum(np.abs(Hid)))*len(Hid)
        norm = np.linalg.norm(Hid,keepdims=True)
        Hid = Hid/norm + ip_noise[0]
        Hid_array = np.concatenate((Hid_array, [Hid]), axis = 0)

        if objective_func=='max':
            delta = np.abs(Output-Target)
            delta = delta.max()
        elif objective_func=='ED':
            delta = np.mean((Output-Target)**2)

        D = np.concatenate((D,[delta]),axis = 0)

        if Modeltype == 'Hid_only':
            delta_lower = 0
        else:
            delta_lower=delta
        H1 = AT_module(t, delta_lower, delta1, Hid, Hid1, RHS0, RHS1, IH_wts1, HO_wts1, 0, tau1, False, objective_func, k2, ip_noise[1])
        Hid1 = H1['Hid']
        RHS1 = H1['RHS'] 
        delta1 = H1['Delta']
        Hid1_array = np.concatenate((Hid1_array, [H1['Hid']]), axis = 0)
        D1 = np.concatenate((D1,[H1['Delta']]),axis = 0)
        
        if Modeltype == 'Hid_only':
            delta_lower = 0
        else:
            delta_lower=delta1

        H2 = AT_module(t, delta_lower, delta2, Hid1, Hid2, RHS1, RHS2, IH_wts2, HO_wts2, 0, tau2, False, objective_func, k3, ip_noise[2])
        Hid2 = H2['Hid']
        RHS2 = H2['RHS'] 
        delta2 = H2['Delta']
        Hid2_array = np.concatenate((Hid2_array, [H2['Hid']]), axis = 0)
        D2 = np.concatenate((D2,[H2['Delta']]),axis = 0) 

        t = t+1

    D = D[1:t] #D_array start from the 3rd input
    D1 = D1[1:t]
    D2 = D2[1:t]
    Hid_array = Hid_array[1:t] #Hid_array start from the 3rd input
    Hid1_array = Hid1_array[1:t]
    Hid2_array = Hid2_array[1:t]
    return {'D':D, 'D1':D1, 'D2':D2, 'Hid':Hid_array, 'Hid1':Hid1_array, 'Hid2':Hid2_array} 

