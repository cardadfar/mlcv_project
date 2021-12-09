import torch
import numpy as np
import math

def sigmoid(x, scale=10.0):
    return 1.0 / (1.0 + np.exp(scale * (-x + 0.5)))

def iden(x):
    return x

def B(x, k, i, t):
    if k == 0:
       return 1.0 if torch.all((t[i] <= x) == True) and torch.all((x < t[i+1]) == True) else 0.0
    c1 = (x - t[i])/(t[i+k] - t[i] + EPS_F) * B(x, k-1, i, t)
    c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1] + EPS_F) * B(x, k-1, i+1, t)
    return c1 + c2

def bspline_interp(x, t, c, k):
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html
    n = len(t) - k - 1
    assert (n >= k+1) and (len(c) >= n)
    return sum(c[i] * B(x, k, i, t) for i in range(n))

def bspline(x, ibf=1, ease=iden):
    '''
    linearly interpolate between frames
    x : encoded tensor (n, encode_dim)
    ibf : inbetween frame num
    '''

    n, encode_dim = x.shape

    y = torch.Tensor((n-1)*(ibf+1)+1, encode_dim)
    idx = 0
    for i in range(n-1):
        y[idx] = x[i]
        idx += 1
        for f in range(1,ibf+1):
            t = ease(f / (ibf + 2.0))
            xt = linear_interp(x[i], x[i+1], t)
            k = 2
            c = np.zeros(n - k - 1)
            c[0] = -1
            c[-1] = 1
            ibtwn = bspline_interp(xt, x, c, k)
            y[idx] = ibtwn
            idx += 1
    
    y[idx] = x[n-1]
    return y

def catmullRom_interp(x0, x1, x2, x3, t):
    # https://www.mvps.org/directx/articles/catmull/
    return  0.5 * ((2*x1) +\
                    (-x0 + x2) * t +\
                    (2*x0 - 5*x1 + 4*x2 - x3) * t**2 +\
                    (-x0 + 3*x1- 3*x2 + x3) * t**3)

def catmullRom(x, ibf=1, ease=iden):
    '''
    linearly interpolate between frames
    x : encoded tensor (n, encode_dim)
    ibf : inbetween frame num
    '''

    n, encode_dim = x.shape

    y = torch.Tensor((n-1)*(ibf+1)+1, encode_dim)
    idx = 0
    '''
    for i in range(n-1):
        y[idx] = x[i]
        idx += 1
        for f in range(1,ibf+1):
            t = ease(f / (ibf + 2.0))
            if i == 0:
                ibtwn = catmullRom_interp(x[i], x[i], x[i+1], x[i+2], t)
            elif i == n - 2:
                ibtwn = catmullRom_interp(x[i-1], x[i], x[i+1], x[i+1], t)
            else:
                ibtwn = catmullRom_interp(x[i-1], x[i], x[i+1], x[i+2], t)
            y[idx] = ibtwn
            idx += 1
    
    y[idx] = x[n-1]
    '''
    L = []
    T = []
    for f in range(n):
        tn = f / float(n)
        t = ease(tn)
        T.append(t)

    for f in range((n-1)*(ibf+1)+1):
        tn = f / ((n-1)*(ibf+1)+1.0)
        t = ease(tn)
        i = math.floor(t * (n-1))

        t1 = i / float(n-1)
        t2 = (i+1) / float(n-1)
        t = (t - t1) / (t2 - t1)
        #print(t)

        if i == n - 1:
            ibtwn = x[i]
        else:
            L.append(t)
            if n == 2:
                ibtwn = catmullRom_interp(x[i], x[i], x[i+1], x[i+1], t)
            elif i == 0:
                ibtwn = catmullRom_interp(x[i], x[i], x[i+1], x[i+2], t)
            elif i == n - 2:
                ibtwn = catmullRom_interp(x[i-1], x[i], x[i+1], x[i+1], t)
            else:
                ibtwn = catmullRom_interp(x[i-1], x[i], x[i+1], x[i+2], t)
            y[f] = ibtwn
            
    return y

def linear_interp(x0, x1, t):
    return (1.0 - t) * x0 + t * x1

def linear(x, ibf=1, ease=iden, p=[(1,1)]):
    '''
    linearly interpolate between frames
    x : encoded tensor (n, encode_dim)
    ibf : inbetween frame num
    '''

    n, encode_dim = x.shape

    y = torch.Tensor((n-1)*(ibf+1)+1, encode_dim)
    idx = 0
    for i in range(n-1):
        y[idx] = x[i]
        idx += 1
        for f in range(1,ibf+1):
            t = (f / (ibf + 2.0))

            lastA = 0
            lastB = 0
            currA = 0
            currB = 0
            for a, b in p:
                currA = a
                currB = b
                if lastA <= t and t <= currA:
                    break
                lastA = currA
                lastB = currB

            t = (t - lastA) / (currA - lastA)
            t = ease(t)
            t = lastB * (1 - t) + currB * t

            ibtwn = linear_interp(x[i], x[i+1], t)
            y[idx] = ibtwn
            idx += 1
    
    y[idx] = x[n-1]
    return y