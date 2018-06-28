""" heredity/deterministic.py -- Evolution of heredity thought experiment, deterministic model.
This file is part of the ecological scaffolding package/ heredity model subpackage.
Copyright 2018 Guilhem Doulcier, Licence GNU GPL3+
--
"""
from functools import partial
import numpy as np
import scipy.integrate

def lotka_volterra(x,_,r,a):
    '''Temporal derivative of the Lotka-Volterra system.
    Args:
        x (array of N floats): number of individual of each type.
        _ (None): time, not used but required by scipy.integrate.
        r (array of N floats): maximum growth rate of each type.
        a (array of NxN floats): inter-type interaction coefficient.
    '''
    return np.multiply(np.multiply(r, x), (1-np.dot(a, x)))

def iterate_ecology(r, A, B, T, tstep=500, generations=8, x0=0.1):
    """Returns the ecological trajectory across several generations.

    r (array of 2 floats): maximum growth rate of each type.
    A (array of 2x2 floats): inter-type interaction coefficient.
    B (float): Bottleneck size.
    T (float): Duration of growth rate
    tstep (int): number of integration steps per generation.
    generations (int): number of collective generations
    x0 (float): initial proportion of types.

    Returns a list of `generations` succesive trajectories and a list
    of `generations`+1 final proportions
    """
    flow = partial(lotka_volterra, r=r, a=A)
    time = np.linspace(0, T, tstep)
    plist = [x0]
    traj = []

    x = x0
    for _ in range(generations):
        xy = scipy.integrate.odeint(flow,
                                    y0=np.array([B*x, B*(1-x)]),
                                    t=time)
        x = float(xy[-1, 0]/xy[-1, :].sum())
        traj.append(xy)
        plist.append(x)

    return traj, plist

def convert_phenotypes_to_lv(phenotypes, K):
    """Convert the parameters of the stochastic model to the parameters of
    the deterministic model.

    Return the growth array r and interaction matrix A."""
    if phenotypes.shape[0] == 2:
        pos_0 = list(phenotypes[:, 0]).index(0)
        pos_1 = list(phenotypes[:, 0]).index(1)
        r = np.array([phenotypes[pos_0, 1], phenotypes[pos_1, 1]])
        A = np.array([[phenotypes[pos_0, 2]/K, phenotypes[pos_1, 3]/K],
                      [phenotypes[pos_0, 3]/K, phenotypes[pos_1, 2]/K]])
    else:
        r = np.array(phenotypes[:, 1])
        A = np.empty(shape=(phenotypes.shape[0],phenotypes.shape[0]))
        for i, focal in enumerate(phenotypes):
            A[i,i] = focal[2]
            for j, other in enumerate(phenotypes):
                A[i,j] = other[2+int(focal[0] != other[0])]
        A /= K
    return r,A

def stable_equilibria(A):
    """Output the stable equilibrium point provided with the interaction
    matrix."""
    if A[0, 0] > A[1, 0] and A[1, 1] > A[0, 1]:
        # Coexistence
        return [(A[1, 1]-A[0, 1])/np.linalg.det(A), (A[0, 0]-A[1, 0])/np.linalg.det(A)]
    elif  A[0, 0] > A[1, 0]:
        # 0 exclude 1
        return [1/A[0, 0], 0]
    elif  A[0, 0] > A[1, 0]:
        # 1 exclude 0
        return [0, 1/A[1, 1]]
    # Bistable.
    return [np.nan,np.nan]

def pstar(A):
    '''Return the proportion of type 1 cells at the stable
    attractor'''
    if A[0, 0] > A[1, 0] and A[1, 1] > A[0, 1]:
        # Coexistence
        return (A[1, 1] - A[0, 1])/(A.trace()-A[1, 0]-A[0, 1])
    elif  A[0, 0] > A[1, 0]:
        # 0 exclude 1
        return 0
    elif  A[1, 1] > A[0, 1]:
        # 1 exclude 0
        return 1
    # Bistable.
    return np.nan

def tstar(r, A, B, precise=False, steps=100, eps=1e-6):
    '''Return the value of T where an internal fixed point appear.

    Args:
        r (np.array, size 2): growth rate
        A (np.array, size 2x2): interaction matrix
        B (int): bottleneck size
        precise (bool): see below
        steps (int): max number of steps for the numerical approx.
        eps (float): precision for the numerical approx.

    If precise is False, will only output a rough analytical
    approximation. Otherwise, will try to numerically compute it.

    Returns np.nan if there is no internal fiexd point.
    '''
    stable_eq = pstar(A)

    ifast, islow = (1, 0) if r[1] > r[0] else (0, 1)

    # If the fast growing type is also the one excluding the other,
    # there is no critical time.
    if ((ifast == 1 and np.isclose(stable_eq, 1))
        or (ifast == 0 and np.isclose(stable_eq, 0))):
        return np.nan

    guesstimate = (((1/r[ifast])*(A[islow, ifast]/A[ifast, ifast]) - (1/r[islow]))
                   / (1 - (A[islow, ifast]/A[ifast, ifast]))
                   * np.log(B*A[ifast, ifast]/(B*A[ifast, ifast]+1)))
    if not precise:
        return guesstimate

    t_0 = guesstimate
    t_max = guesstimate

    low = 0+eps
    high = 1-eps
    gfunc = get_gfunc(r, A, B)
    negative_if_fp = lambda t : (gfunc(low, t)-low) * (gfunc(high, t)-high)

    # Find a 'big' T for which there is no fixed point
    i = 0
    while (negative_if_fp(t_0)<0) and i<steps:
        t_0 *= 0.66
        i += 1

    i = 0
    # Find a 'small' T for wich there is a fixed point.
    while (negative_if_fp(t_max)>0) and i<steps:
        t_max *= 1.33
        i += 1

    if negative_if_fp(t_0)>0 and negative_if_fp(t_max)<0:
        return scipy.optimize.brentq(negative_if_fp, t_0, t_max)

    return np.nan


def get_gfunc(r, A, B, T=None, tstep=1000):
    """ Return the G function (final proportion as a function of initial proportions)
    for a given set of parameters

    Args:
        r (np.array, size 2): growth rate
        A (np.array, size 2x2): interaction matrix
        B (int): bottleneck size
        T (float): duration of the growth phase
        tstep (int): integration setps.

    Return: function (float in [0,1]) -> (float in [0,1])
    """
    func = partial(lotka_volterra, r=r, a=A)
    def gfunc(x, T=T):
        if not T:
            return float(x)
        x = float(x)
        xy = scipy.integrate.odeint(func,
                                    y0=np.array([B*x, B*(1-x)]),
                                    t=np.linspace(0, T, tstep))
        return float(xy[-1, 0]/xy[-1, :].sum())
    return gfunc


def continuation_on_T(gfunc, start, t0=1, tf=1e-3, tstep=100):
    """Use natural parameter continuation to draw the bifurcation diagram
    of the fixed point of G, x*, as a function of T.

    Args:
        gfunc (func: x,T -> x* [0,1]): proportion at the end of the growth
              phase as a function of the proportion at the begining of
              the growth phase (x) and the duration of the growth phase T.
       start (float): initial value of x* for T=t0
       t0 (float): initial value of T along the continuation
       tf (float): final value of T along the continuation
       tstep (int): number of values of T to compute between t0 and tf.

    Returns: list of T, list of x*
    """
    equilibria = []
    t_list = np.linspace(t0, tf, tstep)

    for T in t_list:
        fixed_point_function = lambda x, t=T: gfunc(x,t)-x
        e = scipy.optimize.fsolve(fixed_point_function,
                                  x0=(equilibria[-1]
                                      if equilibria
                                      else start))
        equilibria.append(float(e))

    return t_list, equilibria

def stability_of_01(gfunc, t0=1e-6, tf=1, eps=1e-6):
    """Quickly see if 0 and 1 are stable fixed point of G.

    Useful to draw the bifurcation diagram since 0 and 1 are always fixed
    points so there is no need to do a full continuation on those points.

    Args:
        gfunc (func: x,T -> x* [0,1]): proportion at the end of the growth
              phase as a function of the proportion at the begining of
              the growth phase (x) and the duration of the growth phase T.
       start (float): initial value of x* for T=t0
       t0 (float): initial value of T along the continuation
       tf (float): final value of T along the continuation
       eps (float): numerical precision
    """
    low = 0+eps
    high = 1-eps
    stability_0 = lambda t: gfunc(low,t)-low
    stability_1 = lambda t: gfunc(high,t)-high

    stable0 = stability_0(t0) < 0
    if stability_0(t0)*stability_0(tf)<0:
        tcrit =scipy.optimize.brentq(stability_0, t0, tf)
        seg0 = [[stable0,[t0, tcrit]],[not stable0,[tcrit,tf]]]
    else:
        seg0 = [[stable0,[t0,tf]]]

    stable1 = stability_1(t0) > 0
    if stability_1(t0)*stability_1(tf)<0:
        tcrit= scipy.optimize.brentq(stability_1, t0, tf)
        seg1 = [[stable1,[t0, tcrit]],[not stable1,[tcrit,tf]]]
    else:
        seg1 = [[stable1,[t0,tf]]]


    return {0:seg0, 1:seg1}
