import numpy as np
from numba import jit
from tqdm import tqdm, trange
from multiprocess import Pool


_use_jit = True
_ncore = 5


def compute_fr_multi_inputs(ic_array_sets, tmax, teq=20, dt=0.01, **hhparams):
    
    def compute_fr(ic_array):
        sobj = hhmodel(dt=dt, **hhparams)
        v0 = np.random.randn() * 5 - 70
        sobj.reset(v0)
        sobj.run(lambda t: ic_array[int(t/dt)], tmax=tmax)
        _t = np.array(sobj.t_spk)
        return np.sum(_t > teq) / (tmax - teq) * 1e3
    
    if len(ic_array_sets[0]) != int(tmax/dt):
        raise ValueError("Size ~~")
    
    fr_sets = np.zeros(len(ic_array_sets))
    with Pool(processes=_ncore) as p:
        with tqdm(total=len(ic_array_sets)) as pbar:
            for i, fr in enumerate(p.imap(compute_fr, ic_array_sets)):
                fr_sets[i] = fr
                pbar.update()

    return fr_sets
    

class hhmodel:
    """
    
    dt: time step (ms)
    """
    def __init__(self, dt=0.01, cm=1, gna0=120, gk0=36, gl=0.3, ena=55, ek=-75, el=-65):
        self.cm = cm
        self.gna0 = gna0
        self.gk0 = gk0
        self.gl = gl
        self.ena = ena
        self.ek = ek
        self.el = el
        self.dt = dt
        
        self.params = (
            self.cm, self.gna0, self.gk0, self.gl,
            self.ena, self.ek, self.el
        )
        
        self.v, self.n, self.m, self.h = 0, 0, 0, 0
        self.reset()
    
    def run(self, finput, tmax=100, disable_tqdm=False):
        if len(self.t_spk) > 0:
            Warning("The model state is not resetted")
            
        nmax = int(tmax / self.dt)
        self.gen_empty_array(nmax)
        
        itr = range(nmax) if disable_tqdm else trange(nmax)
        for n in itr:
            iapp = finput(n*self.dt)
            self.update_state(n, iapp)
            self.detect_spike(n * self.dt)
        
    def gen_empty_array(self, nmax):
        self.vs = np.zeros(nmax+1); self.vs[0] = self.v
        self.ns = np.zeros(nmax+1); self.ns[0] = self.n
        self.ms = np.zeros(nmax+1); self.ms[0] = self.m
        self.hs = np.zeros(nmax+1); self.hs[0] = self.h
        self.ts = np.arange(nmax+1) * self.dt
        self.t_spk = []
        
    def update_state(self, n, iapp):
        self.v, self.n, self.m, self.h = _update_state_rk4(iapp, 
                                                           self.v, self.n, self.m, self.h,
                                                           self.dt, self.params)
        
        self.vs[n+1] = self.v
        self.ns[n+1] = self.n
        self.ms[n+1] = self.m
        self.hs[n+1] = self.h

    def reset(self, v0=-70):
        self.v = v0
        self.vold = v0
        self.n = _an(self.v) / (_an(self.v) + _bn(self.v))
        self.m = _am(self.v) / (_am(self.v) + _bm(self.v))
        self.h = _ah(self.v) / (_ah(self.v) + _bh(self.v))
        self.gen_empty_array(1)
        self.t_spk, self.spk_flag = [], True
        
    def detect_spike(self, t):
        if self.spk_flag and self.v > 0:
            dv = self.v - self.vold    
            if dv < 0:
                self.t_spk.append(t - self.dt)
                self.spk_flag = False
                
        elif not self.spk_flag and self.v < 0:
            self.spk_flag = True
            
        self.vold = self.v


@jit(nopython=True)
def _update_state_rk4(iapp, v, n, m, h, dt, param):
    # Use Runge-Kutta 4th order method to solve the equation
    dv1, dn1, dm1, dh1 = _solve_deq(iapp, v,       n,       m,       h,       dt, param)
    dv2, dn2, dm2, dh2 = _solve_deq(iapp, v+dv1/2, n+dn1/2, m+dm1/2, h+dh1/2, dt, param)
    dv3, dn3, dm3, dh3 = _solve_deq(iapp, v+dv2/2, n+dn2/2, m+dm2/2, h+dh2/2, dt, param)
    dv4, dn4, dm4, dh4 = _solve_deq(iapp, v+dv3,   n+dn3,   m+dm3,   h+dh3,   dt, param)
    
    v_next = v + (dv1 + 2*dv2 + 2*dv3 + dv4)/6
    n_next = n + (dn1 + 2*dn2 + 2*dn3 + dn4)/6
    m_next = m + (dm1 + 2*dm2 + 2*dm3 + dm4)/6
    h_next = h + (dh1 + 2*dh2 + 2*dh3 + dh4)/6
    
    return v_next, n_next, m_next, h_next


@jit(nopython=_use_jit)
def _solve_deq(iapp, v, n, m, h, dt, params):
    # params = [cm, gna0, gk0, gl, ena, ek, el]
    dn = _get_dn(v, n)
    dm = _get_dm(v, m)
    dh = _get_dh(v, h)
    
    ina = params[1] * m**3*h * (v - params[4])
    ik  = params[2] * n**4 * (v - params[5])
    il  = params[3] * (v - params[6])
    dv = (-ina -ik -il + iapp) / params[0]
    
    return dv*dt, dn*dt, dm*dt, dh*dt
    

@jit(nopython=_use_jit)
def _get_dn(v, n):
    return _an(v) * (1-n) - _bn(v)*n


@jit(nopython=_use_jit)
def _get_dm(v, m):
    return _am(v) * (1-m) - _bm(v)*m


@jit(nopython=_use_jit)
def _get_dh(v, h):
    return _ah(v) * (1-h) - _bh(v)*h
        
        
@jit(nopython=_use_jit)
def _an(v):
    return 0.01 * (v+55) / (1 - np.exp(-(v+55)/10))


@jit(nopython=_use_jit)
def _am(v):
    return 0.1 * (v+40) / (1 - np.exp(-(v+40)/10))


@jit(nopython=_use_jit)
def _ah(v):
    return 0.07 * np.exp(-(v+65)/20)


@jit(nopython=_use_jit)
def _bn(v):
    return 0.125 * np.exp(-(v+65)/80)


@jit(nopython=_use_jit)
def _bm(v):
    return 4 * np.exp(-(v+65)/18)


@jit(nopython=_use_jit)
def _bh(v):
    return 1 / (np.exp(-(v+35)/10) + 1)

    
