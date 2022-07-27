from numpy.linalg import norm

def explicit_RK_stepper(f,x,t,h,a,b,c):
    """
        Implementation of generic explicit Runge-Kutta update for explicit ODEs
        
        inputs:
            x - current state 
            t - current time
            f - right-hand-side of the (explicit) ODE to be integrated (signature f(x,t))
            h - step size 
            a - coefficients of Runge-Kutta method (organized as list-of-list (or vector-of-vector))
            b - weights of Runge-Kutta method (list/vector)
            c - nodes of Runge-Kutta method (including 0 as first node) (list/vector)

        outputs: 
            x_hat - estimate of the state at time t+h
    """
    s = len(b) # refer to RK update formula for reference
    ks = [f(x,t)] # initialize list of k's with k1
    x_hat = x + h*b[0]*ks[-1] # initialize right hand side with first element of sum
    for i in range(s-1):
        y = x + h*sum(a[i][j]*ks[j] for j in range(i+1)) # compute state component for next k
        ks.append(f(y,t+c[i+1]*h)) # compute and append next k 
        x_hat += b[i+1]*h*ks[-1] # using next k, compute next element of sum
    
    return x_hat 

def integrate(f, x0, tspan, h, step):
    """
        Generic integrator interface

        inputs:
            f     - rhs of ODE to be integrated (signature: dx/dt = f(x,t))
            x0    - initial condition (numpy array)
            tspan - integration horizon (t0, tf) (tuple)
            h     - step size
            step   - integrator with signature: 
                        step(f,x,t,h) returns state at time t+h 
                        - f rhs of ODE to be integrated
                        - x current state
                        - t current time 
                        - h stepsize

        outputs: 
            ts - time points visited during integration (list)
            xs - trajectory of the system (list of numpy arrays)
    """
    t, tf = tspan
    x = x0
    trajectory = [x0]
    ts = [t]
    while t < tf:
        h_eff = min(h, tf-t)
        x = step(f,x,t,h_eff)
        t = min(t+h_eff, tf)
        trajectory.append(x)
        ts.append(t)
    return trajectory, ts