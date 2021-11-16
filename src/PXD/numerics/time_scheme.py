from dolfin import Constant

class TimeScheme:

    available_schemes = {
        'euler_implicit': {
            'nodes': 2
        },
        'BETF': {
            'nodes': 3
        }
    }


    def __init__(self,method,tau = 1):
        assert method in self.available_schemes , 'The method specified is not implemented'
        self.method = method
        self.delta_t = Constant(0, name='dt')
        self.nodes = self.available_schemes[self.method]['nodes']
        if self.method == 'BETF':
            self.tau = Constant(tau)
            self.nu = Constant(self.tau*(1+self.tau)/(1+2*self.tau))


    def num_functions(self):
        return self.nodes

    def dt(self, *args):
        if self.method == 'euler_implicit':
            assert(len(args)==2)
            return self._euler_implicit(args[0],args[1])
        if self.method == 'BETF':
            assert(len(args)>=3 and len(args)<=4)
            return self._BETF(args[0],args[1],args[2])

    def cur_time(self):
        if self.method == 'euler_implicit':
            return 1
        if self.method == 'BETF':
            return 2

    def _euler_implicit(self,u_0,u_1):
        return (u_1-u_0)/self.delta_t
    
    def _BETF(self,u_0,u_1,u_2):
        return ((1+self.tau-self.nu)/(1+self.tau)*u_2+(self.nu-1)*u_1-(self.nu*self.tau)/(1+self.tau)*u_0)/self.delta_t

    def set_timestep(self,timestep):
        self.delta_t.assign(timestep)
    
    def get_timestep(self):
        return self.delta_t.values()[0]

    def set_tau(self,tau):
        self.tau.assign(tau)
        self.nu.assign(self.tau*(1+self.tau)/(1+2*self.tau))
    
    def update_time_step(self, error, dt, tol = 1e-6, max_step = 1e3, min_step = 1e-3):
        
        # Adatpive strategy based on V. Decaria et al (2019)
        if error >= tol or error <= tol/4 and error != 0:
            dt_0 = dt
            dt = 0.9*dt*min(max( (tol/(2*error))**0.5 ,0.5),2)
            dt = min(max(dt, min_step), max_step)
            return float(dt/dt_0)
        if error == 0:
            return float(2*0.9)
        else:
            return 1
        
        # Basic adaptive strategy 
        # if error >= tol:
        #     dt_0 = dt
        #     dt = max(dt*0.8,min_step)
        #     self.set_timestep(dt)
        #     return dt/dt_0
        # elif error <= tol/8:
        #     dt_0 = dt
        #     dt = min(dt*1.25,max_step)
        #     self.set_timestep(dt)
        #     return dt/dt_0
        # else:
        #     return 1

    def list_implemented_methods(self):
        for i in self.available_schemes:
            print(i)