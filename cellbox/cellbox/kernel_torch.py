"""
This module defines the ODE formulations, including the choices of ODE solvers,
degree of ODEs, and the envelope forms
"""

import torch
import torch.nn as nn

def get_envelope(args):
    """get the envelope form based on the given argument"""
    if args.envelope_form == 'tanh':
        args.envelope_fn = torch.tanh
    elif args.envelope_form == 'polynomial':
        k = args.polynomial_k
        assert k > 1, "Hill coefficient has to be k>2."
        if k % 2 == 1:  # odd order polynomial equation
            args.envelope_fn = lambda x: x ** k / (1 + torch.abs(x) ** k)
        else:  # even order polynomial equation
            args.envelope_fn = lambda x: x**k/(1+x**k)*torch.sign(x)
    elif args.envelope_form == 'hill':
        k = args.polynomial_k
        assert k > 1, "Hill coefficient has to be k>=2."
        args.envelope_fn = lambda x: 2*(1-1/(1+nn.functional.relu(torch.tensor(x+1)).numpy()**k))-1
    elif args.envelope_form == 'linear':
        args.envelope_fn = lambda x: x
    elif args.envelope_form == 'clip linear':
        args.envelope_fn = lambda x: torch.clamp(x, min=-1, max=1)
    else:
        raise Exception("Illegal envelope function. Choose from [tanh, polynomial/hill]")
    return args.envelope_fn


def get_dxdt(args, params):
    """calculate the derivatives dx/dt in the ODEs"""
    def print_intermediate_gradients(name, tensor):
            def hook(grad):
                print(f"Intermediate tensor: {name}")
                print(f"Gradient: {grad}")
                print(f"Func: {grad.grad_fn}")
                nan_count = torch.isnan(grad).sum().item()
                total_count = grad.numel()
                print(f"nan {nan_count}, norm {total_count}")
            return hook
        
    def register_hook(tensor, name):
        tensor.register_hook(print_intermediate_gradients(name, tensor))
    
    if args.ode_degree == 1:
        def weighted_sum(x, mask=None):
            if mask is not None: 
                result = torch.matmul(params['W']*mask.to(x.device), x).requires_grad_(True)
                register_hook(result, 'result')
                return result
            else: return torch.matmul(params['W'], x)
    elif args.ode_degree == 2:
        def weighted_sum(x, mask=None):
            if mask is not None: torch.matmul(params['W']*mask.to(x.device), x) + torch.reshape(torch.sum(params['W']*mask, dim=1), (args.n_x, 1)) * x
            result = (torch.matmul(params['W'], x) + torch.reshape(torch.sum(params['W'], dim=1), (args.n_x, 1)) * x).requires_grad_(True)
            register_hook(result, 'result')
            return result
    else:
        raise Exception("Illegal ODE degree. Choose from [1,2].")
    

    if args.envelope == 0:
        # epsilon*phi(Sigma+u)-alpha*x
        # return lambda x, t_mu, mask=None: params['eps'] * args.envelope_fn(weighted_sum(x, mask) + t_mu) - params['alpha'] * x
        def func(x, t_mu, mask=None):
            weighted = weighted_sum(x, mask).requires_grad_(True)
            envelope = args.envelope_fn(weighted + t_mu).requires_grad_(True)
            final = (params['eps'] * envelope - params['alpha'] * x).requires_grad_(True)
            register_hook(weighted, 'weighted')
            register_hook(envelope, 'envelope')
            register_hook(final, 'final')
            return final
        return func
    if args.envelope == 1:
        # epsilon*[phi(Sigma)+u]-alpha*x
        return lambda x, t_mu, mask=None: params['eps'] * (args.envelope_fn(weighted_sum(x, mask)) + t_mu) - params['alpha'] * x
    if args.envelope == 2:
        # epsilon*phi(Sigma)+psi*u-alpha*x
        return lambda x, t_mu, mask=None: params['eps'] * args.envelope_fn(weighted_sum(x, mask)) + params['psi'] * t_mu - \
                               params['alpha'] * x
    raise Exception("Illegal envelope type. Choose from [0,1,2].")


def get_ode_solver(args):
    """get the ODE solver based on the given argument"""
    if args.ode_solver == 'heun':
        return heun_solver
    if args.ode_solver == 'euler':
        return euler_solver
    if args.ode_solver == 'rk4':
        return rk4_solver
    if args.ode_solver == 'midpoint':
        return midpoint_solver
    raise Exception("Illegal ODE solver. Use [heun, euler, rk4, midpoint]")

# def clip_gradients(parameters, max_norm):
#     """Clip gradients of parameters to be within the specified norm."""
#     for param in parameters:
#         if param.grad is not None:
#             param.grad.data = torch.clamp(param.grad.data, min=-max_norm, max=max_norm)


def heun_solver(x, t_mu, dT, n_T, _dXdt, n_activity_nodes=None, mask=None):
    """Heun's ODE solver"""
    xs = []
    n_x = t_mu.shape[0]
    n_activity_nodes = n_x if n_activity_nodes is None else n_activity_nodes
    #dxdt_mask = tf.pad(tf.ones((n_activity_nodes, 1)), [[0, n_x - n_activity_nodes], [0, 0]])  # Add 0 rows to the end of the matrix
    dxdt_mask = nn.functional.pad(
        torch.ones((n_activity_nodes, 1)), 
        (0, 0, 0, n_x - n_activity_nodes)
    ).to(x.device).requires_grad_(True)
    for _ in range(n_T):
        dxdt_current = _dXdt(x, t_mu, mask).requires_grad_(True)
        dxdt_next = _dXdt(x + dT * dxdt_current, t_mu, mask).requires_grad_(True)
        x = x + dT * 0.5 * (dxdt_current + dxdt_next) * dxdt_mask
        # clip_gradients([dxdt_current, dxdt_next, x, dxdt_mask], max_norm=1.0)
        xs.append(x)
    xs = torch.stack(xs, dim=0)

    def print_intermediate_gradients(name, tensor):
            def hook(grad):
                print(f"Intermediate tensor: {name}")
                print(f"Gradient: {grad}")
                print(f"Func: {grad.grad_fn}")
                nan_count = torch.isnan(grad).sum().item()
                total_count = grad.numel()
                print(f"nan {nan_count}, norm {total_count}")
            return hook
        
    def register_hook(tensor, name):
        tensor.register_hook(print_intermediate_gradients(name, tensor))
    register_hook(dxdt_mask, 'dxdt_mask')
    register_hook(dxdt_current, 'dxdt_current')
    register_hook(dxdt_next, 'dxdt_next')
    return xs


def euler_solver(x, t_mu, dT, n_T, _dXdt, n_activity_nodes=None, mask=None):
    """Euler's method"""
    xs = []
    n_x = t_mu.shape[0]
    n_activity_nodes = n_x if n_activity_nodes is None else n_activity_nodes
    #dxdt_mask = tf.pad(tf.ones((n_activity_nodes, 1)), [[0, n_x - n_activity_nodes], [0, 0]])
    dxdt_mask = nn.functional.pad(
        torch.ones((n_activity_nodes, 1)), 
        (0, 0, 0, n_x - n_activity_nodes)
    ).to(x.device)
    for _ in range(n_T):
        dxdt_current = _dXdt(x, t_mu, mask)
        x = x + dT * dxdt_current * dxdt_mask
        xs.append(x)
    xs = torch.stack(xs, dim=0)
    return xs


def midpoint_solver(x, t_mu, dT, n_T, _dXdt, n_activity_nodes=None, mask=None):
    """Midpoint method"""
    xs = []
    n_x = t_mu.shape[0]
    n_activity_nodes = n_x if n_activity_nodes is None else n_activity_nodes
    #dxdt_mask = tf.pad(tf.ones((n_activity_nodes, 1)), [[0, n_x - n_activity_nodes], [0, 0]])
    dxdt_mask = nn.functional.pad(
        torch.ones((n_activity_nodes, 1)), 
        (0, 0, 0, n_x - n_activity_nodes)
    ).to(x.device)
    for _ in range(n_T):
        dxdt_current = _dXdt(x, t_mu, mask)
        dxdt_midpoint = _dXdt(x + 0.5 * dT * dxdt_current, t_mu, mask)
        x = x + dT * dxdt_midpoint * dxdt_mask
        xs.append(x)
    xs = torch.stack(xs, dim=0)
    return xs


def rk4_solver(x, t_mu, dT, n_T, _dXdt, n_activity_nodes=None, mask=None):
    """Runge-Kutta method"""
    xs = []
    n_x = t_mu.shape[0]
    n_activity_nodes = n_x if n_activity_nodes is None else n_activity_nodes
    #dxdt_mask = tf.pad(tf.ones((n_activity_nodes, 1)), [[0, n_x - n_activity_nodes], [0, 0]])
    dxdt_mask = nn.functional.pad(
        torch.ones((n_activity_nodes, 1)), 
        (0, 0, 0, n_x - n_activity_nodes)
    ).to(x.device)
    for _ in range(n_T):
        k1 = _dXdt(x, t_mu, mask)
        k2 = _dXdt(x + 0.5*dT*k1, t_mu, mask)
        k3 = _dXdt(x + 0.5*dT*k2, t_mu, mask)
        k4 = _dXdt(x + dT*k3, t_mu, mask)
        x = x + dT * (1/6*k1+1/3*k2+1/3*k3+1/6*k4) * dxdt_mask
        xs.append(x)
    xs = torch.stack(xs, dim=0)
    return xs
