# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 12:51:34 2025

@author: Jared Praniewicz
Licensing:
For personal and academic uses: GPL 3.0
For commercial and governmental use: contact me

This is an example implementation of the linearized gradient descent (LGD) 
optimization algorithm. In my PhD dissertation I prove that the LGD algorithm
converges in parameter for any function that is: in C^2 bounded below and 
coercive, the search region is bounded and convex, and the base learning rate
is small enough.

In practice, adding noise to the gradient is a useful tool for avoiding sharp
local minima which is why this algorithm works well in applications for 
deep neural networks.
"""
import numpy as np

def numerical_grad(func, x, h):
    """
    numerical_grad
    Inputs:
        1) func: f:R^N -> R and is gateaux differentiable
        2) x: x in R^N is the evaluation location
        3) h: h in R+ is the perturbation.
    outputs:
        1) grad: a numpy vector of the numerical gradient
    """
    grad = np.zeros(len(x))
    for i in range(0, len(grad)):
        h_vec = np.zeros(len(x))
        h_vec[i] = h
        grad[i] = (func(x + h_vec) - func(x - h_vec)) / (2*h)
    return grad

def linearized_gradient_descent(func, x, alpha = 10**-2,
                                numerical_diff = False, h = 10**-3,
                                threshold = 10**-6, kappa = 10**-6):
    """
    linearized_gradient_descent
    Inputs:
        1) func a function that maps R^N -> R that has the needed properties
        2) x a starting point in R^N
        3) alpha is the base learning rate
        4)numerical diff is a boolean that is true if the gradient is calculated 
        numerically
        5) h is the bump size for numerical gradients
        6) threshold is the termination condition (i.e. steps smaller than)
        7) kappa is a small positive number for numerical stability. Theoretically,
        it isn't needed but in some situations it helps.
    Outputs:
        1) 'value': the critical point location
        2) 'error': the absolute difference in function value at the last step
        3) 'mag_grad': the magnitude of the gradient from the last step
    """
    f_current = 0
    f_prev = 2 * threshold
    grad = 0
    first_step = True
    while abs(f_prev - f_current) > threshold:
        f_prev = f_current
        grad_prev = grad
        #Compute function value and gradient
        if numerical_diff is False:
            f_current, grad = func(x)
        else:
            grad = numerical_grad(func, x, h)
            f_current = func(x)
        #Take a regular gradient descent step
        if first_step:
            first_step = False
            alpha_current = alpha
            x = x - alpha_current * grad
        #Take a linearized step
        else:
            mag_sq_grad = np.dot(grad, grad)
            mag_sq_grad_prev = np.dot(grad_prev, grad_prev)
            pred_error = f_current - f_prev + alpha_current * mag_sq_grad_prev
            scaling_factor = pred_error / (mag_sq_grad + kappa)
            alpha_current = alpha + scaling_factor * grad ** 2 / mag_sq_grad
            x = x - alpha_current * grad
    results = {}
    results['value'] = x
    results['error'] = abs(f_prev - f_current)
    results['mag_grad'] = mag_sq_grad ** 0.5
    return results
if __name__ == '__main__':
    def f(x):
        """f
        x_1^2 + x_2^2 + x_1 + x_2 + 1
        This uses analytic derviative.
        """
        y = np.dot(x, x) + np.sum(x) + 1
        df = 2 * x + 1
        return (y, df)
    critical_point_f = linearized_gradient_descent(f, np.array([-1,1]))
    def g(x):
        """g
        x_1^2 + x_2^2 + x_1 + x_2 + 1
        This uses numerical derviative.
        """
        y = np.dot(x, x) + np.sum(x) + 1
        return y
    critical_point_g = linearized_gradient_descent(g,
                                                   np.array([-1,1]),
                                                   10**-2, True)
    