# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 19:29:47 2022

@author: Thom Badings
"""

import cvxpy as cp
import numpy as np

class Controller(object):
    
    def __init__(self, model):
        
        self.A = model.A
        self.B = model.B
        self.Q = model.Q_flat
        
        self.target = cp.Parameter(model.n)
        self.x = cp.Parameter(model.n)
        self.u = cp.Variable(model.p)
        self.x_plus = cp.Variable(model.n)
        
        self.error_neg = cp.Parameter(model.n)
        self.error_pos = cp.Parameter(model.n)
        
        self.constraints = [
            self.x_plus == self.A @ self.x + self.B @ self.u + self.Q,
            self.x_plus - self.target >= self.error_neg,
            self.x_plus - self.target <= self.error_pos,
            self.u <= model.uMax,
            self.u >= model.uMin
            ]
        
        self.obj = cp.Minimize(
            cp.quad_form(self.x_plus - self.target, np.eye(model.n))
                            )
        
        self.prob = cp.Problem(self.obj, self.constraints)
        
    def solve(self, target, x, error):
        
        self.target.value = target
        self.x.value = x
        
        self.error_neg.value = error[:,0]
        self.error_pos.value = error[:,1]
        
        self.prob.solve(warm_start = True, solver='ECOS')
        
        if self.prob.status == 'infeasible':
            return False, None, None
        else: 
            return True, self.x_plus.value, self.u.value
        
class LP_vertices_contained(object):
    
    def __init__(self, model, G_shape, solver):
        
        self.solver = solver
        
        v = 2 ** model.n
        w = G_shape[0]
        
        self.G_curr     = cp.Parameter(G_shape)
        self.P_vertices = cp.Parameter((v, model.n))
        self.alpha      = cp.Variable((v, w), nonneg=True)
        
        constraints = [cp.abs(cp.sum(self.alpha[i]) - 1) <= 1e-3 for i in range(v)] + \
            [self.P_vertices[j] == cp.sum([self.alpha[j,i] * self.G_curr[i] 
                                       for i in range(w)]) for j in range(v)]
            
        obj = cp.Minimize(cp.sum(self.alpha))
        self.prob = cp.Problem(obj, constraints)
        
    def set_backreach(self, BRS_inflated):
        
        # Set current backward reachable set as parameter
        self.G_curr.value = BRS_inflated
        
    def solve(self, vertices):
        
        self.P_vertices.value = vertices
        
        if self.solver == 'GUROBI':
            self.prob.solve(warm_start = True, solver='GUROBI')
        elif self.solver == 'ECOS':
            self.prob.solve(warm_start = True, solver='ECOS')
        elif self.solver == 'OSQP':
            self.prob.solve(warm_start = True, solver='OSQP', eps_abs=1e-4, eps_rel=1e-4)
        elif self.solver == 'SCS':
            self.prob.solve(warm_start = True, solver='SCS')
        else:
            self.prob.solve(warm_start = True)
        
        if self.prob.status != "infeasible":
            return True
        else:
            return False