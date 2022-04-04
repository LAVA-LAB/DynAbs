# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 19:29:47 2022

@author: Thom Badings
"""

import cvxpy as cp
import numpy as np

class abstraction_error(object):
    
    def __init__(self, model, spec, no_verts):
        
        self.A = model.A
        
        # Number of vertices in backward reachable set
        v = 2 ** model.p
        
        self.vertex = cp.Parameter((no_verts, model.n))
        self.G_curr = cp.Parameter((v, model.n))
        
        self.x = cp.Variable((no_verts, model.n))
        self.alpha = cp.Variable((no_verts, v), nonneg=True)
        
        self.e = cp.Variable((no_verts, model.n))
        
        # Point x is a convex combination of the backward reachable set vertices,
        # and is within the specified region
        self.constraints = [sum(self.alpha[w]) == 1 for w in range(no_verts)] + \
            [self.x[w] == cp.sum([self.G_curr[i] * self.alpha[w][i] for i in range(v)]) for w in range(no_verts)] + \
            [self.e[w] == self.A @ (self.vertex[w] - self.x[w]) for w in range(no_verts)]
        
        if 'max_control_error' in spec.error:
            self.constraints += \
                [self.A @ (self.vertex[w] - self.x[w]) <= spec.error['max_control_error'][:,1] for w in range(no_verts)] + \
                [self.A @ (self.vertex[w] - self.x[w]) >= spec.error['max_control_error'][:,0] for w in range(no_verts)]
        
        # self.obj = cp.Minimize(cp.sum([cp.norm2(self.x[w] - self.vertex[w]) for w in range(no_verts)]))
        # self.obj = cp.Minimize(cp.sum([
        #                     cp.quad_form(self.x[w] - self.vertex[w], np.diag([10,1]))
        #                     for w in range(no_verts)
        #                     ]))
        
        self.obj = cp.Minimize(cp.sum([
                            cp.quad_form(self.e[w], np.eye(model.n))
                            for w in range(no_verts)
                            ]))
        
        self.prob = cp.Problem(self.obj, self.constraints)
        
    def solve(self, vertices, G):
        
        self.G_curr.value = G
        
        if len(vertices.shape) == 1:
            self.vertex.value = np.array([vertices])
        else:
            self.vertex.value = vertices
        
        self.prob.solve(warm_start = True, solver='ECOS')
        
        if self.prob.status == 'infeasible':
            return True, None, None
        else: 
            projection = self.x.value - vertices
            project_vec = np.array([row/row[0] for row in projection])
            
            return False, project_vec, self.x.value
        
class LP_vertices_contained(object):
    
    def __init__(self, model, G, solver):
        
        self.solver = solver
        
        v = 2 ** model.n
        w = len(G)
        
        self.G_curr     = cp.Parameter(G.shape)
        self.P_vertices = cp.Parameter((v, model.n))
        self.alpha      = cp.Variable((v, w), nonneg=True)
        
        constraints = [cp.sum(self.alpha[i]) == 1 for i in range(v)] + \
            [self.P_vertices[j] == cp.sum([self.alpha[j,i] * self.G_curr[i] 
                                       for i in range(w)]) for j in range(v)]
            
        obj = cp.Minimize(1)
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