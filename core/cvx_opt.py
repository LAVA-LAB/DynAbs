# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 19:29:47 2022

@author: Thom Badings
"""

import cvxpy as cp
import numpy as np

class abstraction_error(object):
    
    def __init__(self, model, no_verts):
        
        self.A = model.A
        
        # Number of vertices in backward reachable set
        v = 2 ** model.p
        
        self.vertex = cp.Parameter((no_verts, model.n))
        self.G_curr = cp.Parameter((v, model.n))
        
        self.x = cp.Variable((no_verts, model.n))
        self.alpha = cp.Variable((no_verts, v), nonneg=True)
        
        # Point x is a convex combination of the backward reachable set vertices,
        # and is within the specified region
        self.constraints = [sum(self.alpha[w]) == 1 for w in range(no_verts)] + \
            [self.x[w] == cp.sum([self.G_curr[i] * self.alpha[w][i] for i in range(v)]) for w in range(no_verts)]
        
        self.error_direction = [cp.sum(self.x[w] - self.vertex[w]) == 0 for w in range(no_verts)]
        
        if 'max_control_error' in model.setup:
            self.constraints += \
                [self.A @ (self.vertex[w] - self.x[w]) <= model.setup['max_control_error'] for w in range(no_verts)] + \
                [self.A @ (self.vertex[w] - self.x[w]) >=-model.setup['max_control_error'] for w in range(no_verts)]
        
        # self.obj = cp.Minimize(cp.sum([cp.norm2(self.x[w] - self.vertex[w]) for w in range(no_verts)]))
        self.obj = cp.Minimize(cp.sum([
                            (1/2)*cp.quad_form(self.x[w], np.eye(model.n)) - self.vertex[w].T @ self.x[w] 
                            # + 10*cp.sum(self.x[w] - self.vertex[w])
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
            return True, None, None, None
        
        error = (self.A @ (vertices - self.x.value).T).T
        
        if len(vertices.shape) == 1:
            # If single vertex/point is provided, return for that point only
            return False, self.x.value[0], error, error

        else:
            # If multiple vertices/points provided, return max/min among them
            error_pos = np.max(error, axis=0)
            error_neg = np.min(error, axis=0)
            
            return False, self.x.value, error_pos, error_neg