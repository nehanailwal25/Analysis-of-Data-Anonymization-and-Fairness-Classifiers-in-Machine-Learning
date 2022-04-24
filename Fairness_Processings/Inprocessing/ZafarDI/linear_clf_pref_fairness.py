from __future__ import division
import os,sys
import numpy as np
import traceback

sys.path.insert(0, "/home/mzafar/libraries/dccp") # we will store the latest version of DCCP here.
from cvxpy import *
import dccp
from dccp.problem import is_dccp


class LinearClf():


    def __init__(self, loss_function, lam=None, train_multiple=False, random_state=1234):

        """ Setting default lam val and Making sure that lam is provided for each group """
        if lam is None:
            if train_multiple == False: 
                lam = 0.0
            else: 
                lam = {0:0.0, 1:0.0}
                
        else:
            if train_multiple == True:
                assert(isinstance(lam, dict))

        self.loss_function = loss_function
        self.lam = lam
        self.train_multiple = train_multiple

        np.random.seed(random_state)


    def fit(self, X, y, x_sensitive, cons_params=None):

        #Setting up the initial variables

        max_iters = 100 # for CVXPY convex solver
        max_iter_dccp = 50  # for the dccp. notice that DCCP hauristic runs the convex program iteratively until arriving at the solution
        

        #Construct the optimization variables

        constraints = []

        np.random.seed(1234) # set the seed before initializing the values of w
        if self.train_multiple == True:
            w = {}
            for k in set(x_sensitive):
                w[k] = Variable(X.shape[1]) # this is the weight vector
                w[k].value = np.random.rand(X.shape[1]) # initialize the value of w -- uniform distribution over [0,1]
        else:
            w = Variable(X.shape[1]) # this is the weight vector
            w.value = np.random.rand(X.shape[1])

        #optimization problem
        num_all = X.shape[0] # set of all data points

        if self.train_multiple == True:
            
            obj = 0
            for k in set(x_sensitive):
                idx = x_sensitive==k
                X_k = X[idx]
                y_k = y[idx]
                obj += sum_squares(w[k][1:]) * self.lam[k] # first term in w is the intercept, so no need to regularize that

                if self.loss_function == "logreg":
                    obj += sum_entries(  logistic( mul_elemwise(-y_k, X_k*w[k]) )  ) / num_all # notice that we are dividing by the length of the whole dataset, and not just of this sensitive group. this way, the group that has more people contributes more to the loss
                    
                elif self.loss_function == "svm_linear":
                    obj += sum_entries ( max_elemwise (0, 1 - mul_elemwise ( y_k,  X_k*w[k])) ) / num_all
                    
                else:
                    raise Exception("Invalid loss function")

        else:

            obj = 0
            obj += sum_squares(w[1:]) * self.lam # regularizer -- first term in w is the intercept, so no need to regularize that
            if self.loss_function == "logreg":
                obj += sum_entries(  logistic( mul_elemwise(-y, X*w) )  ) / num_all
            elif self.loss_function == "svm_linear":
                obj += sum_entries ( max_elemwise (0, 1 - mul_elemwise ( y,  X*w)) ) / num_all
            else:
                raise Exception("Invalid loss function")


        #setting up Constraints
        if cons_params is not None:
            
            cons_type = cons_params["cons_type"]
            if cons_type == -1: # no constraint
                pass
            elif cons_type == 0: # disp imp with single boundary
                cov_thresh = np.abs(0.) # perfect fairness -- see our AISTATS paper for details
                constraints += self.get_di_cons_single_boundary(X, y, x_sensitive, w, cov_thresh)
            elif cons_type in [1,3]: # preferred imp, pref imp + pref treat
                constraints += self.get_preferred_cons(X, x_sensitive, w, cons_type, cons_params["s_val_to_cons_sum"])
            elif cons_type == 2:
                constraints += self.get_preferred_cons(X, x_sensitive, w, cons_type)
            else:
                raise Exception("Wrong constraint type")

        prob = Problem(Minimize(obj), constraints)

        #Solving the problem

        try:

            tau, mu, EPS = 0.5, 1.2, 1e-4 # default dccp parameters, need to be varied per dataset
            if cons_params is not None: # in case we passed these parameters as a part of dccp constraints
                if cons_params.get("tau") is not None: tau = cons_params["tau"]
                if cons_params.get("mu") is not None: mu = cons_params["mu"]
                if cons_params.get("EPS") is not None: EPS = cons_params["EPS"]

            prob.solve(method='dccp', tau=tau, mu=mu, tau_max=1e10,
                verbose=False, 
                feastol=EPS, abstol=EPS, reltol=EPS,feastol_inacc=EPS, abstol_inacc=EPS, reltol_inacc=EPS,
                max_iters=max_iters, max_iter=max_iter_dccp)

            assert(prob.status == "Converged" or prob.status == "optimal")

            for f_c in constraints:
                try:
                    assert(f_c.value == True)
                except:
                    print("Assertion failed. Fairness constraints not satisfied.")
                    print(traceback.print_exc())
                    sys.stdout.flush()
                    return

        except:
            traceback.print_exc()
            sys.stdout.flush()
            return


        #Storing the results

        if self.train_multiple == True:
            self.w = {}
            for k in set(x_sensitive):
                self.w[k] = np.array(w[k].value).flatten() # flatten converts it to a 1d array
        else:
            self.w = np.array(w.value).flatten() # flatten converts it to a 1d array
        
        
        return self.w

    #for defining the decision boundary
    def decision_function(self, X, k=None):

        if k is None:
            ret = np.dot(X, self.w)
        else:
            ret = np.dot(X, self.w[k])
        
        return ret


    def get_distance_boundary(self, X, x_sensitive):

        distances_boundary_dict = {} # s_attr_group (0/1) -> w_group (0/1) -> distances

        if not isinstance(self.w, dict): # we have one model for the whole data
            distance_boundary_arr = self.decision_function(X)

            for attr in set(x_sensitive): # there is only one boundary, so the results with this_group and other_group boundaries are the same

                distances_boundary_dict[attr] = {}
                idx = x_sensitive == attr

                for k in set(x_sensitive):
                    distances_boundary_dict[attr][k] = self.decision_function(X[idx]) # apply same decision function for all the sensitive attrs because same w is trained for everyone

            
        else:
            distance_boundary_arr = np.zeros(X.shape[0])

            for attr in set(x_sensitive):

                distances_boundary_dict[attr] = {}
                idx = x_sensitive == attr
                X_g = X[idx]

                distance_boundary_arr[idx] = self.decision_function(X_g, attr) # each group gets decision with their own boundary

                for k in self.w.keys(): 
                    distances_boundary_dict[attr][k] = self.decision_function(X_g, k) # each group gets a decision with both boundaries

        return distance_boundary_arr, distances_boundary_dict


    def get_di_cons_single_boundary(self, X, y, x_sensitive, w, cov_thresh):

        #Parity impact constraint

        assert(self.train_multiple == False) # di cons is just for a single boundary clf
        assert(cov_thresh >= 0) # covariance thresh has to be a small positive number

        constraints = []
        z_i_z_bar = x_sensitive - np.mean(x_sensitive)

        fx = X*w
        prod = sum_entries( mul_elemwise(z_i_z_bar, fx) ) / X.shape[0]

        constraints.append( prod <=  cov_thresh )
        constraints.append( prod >= -cov_thresh )

        return constraints


    def get_preferred_cons(self, X, x_sensitive, w, cons_type, s_val_to_cons_sum=None):

        constraints = []

        if cons_type in [1,2,3]: # 1 - pref imp, 2 - EF, 3 - pref imp & EF

            prod_dict = {0:{}, 1:{}} # s_attr_group (0/1) -> w_group (0/1) -> val
            for val in set(x_sensitive):
                idx = x_sensitive == val
                X_g = X[idx]
                num_g = X_g.shape[0]

                for k in w.keys(): # get the distance with each group's w
                    prod_dict[val][k] = sum_entries(  max_elemwise(0, X_g*w[k])   ) / num_g

        else:
            raise Exception("Invalid constraint type")


        if cons_type == 1 or cons_type == 3: # 1 for preferred impact -- 3 for preferred impact and envy free
            
            constraints.append( prod_dict[0][0] >= s_val_to_cons_sum[0][0] )
            constraints.append( prod_dict[1][1] >= s_val_to_cons_sum[1][1] )


        if cons_type == 2 or cons_type == 3: # envy free
            constraints.append( prod_dict[0][0] >= prod_dict[0][1] )
            constraints.append( prod_dict[1][1] >= prod_dict[1][0] )
                
        return constraints
