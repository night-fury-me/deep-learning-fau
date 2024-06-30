# import numpy as np #type: ignore

# class L1_Regularizer:
#     def __init__(self, alpha):
#         self.alpha = alpha

#     def calculate_gradient(self, weight_tensor):
#         return self.alpha * np.sign(weight_tensor)

#     def norm(self, weight_tensor):
#         return self.alpha * np.sum(np.abs(weight_tensor))

# class L2_Regularizer:
#     def __init__(self, alpha):
#         self.alpha = alpha

#     def calculate_gradient(self, weight_tensor):
#         return self.alpha * weight_tensor

#     def norm(self, weight_tensor):
#         return self.alpha * np.sum(weight_tensor ** 2)


import numpy as np

class L2_Regularizer(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def norm(self, weight_tensor):
        return self.alpha * np.sum(np.square(weight_tensor))
    
    def calculate_gradient(self, weight_tensor):
        return self.alpha * weight_tensor

class L1_Regularizer(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def norm(self, weight_tensor):
        return self.alpha * np.sum(np.abs(weight_tensor))
    
    def calculate_gradient(self, weight_tensor):
        return np.sign(weight_tensor) * self.alpha