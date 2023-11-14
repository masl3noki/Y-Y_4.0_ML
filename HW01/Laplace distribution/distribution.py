import numpy as np

class LaplaceDistribution:    
    @staticmethod
    def mean_abs_deviation_from_median(x: np.ndarray):
        '''
        Args:
        - x: A numpy array of shape (n_objects, n_features) containing the data
          consisting of num_train samples each of dimension D.
        '''
        ####
        # Do not change the class outside of this block
        # Your code here

        # Честно, не имею понятия зачем эта функция. Чтобы вычислить b (a.k.a. self.scale)? но ее удобнее вычислить сразу в __init__()...

        ####

    def __init__(self, features):
        '''
        Args:
            feature: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        ####
        # Do not change the class outside of this block
        self.loc = np.median(features, axis=0)                   # Обозначение для mu (см. .ipynb) 
        self.scale = np.mean(np.abs(features - self.loc), axis=0) # Обозначение для b (см. .ipynb)
        ####

    def logpdf(self, values):
        '''
        Returns logarithm of probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        ####
        # Do not change the class outside of this block
        
        # Все таки экспоненту тяжело считать. Поэтому мы посчитаем ln(PDF):

        lgPDF = - np.log(2*self.scale) - np.abs(values - self.loc) / self.scale #ln(1/2b) = ln(1) - ln(2b) = - ln(2b)
        return lgPDF
        ####
        
    
    def pdf(self, values):
        '''
        Returns probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        return np.exp(self.logpdf(values)) # Кажется, тут была опечатка: не value, a values