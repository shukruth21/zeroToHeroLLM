import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # note that N is just len(X) 
        # this funcction was given and i will learn in the future how to calculate such derivatives 
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights)) #linear regression equation

    learning_rate = 0.01

    def train_model(
        self, 
        X: NDArray[np.float64], 
        Y: NDArray[np.float64], 
        num_iterations: int, 
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:

        # you will need to call get_derivative() for each weight
        # and update each one separately based on the learning rate!
        # return np.round(your_answer, 5)
        for i in range(num_iterations):
            model_prediction= self.get_model_prediction(X,initial_weights)
         #correcting of weights is done individually for each 
            d1=self.get_derivative(model_prediction, Y, len(X),X,0)
            d2=self.get_derivative(model_prediction, Y, len(X),X,1)
            d3=self.get_derivative(model_prediction, Y, len(X),X,2)

            initial_weights[0] -= d1*self.learning_rate
            initial_weights[1] -= d2*self.learning_rate
            initial_weights[2] -= d3*self.learning_rate

        return np.round(initial_weights,5)
