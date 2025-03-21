class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        for i in range(iterations):
            d= 2* init # d= derivative of x^2 which is 2x x is initially init 
            init = init - d * learning_rate # gradient descent equation 
        return round(init,5)

        
    
