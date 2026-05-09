import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        arr = []
        
        for x in z:
            arr.append(x - max(z))
        
        exp_result = np.exp(arr)
        total = np.sum(exp_result)
        prob = exp_result / total
        return np.round(prob, 4)

        pass
