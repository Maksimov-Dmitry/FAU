import numpy as np

class CrossEntropyLoss:
    
    def forward(self, prediction_tensor: np.ndarray, label_tensor: np.ndarray) -> float:
        self.predicted = np.clip(prediction_tensor, np.finfo(float).eps, None)
        return -np.sum(label_tensor * np.log(self.predicted))

    def backward(self, label_tensor: np.ndarray) -> np.ndarray:
        return label_tensor * -1/self.predicted
