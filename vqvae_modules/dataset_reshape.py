#transforming dataset
import numpy as np


# Generated training sequences for use in the model.
def create_sequences(values, time_steps):
    output = []
    for i in range(len(values) - time_steps):
        output.append(values.iloc[i : (i + time_steps),:])
    
    return np.stack(output)
