import torch


def fixed_finite_difference(tensor, order=1, axis=0):
    """
    A robust implementation of finite difference that handles different tensor dimensions.
    
    Args:
        tensor: Input tensor to take derivatives of
        order: Order of the derivative (1 or 2)
        axis: Axis along which to take the derivative
        
    Returns:
        Tensor containing the finite difference approximation
    """
    # Get tensor shape
    shape = tensor.shape
    
    # Check if the axis is valid
    if axis >= len(shape):
        raise ValueError(f"Axis {axis} is out of range for tensor with {len(shape)} dimensions")
    
    # Initialize result tensor
    result = torch.zeros_like(tensor)
    
    # Get the size along the derivative axis
    axis_size = shape[axis]
    
    if order == 1:
        # First derivative
        # For interior points, use central difference
        for i in range(1, axis_size - 1):
            # Create slices for i-1, i, and i+1
            s_prev = [slice(None)] * len(shape)
            s_curr = [slice(None)] * len(shape)
            s_next = [slice(None)] * len(shape)
            
            s_prev[axis] = i - 1
            s_curr[axis] = i
            s_next[axis] = i + 1
            
            # Central difference
            result[tuple(s_curr)] = (tensor[tuple(s_next)] - tensor[tuple(s_prev)]) / 2.0
        
        # Forward difference for first point
        if axis_size > 1:
            s_first = [slice(None)] * len(shape)
            s_second = [slice(None)] * len(shape)
            s_first[axis] = 0
            s_second[axis] = 1
            
            result[tuple(s_first)] = tensor[tuple(s_second)] - tensor[tuple(s_first)]
        
        # Backward difference for last point
        if axis_size > 1:
            s_last = [slice(None)] * len(shape)
            s_second_last = [slice(None)] * len(shape)
            s_last[axis] = axis_size - 1
            s_second_last[axis] = axis_size - 2
            
            result[tuple(s_last)] = tensor[tuple(s_last)] - tensor[tuple(s_second_last)]
    
    elif order == 2:
        # Second derivative
        # For interior points, use central difference
        for i in range(1, axis_size - 1):
            # Create slices for i-1, i, and i+1
            s_prev = [slice(None)] * len(shape)
            s_curr = [slice(None)] * len(shape)
            s_next = [slice(None)] * len(shape)
            
            s_prev[axis] = i - 1
            s_curr[axis] = i
            s_next[axis] = i + 1
            
            # Central difference for second derivative
            result[tuple(s_curr)] = tensor[tuple(s_next)] - 2 * tensor[tuple(s_curr)] + tensor[tuple(s_prev)]
        
        # Forward difference for first point
        if axis_size > 2:
            s_first = [slice(None)] * len(shape)
            s_second = [slice(None)] * len(shape)
            s_third = [slice(None)] * len(shape)
            s_first[axis] = 0
            s_second[axis] = 1
            s_third[axis] = 2
            
            result[tuple(s_first)] = tensor[tuple(s_third)] - 2 * tensor[tuple(s_second)] + tensor[tuple(s_first)]
        
        # Backward difference for last point
        if axis_size > 2:
            s_last = [slice(None)] * len(shape)
            s_second_last = [slice(None)] * len(shape)
            s_third_last = [slice(None)] * len(shape)
            s_last[axis] = axis_size - 1
            s_second_last[axis] = axis_size - 2
            s_third_last[axis] = axis_size - 3
            
            result[tuple(s_last)] = tensor[tuple(s_last)] - 2 * tensor[tuple(s_second_last)] + tensor[tuple(s_third_last)]
    
    else:
        raise ValueError("Only 1st and 2nd order derivatives are supported")
    
    return result


# Higher-order methods can be added here
