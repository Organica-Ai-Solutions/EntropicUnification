import torch


def fixed_finite_difference(tensor, order=1, axis=0, dx=1.0):
    """
    A robust implementation of finite difference that handles different tensor dimensions.

    Args:
        tensor: Input tensor to take derivatives of
        order: Order of the derivative (1 or 2)
        axis: Axis along which to take the derivative
        dx: Grid spacing (default 1.0). Derivatives are divided by dx (order 1)
            or dx² (order 2) so that results have correct physical units.

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
            
            # Central difference: (f_{i+1} - f_{i-1}) / (2 dx)
            result[tuple(s_curr)] = (tensor[tuple(s_next)] - tensor[tuple(s_prev)]) / (2.0 * dx)

        # One-sided stencils at the boundary. These must be SECOND order to
        # match the interior: a two-point edge difference is O(dx), and the
        # lower order then dominates the global error norm wherever the field
        # varies near the boundary — which silently degrades the whole scheme
        # to O(dx^1.5) in practice.
        def _sided(i0, i1, i2, sign):
            s0 = [slice(None)] * len(shape); s0[axis] = i0
            s1 = [slice(None)] * len(shape); s1[axis] = i1
            s2 = [slice(None)] * len(shape); s2[axis] = i2
            # (-3f0 + 4f1 - f2) / (2 dx), mirrored for the far edge
            return sign * (-3.0 * tensor[tuple(s0)] + 4.0 * tensor[tuple(s1)]
                           - tensor[tuple(s2)]) / (2.0 * dx), tuple(s0)

        if axis_size > 2:
            val, idx = _sided(0, 1, 2, 1.0)
            result[idx] = val
            val, idx = _sided(axis_size - 1, axis_size - 2, axis_size - 3, -1.0)
            result[idx] = val
        elif axis_size > 1:
            # too few points for a second-order stencil; fall back
            s_first = [slice(None)] * len(shape); s_first[axis] = 0
            s_second = [slice(None)] * len(shape); s_second[axis] = 1
            edge = (tensor[tuple(s_second)] - tensor[tuple(s_first)]) / dx
            result[tuple(s_first)] = edge
            result[tuple(s_second)] = edge
    
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
            
            # Central difference for second derivative: (f_{i+1} - 2f_i + f_{i-1}) / dx²
            result[tuple(s_curr)] = (
                tensor[tuple(s_next)] - 2 * tensor[tuple(s_curr)] + tensor[tuple(s_prev)]
            ) / (dx * dx)

        # Second-order one-sided second derivative at the boundary:
        # (2f0 - 5f1 + 4f2 - f3) / dx^2.  The three-point form previously used
        # here is only O(dx) when evaluated at the edge point.
        def _sided2(i0, i1, i2, i3):
            idx = []
            for i in (i0, i1, i2, i3):
                sl = [slice(None)] * len(shape); sl[axis] = i
                idx.append(tuple(sl))
            return (2.0 * tensor[idx[0]] - 5.0 * tensor[idx[1]]
                    + 4.0 * tensor[idx[2]] - tensor[idx[3]]) / (dx * dx), idx[0]

        if axis_size > 3:
            val, idx = _sided2(0, 1, 2, 3)
            result[idx] = val
            val, idx = _sided2(axis_size - 1, axis_size - 2,
                               axis_size - 3, axis_size - 4)
            result[idx] = val
        elif axis_size > 2:
            # too few points for the four-point stencil; keep the O(dx) form
            for i0, i1, i2 in ((0, 1, 2), (axis_size - 1, axis_size - 2, axis_size - 3)):
                s0 = [slice(None)] * len(shape); s0[axis] = i0
                s1 = [slice(None)] * len(shape); s1[axis] = i1
                s2 = [slice(None)] * len(shape); s2[axis] = i2
                result[tuple(s0)] = (tensor[tuple(s2)] - 2 * tensor[tuple(s1)]
                                     + tensor[tuple(s0)]) / (dx * dx)
    
    else:
        raise ValueError("Only 1st and 2nd order derivatives are supported")
    
    return result


# Higher-order methods can be added here
