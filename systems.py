import numpy as np

def wrap_circular_value(input_value):
    return (input_value + np.pi) % (2*np.pi) - np.pi

class BicycleModel:
    xdim: int = 6
    xnames = ["x_I", "y_I", "theta", "vx_B", "vy_B", "omega"]
    udim: int = 2
    unames = ["u_v", "u_steering"]
    action_lims = lambda self: np.array([[-5.0, 5.0],
                                        [-np.deg2rad(28), np.deg2rad(28)]])

    def dynamics(self, state, action, dt):
        x, y, theta, vx, vy, omega = state
        u_v, u_steering = action

        # vx_dot = ...
        # vy_dot = ...
        # omega_dot = ...
        
        vx_dot = 0.0
        vy_dot = 0.0
        omega_dot = 0.0

        vx_B = vx_B + dt * vx_dot
        vy_B = vy_B + dt * vy_dot
        omega = omega + dt * omega_dot

        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])

        v_B = np.array([vx_B, vy_B])
        v_I = R @ v_B
        x = x + dt * v_I[0]
        y = y + dt * v_I[1]
        theta = theta + dt * omega
        # Note, you can also pick a different representation for theta, then you don't need to wrap the angle.
        theta = wrap_circular_value(theta)

        next_state = np.array([x, y, theta, vx_B, vy_B, omega])

        return next_state