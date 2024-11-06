import numpy as np
import matplotlib.pyplot as plt

def wrap_circular_value(input_value):
    return (input_value + np.pi) % (2*np.pi) - np.pi


class BicycleModel:
    xdim: int = 6
    xnames = ["x_I", "y_I", "theta", "vx_B", "vy_B", "omega"]
    udim: int = 2
    unames = ["u_v", "u_steering"]
    action_lims = lambda self: np.array([[-5.0, 5.0],
                                        [-np.deg2rad(28), np.deg2rad(28)]])

    def __init__(self, params: dict):
        self.dt = params['dt']
        self.params = params

    def dynamics(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        x, y, theta, vx_B, vy_B, omega = state
        u_v, u_steering = action
        
        # Params.
        L = self.params['L']
        mass = self.params['mass']
        Cy = self.params['Cy']
        Iz = self.params['Iz']
        tau = self.params['tau']

        vx_dot = 1/tau * (u_v - vx_B)
        vy_dot = - Cy/(mass * vx_B) * vy_B - vx_B * omega + Cy/mass * u_steering
        omega_dot = -L**2 * Cy/(2 * Iz * vx_B) * omega + L * Cy/mass * u_steering
    
        vx_B = vx_B + self.dt * vx_dot
        vy_B = vy_B + self.dt * vy_dot
        omega = omega + self.dt * omega_dot

        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])

        v_B = np.array([vx_B, vy_B])
        v_I = R @ v_B
        x = x + self.dt * v_I[0]
        y = y + self.dt * v_I[1]
        theta = theta + self.dt * omega
        # Note, you can also pick a different representation for theta, then you don't need to wrap the angle.
        theta = wrap_circular_value(theta)

        next_state = np.array([x, y, theta, vx_B, vy_B, omega])

        return next_state

def plot_states(states: np.ndarray, title: str):
    # Plot states.
    nb_cols = 5
    nb_rows = 1
    fig, ax = plt.subplots(nb_rows, nb_cols, figsize=(20, 3))
    plt.suptitle(title)

    ax[0].plot(states[:, 0], states[:, 1], label='trajectory')
    ax[1].plot(states[:, 2], label='thetas')

    ax[2].plot(states[:, 3], label='vs_B_x')
    ax[3].plot(states[:, 4], label='vs_B_y')

    ax[4].plot(states[:, 5], label='omegas')

    for i in range(nb_cols):
        ax[i].legend(loc='upper right')
        ax[i].grid()
    plt.tight_layout()

    return fig, ax

def test_system():
    params = {"dt": 0.01,
              "tau": 0.1,
              'L': 0.4,
              'Cy': 100,
              'mass': 11.5,
              'Iz': 0.5,
    }

    # Set the model.
    model = BicycleModel(params=params)

    # Initial condition.
    state0 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    action = np.array([1.0, -1.0])
    t = np.arange(0, 30, model.dt)
    states = np.zeros((len(t), model.xdim))
    states[0] = state0
    for i in range(1, len(t)):
        states[i] = model.dynamics(states[i-1], action)

    fig, ax = plot_states(states, "Bicycle Model Performance for u_v=1.0, u_steering=-1.0")
    plt.savefig("system_plots/bicycle_model.pdf")
    
    # Design a controller
    # Initial condition.
    state0 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    kp = 1.0
    v_d_x = 2.0
    b = 1/params['tau']
    a = -1/params['tau']
    t = np.arange(0, 3, model.dt)
    states = np.zeros((len(t), model.xdim))
    states[0] = state0
    action = np.array([0.0, 0.0])
    for i in range(1, len(t)):
        vel_x_B = states[i-1, 3]
        u_v = -1/b*(kp * (vel_x_B - v_d_x) + a * v_d_x)
        action = np.array([u_v, 0.0])
        states[i] = model.dynamics(states[i-1], action)

    _, ax = plot_states(states, "Bicycle Model Performance for Control")
    ax[2].plot(np.ones_like(t) * v_d_x, label='v_d_x')
    plt.savefig("system_plots/bicycle_model_control.pdf")

# main
if __name__ == "__main__":
    test_system()