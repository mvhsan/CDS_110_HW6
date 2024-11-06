import math
import numpy as np
import matplotlib.pyplot as plt

def wrap_circular_value(input_value):
    return (input_value + np.pi) % (2*np.pi) - np.pi

def compute_fig8_simple(period, length, current_time, initial_state_I):
    t = current_time
    omega = 2 * np.pi / period
    x = length * np.sin(omega * t)
    y = length/2  * np.sin(2 * omega * t)
    z = 0.0
    vel_x = length * omega * np.cos(omega * t)
    vel_y = length * omega * np.cos(2 * omega * t)
    vel_z = 0.0

    fig_8_start_heading = initial_state_I[2] - np.pi/4
    R = np.array([[np.cos(fig_8_start_heading), -np.sin(fig_8_start_heading)],
                [np.sin(fig_8_start_heading), np.cos(fig_8_start_heading)]])
    x, y = R @ np.array([x, y]) + initial_state_I[0:2]
    vel_x, vel_y = R @ np.array([vel_x, vel_y])
    return x, y, z, vel_x, vel_y, vel_z

def compute_circle_start_on_circle(angle, dt, v_desired, initial_state_I):
    R = 2.2
    omega = v_desired / R
    angle = angle + dt * omega

    x0, y0, th0 = initial_state_I[0], initial_state_I[1], initial_state_I[2]
    x = x0 - R * math.sin(th0) + R * math.sin((angle + th0))
    y = y0 + R * math.cos(th0) - R * math.cos((angle + th0))
    z = 0.0
    x_dot = R * math.cos(angle + th0) * omega
    y_dot = R * math.sin(angle + th0) * omega
    z_dot = 0.0

    return x, y, z, x_dot, y_dot, z_dot, angle


def plot_states(traj: np.ndarray, title='Title'):
    # Plot traj.
    nb_cols = 5
    nb_rows = 1
    fig, ax = plt.subplots(nb_rows, nb_cols, figsize=(20, 3))
    plt.suptitle(title)

    ax[0].plot(traj[:, 0], traj[:, 1], label='trajectory')
    ax[1].plot(traj[:, 2], label='thetas')
    ax[1].set_ylim(min(traj[:, 2]) - 1, max(traj[:, 2]) + 1)

    ax[2].plot(traj[:, 3], label='vs_B_x')
    ax[2].set_ylim(min(traj[:, 3]) - 1, max(traj[:, 3]) + 1)
    ax[3].plot(traj[:, 4], label='vs_B_y')
    ax[3].set_ylim(min(traj[:, 4]) - 1, max(traj[:, 4]) + 1)

    ax[4].plot(traj[:, 5], label='omegas')
    ax[4].set_ylim(min(traj[:, 5]) - 1, max(traj[:, 5]) + 1)

    for i in range(nb_cols):
        ax[i].legend(loc='upper right')
        ax[i].grid()
    plt.tight_layout()

    return fig, ax

def transform_traj(traj_states):
    traj_states_new_frame = np.zeros((N, 6))
    px_I = traj_states[:, 0]
    py_I = traj_states[:, 1]
    vx_I = np.diff(px_I) / dt
    vx_I = np.append(vx_I, vx_I[-1])
    vy_I = np.diff(py_I) / dt
    vy_I = np.append(vy_I, vy_I[-1])
    theta = np.arctan2(vy_I, vx_I)
    omega = wrap_circular_value(np.diff(theta) / dt)
    omega = np.append(omega, omega[-1])
    vx_B = vx_I * np.cos(theta) + vy_I * np.sin(theta)
    vy_B = -vx_I * np.sin(theta) + vy_I * np.cos(theta)

    traj_states_new_frame[:, 0] = px_I
    traj_states_new_frame[:, 1] = py_I
    traj_states_new_frame[:, 2] = theta
    traj_states_new_frame[:, 3] = vx_B
    traj_states_new_frame[:, 4] = vy_B
    traj_states_new_frame[:, 5] = omega
    return traj_states_new_frame

# Add parser.
NAME_TRAJ = 'fig8_simple'

if NAME_TRAJ == 'fig8_simple':
    period = 10.0
    length = 2.0
    N = 100
    dt = period / N
    t = np.linspace(0, period, N)
    
    initial_state_I = np.array([0.0, 0.0, 0.0])
    traj_states = np.zeros((N, 6))
    for i in range(N):
        current_time = t[i]
        x, y, z, x_dot, y_dot, z_dot = compute_fig8_simple(period, length, current_time, initial_state_I)
        traj_states[i] = np.array([x, y, z, x_dot, y_dot, z_dot])

    traj_states_new_frame = transform_traj(traj_states)
    fig, ax = plot_states(traj_states_new_frame, title='Figure 8')
    
if NAME_TRAJ == 'circle_start_on_circle':
    angle = 0.0
    END_TIME = 12.0 # seconds
    dt = 0.01
    N = int(END_TIME / dt)
    v_desired = 2.0
    initial_state_I = np.array([0.0, 0.0, 0.0])
    traj_states = np.zeros((N, 6))
    for i in range(N):
        x, y, z, x_dot, y_dot, z_dot, angle = compute_circle_start_on_circle(angle, dt, v_desired, initial_state_I)
        traj_states[i] = np.array([x, y, z, x_dot, y_dot, z_dot])
        
    traj_states_new_frame = transform_traj(traj_states)
    fig, ax = plot_states(traj_states_new_frame, title='Circle')    
    folder_trajectories = 'solutions/trajectories/'
    
plt.savefig('solutions/trajectories/' + NAME_TRAJ + '.pdf')

        
