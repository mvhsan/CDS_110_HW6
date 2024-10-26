import math
import numpy as np
import matplotlib.pyplot as plt

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

def compute_circle(angle, dt, v_desired, initial_state_I):
    R = 3.0
    omega = v_desired / R
    angle = angle + dt * omega
    x = R * math.cos(angle) + initial_state_I[0]
    y = R * math.sin(angle) + initial_state_I[1]
    z = 0.0
    x_dot = -R * math.sin(angle) * omega
    y_dot = R * math.cos(angle) * omega
    z_dot = 0.0
    return x, y, z, x_dot, y_dot, z_dot, angle

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

NAME_TRAJ = 'fig8_simple'

if NAME_TRAJ == 'fig8_simple':
    period = 10.0
    length = 2.0
    N = 100
    dt = period / N
    t = np.linspace(0, period, N)
    
    x_list = []
    y_list = []
    for i in range(N):
        current_time = t[i]
        initial_state_I = np.array([0.0, 0.0, 0.0])
        x, y, z, x_dot, y_dot, z_dot = compute_fig8_simple(period, length, current_time, initial_state_I)
        x_list.append(x)
        y_list.append(y)
        
    plt.plot(x_list, y_list)
    plt.show()
    
if NAME_TRAJ == 'circle' or NAME_TRAJ == 'circle_start_on_circle':
    angle = 0.0
    END_TIME = 12.0 # seconds
    dt = 0.01
    N = int(END_TIME / dt)
    v_desired = 2.0
    initial_state_I = np.array([0.0, 0.0, 0.0])
    x_list = []
    y_list = []
    for i in range(N):
        x, y, z, x_dot, y_dot, z_dot, angle = compute_circle(angle, dt, v_desired, initial_state_I)
        x_list.append(x)
        y_list.append(y)
        
    plt.plot(x_list, y_list)
    plt.show()
        
