#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import re
from collections import defaultdict

def parse_kiva_map(map_file):
    with open(map_file, 'r') as f:
        lines = f.readlines()
    dimensions = lines[0].strip().split(',')
    rows, cols = int(dimensions[0]), int(dimensions[1])
    num_endpoints = int(lines[1])
    num_home_stations = int(lines[2])
    simulation_time = int(lines[3])
    map_layout = []
    for line in lines[4:]:
        line = line.strip()
        if line:
            map_layout.append(list(line))
    return {
        'rows': rows,
        'cols': cols,
        'num_endpoints': num_endpoints,
        'num_home_stations': num_home_stations,
        'simulation_time': simulation_time,
        'layout': map_layout
    }

def parse_paths_file(paths_file):
    with open(paths_file, 'r') as f:
        lines = f.readlines()
    num_agents = int(lines[0])
    agent_paths = []
    for line in lines[1:]:
        path_str = line.strip()
        states = []
        for state_str in path_str.split(';'):
            if state_str.strip():
                parts = state_str.split(',')
                if len(parts) >= 3:
                    location = int(parts[0])
                    timestep = int(parts[2])
                    states.append((location, timestep))
        agent_paths.append(states)
    return agent_paths

def parse_tasks_file(tasks_file):
    with open(tasks_file, 'r') as f:
        lines = f.readlines()
    num_agents = int(lines[0])
    agent_tasks = []
    for line in lines[1:]:
        task_str = line.strip()
        tasks = []
        for task in task_str.split(';'):
            if task.strip():
                parts = task.split(',')
                if len(parts) >= 2:
                    location = int(parts[0])
                    timestep = int(parts[1])
                    tasks.append((location, timestep))
        agent_tasks.append(tasks)
    return agent_tasks

def location_to_coords(location, map_data):
    cols = map_data['cols']
    row = location // cols
    col = location % cols
    return row, col

def get_agent_goal_and_state(agent_idx, agent_tasks, agent_paths, frame):
    # Find the current goal for the agent at this frame
    # and whether the agent is carrying (after pickup, before drop-off)
    tasks = agent_tasks[agent_idx]
    path = agent_paths[agent_idx]
    # Find the last task that was finished before or at this frame
    last_finished_idx = -1
    for i, (loc, t) in enumerate(tasks):
        if t != -1 and t <= frame:
            last_finished_idx = i
    # The next task is the current goal
    goal_idx = last_finished_idx + 1
    if goal_idx < len(tasks):
        goal_loc = tasks[goal_idx][0]
    else:
        goal_loc = None
    # Determine if carrying: if agent is between pickup and drop-off
    # We'll say carrying if agent has reached the goal at least once and is now moving to a new goal
    carrying = False
    if last_finished_idx >= 0:
        carrying = True
    return goal_loc, carrying

def create_animation(map_data, agent_paths, agent_tasks, output_file=None):
    rows, cols = map_data['rows'], map_data['cols']
    layout = map_data['layout']
    max_timestep = 0
    for path in agent_paths:
        for _, timestep in path:
            max_timestep = max(max_timestep, timestep)
    color_map = {
        '.': 'white',
        'e': 'lightblue',
        'w': 'orange',
        '@': 'darkgray',
    }
    fig, ax = plt.subplots(figsize=(12, 8))
    for i in range(rows):
        for j in range(cols):
            if i < len(layout) and j < len(layout[i]):
                cell = layout[i][j]
                color = color_map.get(cell, 'white')
                rect = plt.Rectangle((j, rows-1-i), 1, 1, facecolor=color, edgecolor='black', linewidth=0.5)
                ax.add_patch(rect)
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.set_title('RHCR Agent Animation', fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    agent_colors = ['red', 'blue', 'green', 'purple', 'magenta', 'brown', 'pink', 'gray']
    agent_scatters = []
    goal_markers = []
    for i in range(len(agent_paths)):
        color = agent_colors[i % len(agent_colors)]
        # Start with free state: normal size, thin edge
        scatter = ax.scatter([], [], c=color, s=100, alpha=0.8, label=f'Agent {i+1}', edgecolors='black', linewidth=1, marker='o', zorder=3)
        agent_scatters.append(scatter)
        # Goal marker (star)
        goal_marker, = ax.plot([], [], marker='*', color=color, markersize=20, linestyle='None', alpha=1.0, zorder=2)
        goal_markers.append(goal_marker)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    def animate(frame):
        print(f"Frame {frame}")
        for i, path in enumerate(agent_paths):
            x_pos, y_pos = [], []
            for location, timestep in path:
                if timestep <= frame:
                    row, col = location_to_coords(location, map_data)
                    x_pos.append(col + 0.5)
                    y_pos.append(rows - 1 - row + 0.5)
            if x_pos and y_pos:
                agent_x, agent_y = x_pos[-1], y_pos[-1]
                goal_loc, carrying = get_agent_goal_and_state(i, agent_tasks, agent_paths, frame)
                color = agent_colors[i % len(agent_colors)]
                # Carrying: thicker edge, larger marker
                if carrying:
                    agent_scatters[i].set_offsets([[agent_x, agent_y]])
                    agent_scatters[i].set_color(color)
                    agent_scatters[i].set_edgecolor('black')
                    agent_scatters[i].set_linewidths(3)
                    agent_scatters[i].set_sizes([200])
                else:
                    agent_scatters[i].set_offsets([[agent_x, agent_y]])
                    agent_scatters[i].set_color(color)
                    agent_scatters[i].set_edgecolor('black')
                    agent_scatters[i].set_linewidths(1)
                    agent_scatters[i].set_sizes([100])
                if goal_loc is not None:
                    goal_row, goal_col = location_to_coords(goal_loc, map_data)
                    goal_x, goal_y = goal_col + 0.5, rows - 1 - goal_row + 0.5
                    goal_markers[i].set_data([goal_x], [goal_y])
                else:
                    goal_markers[i].set_data([], [])
            else:
                agent_scatters[i].set_offsets([[]])
                goal_markers[i].set_data([], [])
        ax.set_title(f'RHCR Agent Animation - Timestep {frame}', fontsize=14, fontweight='bold')
        return agent_scatters + goal_markers
    anim = animation.FuncAnimation(fig, animate, frames=max_timestep+1, interval=200, blit=False, repeat=True)
    if output_file:
        anim.save(output_file, writer='pillow', fps=5)
        print(f"Animation saved to {output_file}")
    plt.tight_layout()
    plt.show()
    return anim

def show_single_frame(map_data, agent_paths, agent_tasks, frame):
    # Copy the setup from create_animation up to the definition of animate
    rows, cols = map_data['rows'], map_data['cols']
    layout = map_data['layout']
    color_map = {
        '.': 'white',
        'e': 'lightblue',
        'w': 'orange',
        '@': 'darkgray',
    }
    fig, ax = plt.subplots(figsize=(12, 8))
    for i in range(rows):
        for j in range(cols):
            if i < len(layout) and j < len(layout[i]):
                cell = layout[i][j]
                color = color_map.get(cell, 'white')
                rect = plt.Rectangle((j, rows-1-i), 1, 1, facecolor=color, edgecolor='black', linewidth=0.5)
                ax.add_patch(rect)
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.set_title(f'RHCR Agent Animation - Timestep {frame}', fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    agent_colors = ['red', 'blue', 'green', 'purple', 'magenta', 'brown', 'pink', 'gray']
    for i, path in enumerate(agent_paths):
        x_pos, y_pos = [], []
        for location, timestep in path:
            if timestep <= frame:
                row, col = location_to_coords(location, map_data)
                x_pos.append(col + 0.5)
                y_pos.append(rows - 1 - row + 0.5)
        if x_pos and y_pos:
            agent_x, agent_y = x_pos[-1], y_pos[-1]
            color = agent_colors[i % len(agent_colors)]
            ax.scatter([agent_x], [agent_y], c=color, s=100, alpha=0.8, edgecolors='black', linewidth=1, marker='o', zorder=3)
            goal_loc, _ = get_agent_goal_and_state(i, agent_tasks, agent_paths, frame)
            if goal_loc is not None:
                goal_row, goal_col = location_to_coords(goal_loc, map_data)
                goal_x, goal_y = goal_col + 0.5, rows - 1 - goal_row + 0.5
                ax.plot([goal_x], [goal_y], marker='*', color=color, markersize=20, linestyle='None', alpha=1.0, zorder=2)
    plt.show()

if __name__ == "__main__":
    # File paths - UPDATE THESE TO USE YOUR CUSTOM MAP
    map_file = "./maps/human/kiva_small_2x2_w_mode.map"  # Change this to your custom map
    paths_file = "./exp/run_agents50_endpoints240/paths.txt"     # Change this to your output directory
    tasks_file = "./exp/run_agents50_endpoints240/tasks.txt"     # Change this to your output directory
    
    try:
        map_data = parse_kiva_map(map_file)
        agent_paths = parse_paths_file(paths_file)
        agent_tasks = parse_tasks_file(tasks_file)
        print(f"Map size: {map_data['rows']}×{map_data['cols']}")
        print(f"Number of agents: {len(agent_paths)}")
        print(f"Max timestep: {max(timestep for path in agent_paths for _, timestep in path)}")
        # for full animation
        create_animation(map_data, agent_paths, agent_tasks, "custom_agent_animation.gif")
        #frame_to_show = 35  # or any frame you want
        #show_single_frame(map_data, agent_paths, agent_tasks, frame_to_show)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Make sure you're running from the RHCR directory and have run RHCR with output enabled")
    except Exception as e:
        print(f"Error: {e}") 