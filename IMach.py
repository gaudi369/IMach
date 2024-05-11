from typing import Match
#Final VISCOUS AND INVISCOUS CASE ATTEMPTING GRAPHING
#THERE IS AN ERROR. this viscous drag should affect the pressure at every point
import ipywidgets as widgets
from IPython.display import display, clear_output
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from sympy import symbols, sin, cos, tan, solveset, S, Eq
from sympy.solvers import solve
from scipy.integrate import odeint
import math


#Given


# Assume m1 and Pin are given
m1 = 3  # Upstream Mach number at the leftmost point
Pin = 101325  # Incoming pressure in Pascals
rhoin= 1.204 #kg/m^3
gamma=1.4


# Initialize the list with the normal shock pressure and Mach number
flow_props = [(m1,Pin,rhoin)]
angles=[]





def calculate_clockwise(points,flow_props):
     # Find indices of leftmost and rightmost points
    leftmost_point_index = min(range(len(points)), key=lambda i: points[i][0])
    rightmost_point_index = max(range(len(points)), key=lambda i: points[i][0])

    # Initialize the current index to start from the leftmost point
    current_index = leftmost_point_index

    #setting up the inital. Its ALWAYS GOING TO BE AN OBLIQUE SHOCK AT THE FRONT
    next_index = (current_index - 1) % len(points)
    horizontal_vector = np.array([1, 0])  # Horizontal vector to the right
    vec_to_next = np.array(points[next_index]) - np.array(points[leftmost_point_index])
    vec_to_next_norm = vec_to_next / np.linalg.norm(vec_to_next)
    dot_product = np.dot(horizontal_vector, vec_to_next_norm)
    initial_angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    initial_angle_degrees = np.degrees(initial_angle_rad)
    # Fetch the leftmost point using the leftmost_point_index
    leftmost_point = points[leftmost_point_index]
    #this virtual point guarantees a shockwave at the front
    virtual_point = (leftmost_point[0] - 1, leftmost_point[1])
    next_index = (current_index - 1) % len(points)
    next_next_index = (next_index - 1) % len(points)
    turning_angle, wave_type, flow_props = calculate_fp(virtual_point, points[next_index], points[current_index], flow_props)
    #print(f"Point {leftmost_point_index + 1}): {initial_angle_degrees:.2f} degrees- {wave_type}")


    # Loop to calculate turning angles until we reach the rightmost point
    while True:

         # Calculate the indices for the next and the point after the next, ensuring clockwise direction
        next_index = (current_index - 1) % len(points)

        # New condition: Stop when next index is about to be the rightmost point, thereby skipping the last segment
        if next_index == rightmost_point_index:
            break

        next_next_index = (next_index - 1) % len(points)

        # Perform the calculations for the current segment
        turning_angle, wave_type, flow_props = calculate_fp(points[next_next_index], points[next_index], points[current_index], flow_props)

        # Debug or logging
      #  print(f"Point {next_index + 1}: {turning_angle:.2f} degrees - {wave_type}")

        # Update the current index to move to the next point for the next iteration
        current_index = next_index

    segment_drags,segment_vdrags = calculate_drag(points, flow_props, leftmost_point_index, rightmost_point_index,angles)

    # Print the drag for each segment along with the indices of the connecting points
    topside_drag = 0
    viscous_drag = 0
    #for segment_info, drag in segment_drags:
        #start_point, end_point = segment_info
        # Adding 1 to the point indices for human-readable output (1-indexed instead of 0-indexed)
        #print(f"Segment {start_point+1}-{end_point+1}: Drag = {drag:.2f}")

        #topside_drag+=drag
    for index, (segment_info, drag) in enumerate(segment_drags):
        start_point, end_point = segment_info

        # Exclude the drag calculation for the last segment
        if index < len(segment_drags) - 1:  # Check if it's not the last segment
            topside_drag += drag

    for index, (segment_info, drag) in enumerate(segment_vdrags):
        start_point, end_point = segment_info

        # Exclude the drag calculation for the last segment
        if index < len(segment_vdrags) - 1:  # Check if it's not the last segment
            viscous_drag += drag
    return topside_drag,viscous_drag, segment_vdrags





def calculate_fp(a, b, c, flow_props, gamma=1.4,):
    # Calculate vectors from points, angle, and wave type as before
    vec_ab = np.array([b[0] - a[0], b[1] - a[1]])
    vec_bc = np.array([c[0] - b[0], c[1] - b[1]])

    vec_ab_norm = vec_ab / np.linalg.norm(vec_ab)
    vec_bc_norm = vec_bc / np.linalg.norm(vec_bc)

    dot_product = np.dot(vec_ab_norm, vec_bc_norm)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle = np.arccos(dot_product)
    angle_degrees = np.degrees(angle)

    cross_product = np.cross(vec_ab_norm, vec_bc_norm)
    wave_type = "expansion wave" if cross_product > 0 else "shock wave"

    # Get the last Mach number and pressure from the list
    last_mach, last_pressure, last_rho = flow_props[-1]

    if wave_type == "expansion wave":

        # Calculate the Prandtl-Meyer function for the last Mach number
        nu_last_mach = prandtl_meyer_function(last_mach, gamma, 0)
        # The turning angle needs to be added to nu_last_mach to find the new nu_mach
        nu_mach = nu_last_mach + np.radians(angle_degrees)

        # Solve for the new Mach number using the Prandtl-Meyer function
        new_mach = fsolve(prandtl_meyer_function, last_mach, args=(gamma, nu_mach))[0]

        # WRITE EXPANSION PRESSURE AND DENSITY FORMULA
        new_pressure= calculate_exp_pressure(last_mach,new_mach,last_pressure,gamma)
        new_rho= calculate_exp_rho(last_mach,new_mach,last_rho,gamma)

        # Update the list with the new Mach number and pressure
        flow_props.append((new_mach, new_pressure,new_rho))
        angles.append((angle_degrees))
        #print(f"current angle {angle_degrees}")
       # print(f"List of angles:{angles}")

    else:

        # Find indices of leftmost and rightmost points
        leftmost_point_index = min(range(len(points)), key=lambda i: points[i][0])
        rightmost_point_index = max(range(len(points)), key=lambda i: points[i][0])

        # Initialize the current index to start from the leftmost point
        current_index = leftmost_point_index

        #setting up the inital. Its ALWAYS GOING TO BE AN OBLIQUE SHOCK AT THE FRONT
        next_index = (current_index - 1) % len(points)
        horizontal_vector = np.array([1, 0])  # Horizontal vector to the right
        vec_to_next = np.array(points[next_index]) - np.array(points[leftmost_point_index])
        vec_to_next_norm = vec_to_next / np.linalg.norm(vec_to_next)
        dot_product = np.dot(horizontal_vector, vec_to_next_norm)
        initial_angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
        initial_angle_degrees = np.degrees(initial_angle_rad)

        if flow_props and (flow_props[0][0], flow_props[0][1]) == (last_mach, last_pressure):
            result = oblique_shock(initial_angle_degrees * np.pi / 180, last_mach, 280, last_pressure,last_rho, gamma)
            angles.append((initial_angle_degrees))
        else:
          result = oblique_shock(angle_degrees * np.pi / 180, last_mach, 280, last_pressure,last_rho, 0.1)
          angles.append((angle_degrees))


        #print(result[2])
        M2 = result[2]
        P2 = result[4]
        rho2 = result[5]
        #print(angle_degrees)
        #print(last_mach)
        flow_props.append((M2, P2, rho2))
        pass
    return angle_degrees, wave_type, flow_props







#CALCULATING PRESSURES AND MACH
#oblique shock

def temp_to_sos(T):
    # Speed of sound in dry air given temperature in K
    return 20.05 * T**0.5




def oblique_shock(theta, Ma, T, p, rho, gamma=1.4):

#cited from gusgordon on github

    x = np.tan(theta)
    for B in np.arange(1, 500) * np.pi/1000:
        r = 2 / np.tan(B) * (Ma**2 * np.sin(B)**2 - 1) / (Ma**2 * (gamma + np.cos(2 * B)) + 2)
        if r > x:
            break
    cot_a = np.tan(B) * ((gamma + 1) * Ma ** 2 / (2 * (Ma ** 2 * np.sin(B) ** 2 - 1)) - 1)
    a = np.arctan(1 / cot_a)

    Ma2 = 1 / np.sin(B - theta) * np.sqrt((1 + (gamma - 1)/2 * Ma**2 * np.sin(B)**2) / (gamma * Ma**2 * np.sin(B)**2 - (gamma - 1)/2))

    h = Ma ** 2 * np.sin(B) ** 2
    T2 = T * (2 * gamma * h - (gamma - 1)) * ((gamma - 1) * h + 2) / ((gamma + 1) ** 2 * h)
    p2 = p * (2 * gamma * h - (gamma - 1)) / (gamma + 1)
    rho2 = rho * ((gamma + 1) * h) / ((gamma - 1) * h + 2)

    v2 = Ma2 * temp_to_sos(T2)
    v_x = v2 * np.cos(a)
    v_y = v2 * np.sin(a)

    return B, a, Ma2, T2, p2, rho2, v_x, v_y


#expansion wave
# Define the Prandtl-Meyer function
def prandtl_meyer_function(M, gamma, nu_angle):
    return (np.sqrt((gamma + 1) / (gamma - 1)) * np.arctan(np.sqrt((gamma - 1) / (gamma + 1) * (M**2 - 1))) -
            np.arctan(np.sqrt(M**2 - 1))) - nu_angle

def calculate_exp_rho(M1,M2,rho1,gamma):
    rho2= ((1 + ((gamma - 1) / 2) * M1**2) / (1 + ((gamma - 1) / 2) * M2**2))**(1/(gamma-1))*rho1
    return rho2

def calculate_exp_pressure(M1,M2,P1,gamma):
    P2 = (((1 + ((gamma - 1) / 2) * M1**2) / (1 + ((gamma - 1) / 2) * M2**2))**(gamma / (gamma - 1)))*P1
    return P2



def calculate_angle(point_a, point_b):
    # Calculate angle with horizontal
    dy = point_b[1] - point_a[1]
    dx = point_b[0] - point_a[0]
    angle_radians = np.arctan2(dy, dx)
    angle_degrees = np.degrees(angle_radians)
    return abs(angle_degrees)  # Return the absolute value of the angle






#GEOMETRY
def calculate_area(points):
    # Using the Shoelace formula to calculate the area of a polygon
    #print(points)
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
   # print(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
    area =(1/(10**6))* (0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
    #print(f"The area is {area}")
    return area

def trip_calculator(available_volume):

    # Constant volume to be transported
    volume_to_transport = 1

    # Calculate the total number of trips required
    total_trips = (volume_to_transport/available_volume)*2
    # Check for any remaining volume that would require an additional trip
    if volume_to_transport % available_volume > 0:
        total_trips += 1

    return int(total_trips)







def calculate_drag(points, flow_props, leftmost_point_index, rightmost_point_index,angles):
    segment_drags = []
    segment_vdrags=[]
    num_points = len(points)

    # Adjust the starting index for pressures based on the traversal starting point (rightmost)
    pressure_start_index = 0

    angles_start_index = 0

    # Initialize the current index for traversal, starting from the leftmost point
    current_index = leftmost_point_index

    # Counter for flow_props index, considering the order starts from the rightmost point
    pressure_index = pressure_start_index

    #viscous drag coefficient
    cd = 0.01


    angles_index = angles_start_index
   # print(f"We are calculating for index : {pressure_index}")
    while True:
        # Calculate the index for the next point, ensuring clockwise direction
        next_index = (current_index - 1) % num_points

        # Calculate the Euclidean distance between the current point and the next point
        current_point = points[current_index]
        next_point = points[next_index]
        distance = np.linalg.norm(np.array(current_point) - np.array(next_point))

        # Fetch the pressure corresponding to the current segment
        # Fetch the pressure corresponding to the current segment
        _, pressure, _ = flow_props[pressure_index % len(flow_props)]

        _,_,rho =  flow_props[pressure_index % len(flow_props)]
        mach,_,_ =  flow_props[pressure_index % len(flow_props)]

        # Fetch the angle corresponding to the current segment

        # Calculate pressure "drag" for the segment
        a= np.sin( angles[pressure_index % len(flow_props)]*(np.pi/180))

        segment_drag = (distance/1000) * pressure * a


        #Calculate viscous drag for each segment
        u = mach * 343
        segment_vdrag = (0.5*rho*(u**2)*cd*(distance/1000))+segment_drag


        # Store the drag along with the current and next index
        segment_drags.append(((current_index, next_index), segment_drag))
        segment_vdrags.append(((current_index,next_index),segment_vdrag))
        # Update the current index to the next index for the next iteration
        current_index = next_index

        # Update the pressure index for the next segment
        pressure_index += 1


        # Break the loop if the next point is the rightmost point
        if current_index == rightmost_point_index:
            break

    return segment_drags, segment_vdrags

# i need a function to calculate the drag on the bottom
def calculate_bottom_distance(points):
    # Extract x-coordinates from the points
    x_coords = [point[0] for point in points]

    # Find the maximum and minimum x-coordinates (rightmost and leftmost points)
    max_x = max(x_coords)
    min_x = min(x_coords)

    # Calculate the distance between the rightmost and leftmost points
    distance = max_x - min_x

    return distance




def draw_shape(points):
    clear_output(wait=True)
    display_widgets()

    # Create a figure and a set of subplots

    fig, ax = plt.subplots(figsize=(7, 7))

    # Set the x and y axis limits to be from 0 to 1000 instead of 0 to 100
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)

    # Plot the shape with the updated axis limits
    xs, ys = zip(*points)
    xs += (xs[0],)
    ys += (ys[0],)
    ax.plot(xs, ys, color='grey')

    # Define the colors for each point
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'orange', 'brown', 'pink', 'gray']

    # Plot each point with its corresponding color and add a text label
    for i, (x, y) in enumerate(points):
        ax.plot(x, y, 'o', color=colors[i % len(colors)])
        ax.text(x, y, f' {i+1}', verticalalignment='bottom', horizontalalignment='right')

    # Enable grid
    plt.grid(True)

    # Set title of the plot
    plt.title(f"Polygon with {len(points)} Points")

    # Display the plot
    plt.show()


def update_points(button):

    global points, flow_props, angles
    #resets
    Cost= 0
    flow_props=[]
    flow_props = [(m1,Pin,rhoin)]
    angles= []

    new_points = []
    for widget in point_widgets:
        x, y = map(float, widget.value.split(','))
        new_points.append((x, y))
    points = new_points
    draw_shape(points)


    area = calculate_area(points)
    # Assume depth is 1 meter, so the area is equal to the volume
    #mm to meter conversion
    volume = area

    topdrag, vdrag, segment_vdrags = calculate_clockwise(points, flow_props)
    #adding the viscous drag at the bototm
    bottom_distance = calculate_bottom_distance(points)
    bottom_drag = (0.5 * 1.204* (1029**2)* 0.01 * bottom_distance )
    #print(f"the bottom drag is {bottom_drag}")
    vdrag+=bottom_drag
    print(f"Flow Properties (Mach, Pressure, Density): {flow_props}")
    #print(f"Angles: {angles}")
    print(f"Spatial Capacity of Train{volume}")
    print(f"Total Pressure Drag: {topdrag}")
    print(f"Total Viscous Drag: {vdrag}")

    number_of_trips = trip_calculator(volume)
    print(f"Number of trips needed: {number_of_trips}")

    #FINAL COST
    #doing a pseudo conversion so we can have a price that makes sense.
    InvCost = 20*number_of_trips*topdrag
    print(f"Inviscid Case Estimated Cost: {InvCost:,.2f}")
    VCost = (20*number_of_trips*(vdrag))
    print(f"Viscous Case Estimated Cost: {VCost:,.2f}")


    flow_props_except_last = flow_props[:-1]

    # Extract Mach numbers and pressures into separate lists, excluding the last entry
    mach_numbers_except_last = [fp[0] for fp in flow_props_except_last]
    pressures_except_last = [fp[1] for fp in flow_props_except_last]

    # The X-axis will simply be a range from 1 to the length of the modified flow_props list
    points_range_except_last = list(range(1, len(flow_props_except_last) + 1))

    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plot Mach numbers excluding the last point
    axs[0].plot(points_range_except_last, mach_numbers_except_last, marker='o', linestyle='-', color='b')
    axs[0].set_title('Mach Numbers')
    axs[0].set_xlabel('Point Number')
    axs[0].set_ylabel('Mach Number')

    # Plot Pressures excluding the last point
    axs[1].plot(points_range_except_last, pressures_except_last, marker='s', linestyle='-', color='r')
    axs[1].set_title('Pressures')
    axs[1].set_xlabel('Point Number')
    axs[1].set_ylabel('Pressure (Pa)')


    plt.tight_layout()
    plt.show()





from ipywidgets import Text

def create_point_widgets(num_points):
    global point_widgets
    point_widgets = []
    base_y = 200  # Y-coordinate for the flat bottom
    top_y = 500   # Y-coordinate for the top (you can adjust this)
    left_x = 100  # X-coordinate for the leftmost point
    right_x = 900  # X-coordinate for the rightmost point

    # Ensure there are at least 3 points
    if num_points < 3:
        print("Number of points must be 3 or more.")
        return

    # Add the rightmost bottom point (this will be point 1)
    point_widgets.append(Text(value=f'{right_x:.2f},{base_y:.2f}', description='Point 1:', disabled=False))

    # Generate the upper points from right to left
    for i in range(num_points - 2):
        # Generate `x` as a linear interpolation between `right_x` and `left_x`
        x = right_x - (right_x - left_x) * (i + 1) / (num_points - 1)

        # Assuming a straight line for the upper part, calculate `y`
        y = top_y

        # Add each upper point to the list
        point_widgets.append(Text(value=f'{x:.2f},{y:.2f}', description=f'Point {i+2}:', disabled=False))

    # Add the leftmost bottom point (this will be the last point)
    point_widgets.append(Text(value=f'{left_x:.2f},{base_y:.2f}', description=f'Point {num_points}:', disabled=False))

# Call the function with the desired number of points
# create_point_widgets(8) # For an octagon


def display_widgets():
    for widget in point_widgets:
        display(widget)

    update_button = widgets.Button(description="Update Points")
    update_button.on_click(update_points)
    display(update_button)

# Replace slider with text input
num_points_input = widgets.Text(value='3', description='# of Points:')

def on_confirm(button):
    try:
        num_points = int(num_points_input.value)
        if num_points < 3:
            print("Please enter a number of 3 or more.")
            return
        create_point_widgets(num_points)
        display_widgets()
    except ValueError:
        print("Please enter a valid integer.")

confirm_button = widgets.Button(description="Generate Shape")
confirm_button.on_click(on_confirm)





display(num_points_input, confirm_button)
