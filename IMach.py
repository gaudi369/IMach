import pygame
import sys
import numpy as np
from pygame.locals import *

# For aerodynamic calculations:
from scipy.optimize import fsolve
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

def sort_points(points):
    # Sort points based on x-coordinate primarily and y-coordinate secondarily to separate top and bottom profiles
    points_sorted = sorted(points, key=lambda x: (x[0], x[1]))
    top_profile = []
    bottom_profile = []
    
    # Assuming the first and last points after sorting are the leftmost and rightmost points respectively
    leftmost_point = points_sorted[0]
    rightmost_point = points_sorted[-1]
    
    # Divide points into top and bottom profiles
    for point in points_sorted:
        if point[1] <= (leftmost_point[1] + rightmost_point[1]) / 2:
            top_profile.append(point)
        else:
            bottom_profile.append(point)
    
    # Ensure top profile is sorted from left to right and bottom profile is sorted from right to left
    bottom_profile = sorted(bottom_profile, key=lambda x: -x[0])
    
    return top_profile, bottom_profile


def calculate_clockwise(top_profile, flow_props):
    # Start calculation from the leftmost point
    leftmost_point_index = 0  # By definition of sorting, first index is leftmost
    rightmost_point_index = len(top_profile) - 1  # Last index is rightmost

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
        next_index = (current_index + 1) % len(top_profile)

        # New condition: Stop when next index is about to be the rightmost point, thereby skipping the last segment
        if next_index == rightmost_point_index:
            break

        next_next_index = (next_index - 1) % len(top_profile)

        # Perform the calculations for the current segment
        turning_angle, wave_type, flow_props = calculate_fp(top_profile[current_index], top_profile[next_index], flow_props)
        
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


def calculate_fp(a, b, c, gamma):
    # Calculate vectors from points a to b and b to c
    vec_ab = np.array([b[0] - a[0], b[1] - a[1]])
    vec_bc = np.array([c[0] - b[0], c[1] - b[1]])

    # Calculate norms to check for zero-length vectors
    norm_ab = np.linalg.norm(vec_ab)
    norm_bc = np.linalg.norm(vec_bc)

    if norm_ab == 0 or norm_bc == 0:
        # Handling the case where no valid calculation can be made
        return 0, "undefined", flow_props

    # Normalize vectors
    vec_ab_norm = vec_ab / norm_ab
    vec_bc_norm = vec_bc / norm_bc

    # Calculate the angle between vectors ab and bc
    dot_product = np.dot(vec_ab_norm, vec_bc_norm)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    angle_degrees = np.degrees(angle)

    # Determine the wave type based on the direction of the cross product
    cross_product = np.cross(vec_ab_norm, vec_bc_norm)
    wave_type = "expansion wave" if cross_product > 0 else "shock wave"

    # Retrieve the last flow properties
    last_mach, last_pressure, last_rho = flow_props[-1]

    if wave_type == "expansion wave":
        # Handle expansion wave calculations
        nu_last_mach = prandtl_meyer_function(last_mach, gamma, 0)
        nu_mach = nu_last_mach + np.radians(angle_degrees)
        new_mach = fsolve(prandtl_meyer_function, last_mach, args=(gamma, nu_mach))[0]

        new_pressure = calculate_exp_pressure(last_mach, new_mach, last_pressure, gamma)
        new_rho = calculate_exp_rho(last_mach, new_mach, last_rho, gamma)

        # Update flow properties list
        flow_props.append((new_mach, new_pressure, new_rho))
        # Assuming 'angles' is globally accessible; otherwise, it should be passed and returned
        angles.append(angle_degrees)

    elif wave_type == "shock wave":
        # Handle shock wave calculations
        result = oblique_shock(np.radians(angle_degrees), last_mach, 280, last_pressure, last_rho, gamma)
        M2, P2, rho2 = result[2], result[4], result[5]

        # Update flow properties list
        flow_props.append((M2, P2, rho2))
        angles.append(angle_degrees)

    return angle_degrees, wave_type, flow_props






#CALCULATING PRESSURES AND MACH
#oblique shock

def temp_to_sos(T):
    # Speed of sound in dry air given temperature in K
    return 20.05 * T**0.5




def oblique_shock(theta, Ma, T, p, rho, gamma):

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


# Initialization
pygame.init()
WIDTH, HEIGHT = 800, 600
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Dynamic Area and Geometry Calculator')
WHITE = (255, 255, 255)
RED = (255, 0, 0)
POINT_RADIUS = 5
points = []  # This will store the points
font = pygame.font.Font(None, 36)

def draw():
    window.fill(WHITE)
    if len(points) > 1:
        pygame.draw.lines(window, RED, True, points, 1)
    for point in points:
        pygame.draw.circle(window, RED, point, POINT_RADIUS)
    pygame.display.flip()

def update_aerodynamics(points):
    if len(points) >= 3:  # Ensure at least three points are available
        flow_properties = [(m1, Pin, rhoin)]  # Reset or use current values
        top_drag, viscous_drag, segment_drags = calculate_clockwise(points, flow_props)
        # Update display or log results as needed
    else:
        print("Not enough points to perform aerodynamics calculations.")


# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == MOUSEBUTTONDOWN:
            points.append(event.pos)
            update_aerodynamics(points)  # Recalculate whenever points are added
        elif event.type == MOUSEMOTION:
            if event.buttons[0]:  # If mouse button is held down
                points[-1] = event.pos
                if len(points) > 3:
                    update_aerodynamics(points)  # Update in real-time as the point is moved

    draw()

pygame.quit()
sys.exit()


