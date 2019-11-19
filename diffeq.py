from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#define differential equation
def diffeq(t, state, inertia):
    #function must always take time as a parameter, followed by state
    #all parameters afterwards are constants

    #unnecessary in this case, but this is how you might unpack your state if it was larger
    ang_rate = state[0:3]

    #do some math
    derivatives = - np.cross(ang_rate, inertia@ang_rate)

    #return the state derivatives
    return derivatives


#made up inertia matrix
inertia = np.array([[1,   .02, .5],
                    [.02, 1,   .1],
                    [.5,  .1,   1]])

#made up angular velocity
state = np.array([1, 2, 3])

#create ODE solver
solver = ode(diffeq) #pass the function to integrate
solver.set_integrator('lsoda') #select your solving algorythm. Dont overthink it
                               #I use lsoda or dopri5
solver.set_initial_value(state, 0) #set initial state and time, t0 = 0 in this case
solver.set_f_params(inertia) #define constants that will be passed to your function

#create python lists to hold outputs
newstate = [] 
time = [] 
#you can append to python lists, which means
#you dont have to know the size before hand
#if you do know the final size you could use
#newstate = zeros((m, n))
#time = zeros(m)
#this would be much faster than appending, but
#unless youre making HUGE lists you probably wont
#notice

#define some variables
tspan = 100
timestep = .1

#use a while loop to integrate your function.
#here you can put any stopping criteria you want
#in this case its when the simulation has simulated
#100 seconds.
while solver.successful() and solver.t < tspan:

    #save the state (solver.y) and time (solver.t)
    newstate.append(solver.y)
    time.append(solver.t)

    #integrate forwards 1 timestep
    solver.integrate(solver.t + timestep)
    #this doesnt return anything, it just updates the
    #solvers internal variables
    #This is why you have to access solver.y to get the state.
    #notice that unlike matlab, it does not return an array of all 
    #states calculated, it only gives you the final state after your
    #integration step.


#vstack creates a numpy array from a list of arrays by appending them
#vertically.
#hstack does the same but horizontally
newstate = np.vstack(newstate)
time = np.hstack(time)


#use matplotlib to create a figure and a set of axes to plot on
fig = plt.figure()
ax = fig.gca(projection='3d') #create a 3D plot
ax.plot(newstate[:,0], newstate[:,1], newstate[:,2])
plt.show() #Depending on your ide you may not need this line.
           #Generally it is required for your plots to pop up on
           #your screen