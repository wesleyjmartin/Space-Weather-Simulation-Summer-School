Final Project
=============

## Project A

Solve and plot the temperature using steady-state heat equation using the 
data and functions from `day_7` and `day_8`:

$$
\frac{\partial^2 T_i}{\partial z^2} = - \frac{Q_{EUV,i}}{4 \cdot 10^{-8}}
$$

The boundary conditions should be $\partial T / \partial z = 0$ at the top and
$T=200$ K at the bottom.

## Project B

Using RK4 to solve the below question for 24 hours:

$$
\frac{\partial T}{\partial t} = \frac{Q_{EUV}}{1500 \cdot \rho}
$$

Start with a 5 min time step. Treat the energy deposition for $O$, $O_2$, and $N_2$. In each timestep, you will need to:
1. Calculate the scale heights.
2. Calculate the hydrostatic densities using the latest temperature profile.
3. Calculate the solar zenith angle.
4. Calculate the Taus for $O$, $O_2$, and $N_2$ given the densities, scale heights, solar zenith angles, and cross sections. Limit the calculation to +/- 75 degrees, and set Tau equal to a large number outside of this range.
5. Calculate $Q_{EUV}$ for each species and the total $Q_{EUV}$.
6. Calculate $\frac{\partial T}{\partial t}$.
7. Update the temperature.
Make appropriate functions to make your code clean and beautiful. If you have time, make an animation and explain what is seen.

## Project C

Solve the chemistry equation using an implicit method

$$
\frac{\partial N_i}{\partial t} + V\frac{\partial N_i}{\partial z} = S_i - L_i 
$$

Where $N_i$ is the number density of species i ($O^+$, $O_2^+$, $N_2^+$, $NO^+$), V=-10m/s, $S_i$ are the sources (one of which will be EUV), and $L_i$ are the losses. You will need to include electrons as a species, but you don't have to advect it or solve for the chemistry for it, since $e- = \sum N_i$.

You can get the ionization rates from the euv_37.csv file.  The chemical reactions and the reaction rates are specified in the Chemistry lecture towards the end.  Note that this includes $He^+$ and $H^+$ and others, which you can ignore.

This project is hard.  Some recommendations:

1. Solve the equation using time spiltting - solve just the chemistry in one step, then the advection in another step.  This is significantly easier than solving the whole problem in one step.
2. Don't try to tackle all of the chemistry at once. Get something like the EUV ionization for $O^+$ working first (no loss term - only once source term and one species).  Then, take one loss term for $O^+$ and get that working. You will then have a single source term and loss term for $O^+$. Then try to do the same thing for $O_2^+$. Then, once those two species work, add more chemical equations.
3. Treat the neutrals as static - take the initial temperature and hydrostatic solutions and just keep those.  Don't implement chemistry or advection for the neutrals.
4. There are lines in the EUV file that include things like $O_2$ ionizing to $O^+$. You can ignore these lines. Just implement the three lines of $O_2$ going to $O_2^+$, $N_2$ going to $N_2^+$ and $O$ going to $O^+$.
