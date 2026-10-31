1. in MuJoCo every contact is made up of dampners and springs, springs decide how much of that contact results in bounce in the normal direction while dampners decide how much of that contact's energy dissipates into heat
2. Our ball physics works in reverse, measure bounciness (restitution) and amount of contact and sim calculates optimal spring and dampner pairs
3. contact_solref is where we set stiffness and dampening, these solve for contact time (how long a collision lasts), and finally mujoco decides which param has priority for contact physics
4. This means each potential contact pairing needs its own bounciness and contact duration values
5. 2 Types of friction exist, rolling friction and sliding friction