'''
MAX_ATOM_BONDS: maximum number of atom neighbors accepted by the model and data processing algorithm (will fail/crash if set too low).
Typically a value of 4 is sufficient for organic molecules, but some compounds (e.g. containing Pd) can have a higher connectivity.

'''


MAX_ATOM_BONDS = 5
ATOM_DEGREES = range(0, MAX_ATOM_BONDS+1)