### CHANGE OF BASIS PRIMER ###

# 0. Let v be a vector in space. v is not a list of numbers, it is a physical
# quanity independent of its representation using numbers.

# 1. There is always an implicit/standard coordintate system (Iota) whose basis are called
# the standard basis (STD).

# 2. Any matrix B which is invertible can be interpreted as change of basis matrix from coordinate system Beta (CSB) to the implicit coordinate system (STD). In this
# case, the columns of B are the representation of the basis vectors (which are physical) in the coordinate system Iota.

# If v_beta is its representation in the Beta coordinate system,
# then the representation of v in CSI is v_iota = B * v_beta

# two vectors v1 and v2 (both are physical) can be compared to each other
# only via their represntation in a common basis. If the representation
# of v1 and v2 in some common basis (coordinate system) is numerically same (the numerical vectors have the same numbers in their coordinates,
# then v1 and v2 are the same vector.

# If v1_alpha and v2_beta are representations of two vectors in some
# basis Alpha and Beta, we cannot compare them directly. We need to first
# bring them to the same common basis before comparision.

# It is sometimes easier to bring the vectors to the standard basis (implicit
# coordinate system) and then compare them

# If A and B are the change of basis matrices (Alpha/Beta coordinate system to the
# implicit coordinate system Iota (STD))

# Represent v1 and v2 in same coordinate system.
# v1_iota = A * v1_alpha
# v2_iota = B * v2_beta
# Then these representations (v1_iota and v2_iota) can be compared.

# NOTE: If D is the change of basis matrix from Delta system to Iota system, then
# inverse of D, i.e inv(D), is the change of basis matrix from Iota system to Delta system.
# Note that the coordinates of v in Alpha and Beta can be gotten from v_iota as
# v_alpha = inv(A) * v_iota
# v_beta = inv(B) * v_iota

######################
## In this problem, we have 4 coordinate systems

### I. Implicit coordinate system with standard basis (STD)

### II. Coordinate system of the bottom layer (Alpha coordinate system, A as change of
### basis matrix from Aplha coordinate system to standard/implicit coordinate system (Iota)

### III. Coordinate system of top layer (Gamma coordinate system, G as change of basis
### matrix from standard to Iota

### IV. Beta coordinate system (this is the rotated Gamma coordinate system by theta anticlockwise)
### B is the change of basis matrix from Beta coordinate system to Iota.
### Note: B = R * G, where R is the rotation by theta matrix.

### LATTICES
# The lattices in any coordinate system Mu, when represented in its own basis (Mu basis) will
# always have integer as numbers in the components.

# We call the lattices of Mu coordinate system by the variable MuLattice. When represented
# in its own coordinate system, it is MuLattice_mu. In the standard coordinate system,
# it is MuLattice_iota and hence it may or maynot have integral components.
# In general, some Delta and Eta coordinate system, DeltaLattice_eta is not going to be
# integral for most lattice points of Delta as most lattice points of Delta won't
# also be lattice points of Eta system (unless Eta and Delta are specially related).

### Checking if some vector v (whose representation is known in STD as v_iota) happens
### to be a lattice in some basis Delta, we need to check if its representation
### in Delta coordinate system, i.e v_delta, has integer components.
### So we first get v_delta ( assuming D is the change of basis matrix from Delta to Iota)
### v_delta = inv(D) v_iota
### Now we just check if v_delta has integer components, if it does, it is a lattice point in Delta system

### KNOWN SOLUTION
# theta = 22.62*np.pi/180 # degrees

import numpy as np
from tqdm import tqdm

import polars as pl

from icecream import ic

#from tool import angle_in_degrees


def scan(A, G, theta, tol, xlim, ylim):
    theta = np.deg2rad(theta)  # convert to radians
    # create rotation by theta matrix R
    R = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float64,
    )

    # the rotated top layer basis is Beta coordinate system
    # the change of basis matrix (from Beta to Iota) is B = R * G
    B = R @ G

    # lattices of bottom layer ( represented in Alpha coordinate system)
    # ofcourse the lattices in its own coordinate system have integer coordinates.
    AlphaLattice_alpha = []
    for x in range(*xlim):
        for y in range(*ylim):
            if x==0 or y==0:
                continue
            AlphaLattice_alpha.append([x, y])
    #AlphaLattice_alpha.remove([0, 0])  # remove origin as we don't care about this

    AlphaLattice_alpha = np.vstack(AlphaLattice_alpha).T

    # get the coordinates of the AlphaLattices in implicit system Iota
    AlphaLattice_iota = A @ AlphaLattice_alpha

    # Now, we want AlphaLattice_beta = inv(B) * AlphaLattice_iota
    # As discussed in the notes in the begining of the code,
    # most AlphaLattice_beta are not going to be integral.
    # We are looking for those special points in AlphaLattice which are indeed
    # also lattice points in BetaLattice and hence those special points
    # whose AlphaLattice_beta have integral coordinates

    sol = AlphaLattice_beta = np.linalg.solve(B, AlphaLattice_iota)
    solrounded = np.round(sol)
    solfrac = sol - solrounded
    solfrac_abs = np.abs(solfrac)

    matches = []

    for j in range(AlphaLattice_alpha.shape[1]):
        f1, f2 = solfrac[:, j]
        f1_abs, f2_abs = solfrac_abs[:, j]
        if f1_abs < tol:
            if f2_abs < tol:
                m1_true, m2_true = sol[:, j]
                m1, m2 = solrounded[:, j]
                n1, n2 = AlphaLattice_alpha[:, j]
                matchpoint_iota = A @ np.array([n1, n2]) # match points in iota basis
                delvec = np.linalg.norm(B @ np.array([m1, m2]) - matchpoint_iota)
                # delvec = np.linalg.norm(B @ np.array([m1, m2]) - A @ np.array([n1, n2i]) )
                match = {
                    "n1": n1,
                    "n2": n2,
                    "m1": int(m1),
                    "m2": int(m2),
                    "norm_m1m2": np.linalg.norm([m1, m2]),
                    "err_m1": f1,
                    "err_m2": f2,
                    "delvec": delvec,
                    "m1_true": m1_true,
                    "m2_true": m2_true,
                    "matchpoint": matchpoint_iota.tolist()
                }
                matches.append(match)
                # print(match)

    matches = sorted(matches, key=lambda x: np.linalg.norm([x["n1"], x["n2"]]))
    return matches[:25]


def run(a1, a2, g1, g2, degmin, degmax, degstep, dtol, nmin, nmax):
    
    A = np.array([a1,a2]).T
    G = np.array([g1,g2]).T

    xlim = ylim = (nmin,nmax)
    angles = np.arange(degmin,degmax,degstep)
    results = []
    for ang in tqdm(angles, colour="red", leave=False):
        if ang == 0:
            continue
        res = scan(A, G, theta=ang, xlim=xlim, ylim=ylim, tol=dtol)
        for r in res:
            r.update({"angle": ang})
            results.append(r)

    df = pl.DataFrame(results)
    return df


def filtermatches(a1,a2,df,goodangle=90.0):
    A = np.array([a1,a2]).T

    results = []

    for ang in df["angle"].unique():
        angdf = df.filter(df["angle"]==ang)
        N = angdf["n1","n2"].to_numpy()
        
        V = N @ A.T
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        dot =  V @ V.T
        cos = np.cos(np.deg2rad(goodangle))
        #if np.isclose(ang,27.8):
        #    print(angdf[0:10])
        
        
        ix,iy = np.where( np.isclose(dot, cos, rtol=1e-3) )
        
        indpairs = list(zip(ix,iy))

        for p in indpairs:
            a,b = p
            delta = np.rad2deg(np.acos(dot[a,b])) - goodangle 
            results.append([angdf.row(a,named=True),angdf.row(b,named=True),{"delgoodangle":delta}])

    return results


def run_and_filter(a1, a2, g1, g2, degmin, degmax, degstep, dtol, nmin, nmax, goodangles):
    
    df = run(a1, a2, g1, g2, degmin, degmax, degstep, dtol, nmin, nmax)
    
    allresults = []
    
    if df.height == 0:
        return allresults
    
    for good_angle in goodangles:
        
        results = filtermatches(a1,a2,df,good_angle)
        
        allresults.append({ 
                        "goodangle":good_angle,
                        "results": results,
                           })

    return allresults
