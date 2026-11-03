class MoireStructure:
    def __init__(self, id, level, n_layers, material, parents, twist, lattice_vectors, residual, source_poscar, units='angstrom'):
        self.id = id
        self.level = level
        self.n_layers = n_layers
        self.material = material
        self.parents = parents
        self.twist = twist
        self.lattice_vectors = np.asarray(lattice_vectors, dtype=float)
        self.residual = residual
        self.source_poscar = source_poscar
        self.units = units 
        
        # --- alternative constructors -------------------------------------
        @classmethod
        def from_poscar(cls, file_path, material, units="angstrom"):
          """Build a level-0 (root) structure from a POSCAR."""
          ...                      # read poscar, return cls(...)

        @classmethod
        def from_yaml(cls, path):
          """Rebuild a structure from a saved YAML file."""
          ...                      # safe_load, return cls(...)

        # --- behavior that belongs to a structure -------------------------
        @property
        def angle_between_vectors(self):
          a1, a2 = self.lattice_vectors
          return angle_between(a1, a2)         # reuse, not redefine
