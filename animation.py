class Animation(object):
    def __init__(
        self,
        rotations_6d,
        rotations_quat,
        positions,
        offsets,
        parents,
        names
    ):
        self.rotations_6d = rotations_6d
        self.rotations_quat = rotations_quat
        self.positions = positions
        self.offsets = offsets
        self.parents = parents
        self.names = names
#Animation