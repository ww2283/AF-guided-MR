import gemmi

class DataManager:
    def __init__(self) -> None:
        pass

    def get_space_group(self, mtz_path):
        mtz = gemmi.read_mtz_file(mtz_path)
        space_group = mtz.spacegroup
        return space_group.hm

    def get_high_resolution(self, mtz_path):
        mtz = gemmi.read_mtz_file(mtz_path)
        return mtz.resolution_high()