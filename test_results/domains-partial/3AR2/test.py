#!/home/wei/anaconda3/envs/colabfold152/bin/python

def check_valid_partial_solutions(phaser_log_path):
    with open(phaser_log_path, "r") as f:
        lines = f.readlines()

    inside_solution_block = False
    valid_partial_solutions = []
    highest_tfz = 0.0

    for line in lines:
        if "Solution" in line and "written to PDB file" in line:
            inside_solution_block = True
            continue

        if inside_solution_block and line.strip() == "":
            inside_solution_block = False

        if inside_solution_block and "SOLU 6DIM ENSE" in line:
            tfz_part = line.split("#")[-1]
            print(f"TFZ part: {tfz_part}")
            if "TFZ" in tfz_part:
                print ("TFZ found")
                tfz = float(tfz_part.split("==")[-1])
                valid_partial_solutions.append((line, tfz))
                if tfz > highest_tfz:
                    highest_tfz = tfz

    return valid_partial_solutions, highest_tfz

if __name__ == "__main__":
    phaser_log_path = "alternative_phaser_output/PHASER.log"
    valid_partial_solutions, highest_tfz = check_valid_partial_solutions(phaser_log_path)
    print(f"Valid partial solutions: {valid_partial_solutions}")
    print(f"Highest TFZ: {highest_tfz}")