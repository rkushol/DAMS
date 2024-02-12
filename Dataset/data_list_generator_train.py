import os
import glob
import random

if __name__ == "__main__":
    train_file = open("train_ADNI1_GE_94to125.txt", "w")

    train_list = []
    data_dir = "."
    
    for entry in os.listdir("."):
        #entry_full_path = os.path.join(data_dir, entry)
        if os.path.isdir(entry):
            nii_files = glob.glob(os.path.join(entry, "**/*.nii"), recursive=True)
            category = 0
            nii_filtered = []
            for nii_file in nii_files:
                if "GE" in nii_file:
                    if "MCI_" not in nii_file:
                        nii_filtered.append(nii_file)
                                        
                    
            nii_files = nii_filtered

            for nii_file in nii_files:
                category = 0 if "_CN_" in nii_file else (1 if "_AD_" in nii_file else (2 if "_MCI_" in nii_file else -1))
                if "94to125" in nii_file:
                    train_list.append((nii_file, category))

    random.shuffle(train_list)

    for data in train_list:
        train_file.write("{} {}\n".format(data[0], data[1]))
        
    train_file.close()


