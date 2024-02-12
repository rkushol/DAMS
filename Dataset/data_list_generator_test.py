import os
import glob
import random

if __name__ == "__main__":
    test_file = open("test_ADNI1_GE_MNI.txt", "w")

    test_list = []
    data_dir = "."
    
    for entry in os.listdir("."):
        #entry_full_path = os.path.join(data_dir, entry)
        if os.path.isdir(entry):
            nii_files = glob.glob(os.path.join(entry, "**/*.nii"), recursive=True)
            category = 0
            nii_filtered = []
            for nii_file in nii_files:
                if "GE" in nii_file:
                    if "94to125" not in nii_file:
                        nii_filtered.append(nii_file)
                                        
                    
            nii_files = nii_filtered

            for nii_file in nii_files:
                category = 0 if "_CN_" in nii_file else (1 if "_AD_" in nii_file else (2 if "_MCI_" in nii_file else -1))
                if "GE" in nii_file:
                    test_list.append((nii_file, category))


    random.shuffle(test_list)

  
    for data in test_list:
        test_file.write("{} {}\n".format(data[0], data[1]))
        
   
    test_file.close()
