Monocular Metric Depth Estimation Using DC-Depth and Adaptive Bins


For Dataset Creation : 
1. Create Folder Data ---- images
                       --- depths
                       --- masks    : using Data_gen.py with extracted Image and exr files.

2. Create Config file and add to \configs : Use template provided within

3. Make appropriate changes to DataLoader_DC.py file   # hopefully not needed

4. Run TrainDCT.py : check if the file is in fine tuning mode or pre-training mode  #needs to refined to be directly done by the config file

5. Training logs should show in folder called experiments

6. Weights saved in folder "Weights", for inference or evaluation use model.evaluate_depth(), inferencing may require new code for overall integration with PCC and PLUM
                   
