PLuM Pose Solver Files 

Lookup.py : Generates reward/lookup file using Model STL file
LookupVis.py : Visualizing generated Lookup File
pf_sa_gpu.py : GPU Accelerated implmentation of Improved Plum 

If possible try to convert code to C++ to acheive sub 50ms speed. sorry for not doing so. other wise looking at 1-3 second compute time per input. or drop reiteration to 0, possible solution under 0.5 seconds.
Careful manual fine tuning required while calculating MaxXYZ, Lookup_to_Model.
Calculate absolute scaling factor of the GT 3D model after centreing it (the one used for creating the lookup reward file). Necessary for good pose to reward corelation. 
Integration into pipeline requires just pulling the main funtion from the pf_sa_gpu.py file. Possibel JSON Config can be made for final deployment. 
