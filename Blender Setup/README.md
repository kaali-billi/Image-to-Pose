Quick Data Generation Belnder Setup :


1. addon_ground_truth_generation.py : Blender Addon based on VisionBlender (https://github.com/Cartucho/vision_blender)
                                      Changes made to work with quaternions, and Open3D Coordinate system, using Pre-processing funtions from UTILS.py

2. UTILS.py : Utility functions to process extracted blender data into datasets for depth estimation, point cloud completion and pose evaluation Metrics

3. DATA_GEN.py : Uses given directories to extracted data to give back Images,Depth,Segmentation Mask, Pose and Translation 

All files need improvement. For Incomplete and complete Point Cloud datasets for PCC, use function create_test_train(), after updating pcd_block() with camera intrinsic properties. Also Install HDRI Sun Aligner (https://github.com/akej74/hdri-sun-aligner) for realistic lighting that aligns itself with provided HDRI's

