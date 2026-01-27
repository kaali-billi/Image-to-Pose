# **Image to Pose : A Modular Approach to Spacecraft Pose Estimation**

# 



## **Folders :**

#### 

#### **MMD** - Metric Depth Estimation Algorithm

#### **PCC** - Point Cloud Completion (AdaPoinTr)

#### **PLuM** - Pose Solver for Completed Point Clouds

#### Blender - Blender Files provided for generating Sim Data, and pre-processing into datasets



## 

## **Pipeline** : Image -> MMD -> PCC -> PLuM -> Pose





##### **Training the Pipeline :**



1\. MMD needs to be pre-trained on simulated data, and then fine-tuned on real data. Suggested split between Sim data/Real Data : 10,000/800



2\. PCC can be purely trained on Sim Data (Same dataset or more as MMD training data)



3\. PLuM only needs the Reward file to be generated once given a CAD Model ***(No Training Required : Purely Numerical Approach)***



4\. For Sim data generation, upload CAD file with texture and material properties into Blender File, and select Segmentation indexes, object tracking and other properties that need to extracted



5\. Pre-processing files provided that converts blender data into a combined set for ***MMD*** and ***PCC Training***


