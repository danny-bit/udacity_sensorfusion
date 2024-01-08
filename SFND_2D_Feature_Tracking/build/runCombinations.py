import subprocess;
import os;
print(__file__)

# Change the working directory
#os.chdir('build')

print(__file__)

listDetectors = ["SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"]
listDescriptors = ["BRISK", "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT"]

#listDetectors = ["SHITOMASI"]# , "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"]
#listDescriptors = ["BRISK"]#, "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT"]

for detector in listDetectors:
    for descriptor in listDescriptors:
        if (descriptor == "AKAZE" and detector != "AKAZE"):
            # AKAZE descriptor only works with AKAZE detector
            continue
        if (descriptor == "ORB" and detector == "SIFT"):
            continue

        print("Detector: ", detector, " Descriptor: ", descriptor)
        subprocess.run(r"Debug\2D_feature_tracking.exe " + detector + " " + descriptor, 
                       shell=True)

#//subprocess.run(r"build\Debug\2D_feature_tracking.exe BRISK FREAK", shell=True)