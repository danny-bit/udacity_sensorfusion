import shutil
import subprocess
import os

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
        subprocess.run(r"..\build\Debug\3D_object_tracking.exe " + detector + " " + descriptor, 
                       shell=True)

        ident_str = detector + "_" + descriptor;

        source_folder = './'
        destination_folder = './results/' + ident_str + '/'

        # Check if the destination folder exists
        if not os.path.exists(destination_folder):
            os.mkdir(destination_folder)

        # Get a list of all files in the source folder
        files = os.listdir(source_folder)

        # Iterate over the files and copy the JPEG files to the destination folder
        for file in files:
            if file.endswith('.jpeg') or file.endswith('.jpg') or file.endswith('.png') or file.endswith('.csv'):
                source_file = os.path.join(source_folder, file)
                destination_file = os.path.join(destination_folder, file)
                shutil.move(source_file, destination_file)

#//subprocess.run(r"build\Debug\2D_feature_tracking.exe BRISK FREAK", shell=True)