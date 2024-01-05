import os
import re
import pandas as pd
import matplotlib
import numpy as np
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess;
import os;
print(__file__)

# Change the working directory
#os.chdir('build')
folder_path = 'results/'

files = os.listdir(folder_path)

df = pd.DataFrame();
for file in files:
    if os.path.isfile(os.path.join(folder_path, file)):
        if (not '.csv' in file):
            continue

        pattern = r"log_(\w+)_(\w+)\.csv"
        match = re.match(pattern, file)

        print (file)
        if match:
            # string strRef = detectorType + "_" + descriptorType;
            detector = match.group(1)
            descriptor = match.group(2)
        else:
            print("Filename does not match the pattern.")

        try:
            df_file = pd.read_csv(folder_path + file) #%, dtype=str)
        except:
            print("Error reading file: ", file)
            continue;


        df_file['detector'] = detector;
        df_file['descriptor'] = descriptor;

        def parse_size(row):
            kptSizeStr = row['sizeKeypoints']
            kptSizeStr = kptSizeStr[1:-2]
            if (kptSizeStr == ''):
                return []
            item_list = [float(item) for item in kptSizeStr.split(";")]
            return item_list

        df_file['size'] = df_file.apply(lambda row: parse_size(row), axis=1)
        df_file['sizeMedian'] = df_file.apply(lambda row: np.median(row['size']), axis=1)


        df = pd.concat([df, df_file], ignore_index=True)

df['tTotal'] = df['tDetection'] + df['tDescription']

#vis: time detection and description 
fig, ax = plt.subplots(2,1, figsize=(7,2.7))
df_mean = df.groupby(['detector']).mean()
sns.barplot(df_mean, x='detector',y='tDetection', 
            ax=ax[0], 
            order=df_mean.sort_values('tDetection', ascending=False).index)
ax[0].set_ylabel('time [ms]')
ax[0].set_title('Time for computing detector', fontsize=10)

df_mean = df.groupby(['descriptor']).mean()
sns.barplot(df_mean, x='descriptor',y='tDescription',
            ax=ax[1], 
            order=df_mean.sort_values('tDescription', ascending=False).index)
ax[1].set_ylabel('time [ms]')
ax[1].set_title('Time for computing descriptor', fontsize=10)


fig.tight_layout()
plt.savefig('timeDetectorDescriptor.png')

#vis: combinations
df_pivot = df.pivot_table(index='detector', columns='descriptor', values='tTotal',
                          aggfunc='median')

fig = plt.figure()
sns.heatmap(df_pivot, annot=True, fmt='.2f', cbar_kws={'label': 'time [ms]'}, cmap='Greens')
plt.title('Comparison of total time (detection + description) \n for descriptor/detector combinations')
fig.tight_layout()
plt.savefig('timeTotal.png')

#vis: combinations
df_pivot = df.pivot_table(index='detector', columns='descriptor', values='nMatches',
                          aggfunc='median')
 

fig = plt.figure()
sns.heatmap(df_pivot, annot=True, fmt='.2f', cbar_kws={'label': 'number matches [-]'}, cmap='Greens')
plt.title('Median of match count for descriptor/detector combinations')
fig.tight_layout()

plt.savefig('matchCount.png')

plt.figure()
df_firstimg = df.query('img.str.contains("0000000001")')

xlabels = df_firstimg['detector'].unique()
N = len(xlabels)
w = N/2;
w = int(np.ceil(w))
print(w)

plt.xticks(fontsize=8)
fig, ax = plt.subplots(w,2, figsize=(7,6))
for n,xlabel in enumerate(xlabels):
        ylabel='BRISK'
        df_tmp = df_firstimg.query('detector == @xlabel and descriptor == @ylabel')
        if (len(df_tmp) == 0):
            continue
        ax_idx = np.unravel_index(n, ax.shape)
        ax[ax_idx].hist(df_tmp['size'].values[0], bins=10)
        ax[ax_idx].set_title(xlabel + " nKeypoints: " + str(df_tmp['nKeypoints'].values[0]), fontsize=10)
        ax[ax_idx].set_xlabel('keypoint size')
        ax[ax_idx].set_ylabel('kpt count')
        #ax[ax_idx].set_xticklabels(ax[ax_idx].get_xticklabels(),fontsize=8)
        #ax[ax_idx].tick_params(axis='both', which='minor', size=8)


fig.tight_layout()
plt.savefig('keypointSizeDist.png')

fig = plt.figure(figsize=(8,3.5))
sns.lineplot(df,x='img', y='nKeypoints', hue='detector', style='descriptor')
plt.xlabel('image')
plt.xticks(np.arange(1,11),np.arange(1,11))
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.savefig('keyPointsPerImg.png')

df_pivot = df.pivot_table(index='detector', columns='descriptor', values='sizeMedian',
                          aggfunc='mean')

fig = plt.figure()
sns.heatmap(df_pivot, annot=True, fmt='.2f', cbar_kws={'label': 'number matches [-]'}, cmap='Greens')
fig.tight_layout()

plt.show()
quit()

from matplotlib.collections import PatchCollection
fig = plt.figure()
N = df_pivot.shape[1]
M = df_pivot.shape[0]
ylabels = df_pivot.columns.values
xlabels = df_pivot.index.values

x, y = np.meshgrid(np.arange(M), np.arange(N))
#s = df_pivot.values*100
s = np.random.randint(0, 180, size=(N,M))
s = df_pivot.values 
c= np.random.rand(N, M)-0.5
fig, ax = plt.subplots()

R = s/np.nanmax(s)/2
circles = [plt.Circle((j,i), radius=r) for r, j, i in zip(R.flat, x.flat, y.flat)]
col = PatchCollection(circles, array=c.flatten(), cmap="RdYlGn")
ax.add_collection(col)

ax.set(xticks=np.arange(M), yticks=np.arange(N),
       xticklabels=xlabels, yticklabels=ylabels)
ax.set_xticks(np.arange(M+1)-0.5, minor=True)
ax.set_yticks(np.arange(N+1)-0.5, minor=True)
ax.grid(which='minor')

fig.colorbar(col)
plt.show()

plt.show()
quit()


print(__file__)

listDetectors = ["SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"]
listDescriptors = ["BRISK", "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT"]

#listDetectors = ["SHITOMASI"]# , "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"]
#listDescriptors = ["BRISK"]#, "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT"]

for detector in listDetectors:
    for descriptor in listDescriptors:
        if (detector == "AKAZE" and descriptor != "AKAZE"):
            continue
        if (detector == "ORB" and descriptor != "SIFT"):
            continue

        print("Detector: ", detector, " Descriptor: ", descriptor)
        subprocess.run(r"Debug\2D_feature_tracking.exe " + detector + " " + descriptor, 
                       shell=True)

#//subprocess.run(r"build\Debug\2D_feature_tracking.exe BRISK FREAK", shell=True)