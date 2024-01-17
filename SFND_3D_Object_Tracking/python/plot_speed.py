import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os


resultFolder = './results/'

subfolders = [f.path for f in os.scandir(resultFolder) if f.is_dir()]

print(subfolders)

fig, ax = plt.subplots(figsize=(8,3.5));

df_main = pd.DataFrame();

bLidarPlotDone = False
for idxfile, folder in enumerate(subfolders):
    file = folder+'/log.csv'
    #df_read = pd.read_csv(r"build/log.csv",header=None)
    
    print(file)
    if (os.stat(file).st_size == 0):
        print("skip")
        continue;

    df_read = pd.read_csv(file,keep_default_na=False,na_values=['inf','-inf'])
    df_read.columns = ['imgIndex','ttcLidar','ttcCamera']
    df_read = df_read.replace(r'^s*$', float('NaN'), regex = True)  # Replace blanks by NaN
    df_read.dropna(inplace=True)
    df_read = df_read.astype({'imgIndex': 'int', 'ttcLidar': 'float32', 'ttcCamera': 'float32'})

    subfolder = os.path.split(folder)[1];
    vec = subfolder.split('_')

    df_read['detector'] = vec[0]
    df_read['descriptor'] = vec[1]
    if (not bLidarPlotDone):
        df_main = df_read
    else:
        df_main = pd.concat([df_main, df_read], ignore_index=True)

    if (not bLidarPlotDone):
        bLidarPlotDone = True
        plt.plot(df_read.imgIndex, df_read.ttcLidar, 'o-', label='lidar', markersize=7)

    plt.plot(df_read.imgIndex, df_read.ttcCamera, 'o-', label='camera_'+file[10:], linewidth=1.0, markersize=7.0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()

    plt.xlabel('image frame')
    plt.ylabel('time to collision [s]')
    #plt.title('time to collision (TTC) per frame @ 10 fps')

fig.tight_layout()

df_main['ttcCamera'] = df_main['ttcCamera'].clip(lower=0)
df_main[df_main['ttcCamera'] <= 0] = np.nan

df_main = df_main.query('detector not in ["ORB","HARRIS", "BRISK"]')
print(df_read)
plt.figure(figsize=(8,2))
#sns.color_palette("crest", as_cmap=True)
sns.lineplot(df_main,x='imgIndex',y='ttcCamera',hue='detector',style='descriptor',
             markers=True, dashes=False, linewidth=2, markersize=8, palette="Dark2")
plt.plot(df_read.imgIndex, df_read.ttcLidar, 'o-', color='b', 
         label='lidar', linewidth=3)
plt.legend(loc='upper right');
plt.xlim([0, 30])
plt.tight_layout()
plt.show()
