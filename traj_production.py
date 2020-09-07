import os
from ovito.pipeline import *
from ovito.modifiers import *
from ovito.io import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pymatgen.optimization.neighbors import find_points_in_spheres

# general function for pandas dataframe cross-correlation with lag
def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation.
    Shifted data filled with NaNs

    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else:
        return datax.corr(datay.shift(lag))

class trajstats():
    """
    Takes LAMMPS NVT outputs, extracts per atom trajectories, and provides several functions to compare them
    Input parameters: filename and atomid (particle type label)
    """

    def __init__(self, filename, atomid, vacid, r = 3.0):
        self.filename = filename
        self.atomid = atomid
        self.vacid = vacid
        self.pipeline = import_file(self.filename)
        data = self.pipeline.compute(0)
        # parameters for pymatgen's find_points_in_spheres
        self.r = r
        self.cell = np.array(data.cell).copy(order = 'C').astype('double')
        # parse this file into a numpy array and pandas dataframe for further study
        # put types and xyz positions into a dictionary
        self.trajs = {}
        self.atomtrajs = {}
        self.vactrajs  = {}

        for frame_index in range(0, self.pipeline.source.num_frames):
            data = self.pipeline.compute(frame_index)
            pos = np.array(data.particles['Position'])
            types = np.array(data.particles['Particle Type'])
            # must be 2D for np.append
            types = np.reshape(types, (len(types), 1))
            self.trajs[frame_index] = np.append(types, pos, axis = 1)
            self.atomtrajs[frame_index] = self.trajs[frame_index][np.ma.where(self.trajs[frame_index][:,0] == self.atomid)]
            self.vactrajs[frame_index] = self.trajs[frame_index][np.ma.where(self.trajs[frame_index][:,0] == self.vacid)]

        # array with atom coordinates vs time (timesteps,natoms,)
        self.atomsvstime = np.array([self.atomtrajs[frame][:,1:] for frame in self.atomtrajs.keys()], dtype = float)
        self.natoms = len(self.atomsvstime[0,:,0])

        # tracking the vacancies frame to frame is a little more complicated and requires finding nearest neighbors between frames
        self.vactrackedtrajs = {}
        self.vactrackedtrajs[0] = self.vactrajs[0][:,1:]
        for frame in range(len(self.vactrajs)-1):
            second = self.vactrajs[frame+1][:,1:].copy(order='C')
            # shapes of first and second must be identical otherwise malloc will freak out because of invalid chunk sizing, hence we slice 'second'
            if len(self.vactrackedtrajs[frame]) != len(second):
                second = second[:len(self.vactrackedtrajs[frame]),:]
            result = find_points_in_spheres(center_coords = self.vactrackedtrajs[frame].copy(order='C'), all_coords = second, r = self.r, pbc = np.array([1,1,1]).copy(order='C'), lattice = self.cell)
            self.vactrackedtrajs[frame+1] = second[result[1][:len(second)]]


        # array w vacancies
        self.vacsvstime = np.array(list(self.vactrackedtrajs.results()), dtype = float)
        self.nvacs = len(self.vacsvstime[0,:,0])
        # put only the z-coordinate atomic data into a pandas dataframe
        ids = [atom + 1 for atom in range(0, self.natoms)]
        self.df = pd.DataFrame(index = range(0, self.pipeline.source.num_frames))
        for atom_id in ids:
            # z -coordinate only over time for each atom of interest
            self.df[atom_id] = self.atomsvstime[:, atom_id - 1, 2]
        self.cols = list(self.df)
        # calculate variance of each particle's z-trajectory
        self.variances = {}
        for col in self.cols:
            self.variances[col] = self.df.var()[col]
        # differenced dataframe (removes trends from time series data)
        self.diffdf = self.df.diff(axis = 0, periods = 1)
        # drop first row where diff results in NaN
        self.diffdf.drop(labels = 0)
        self.diffvariances = {}
        for col in self.cols:
            self.diffvariances[col] = self.diffdf.var()[col]

        # same process for the vacancies in a dataframe
        ids = [vac + 1 for vac in range(0, self.nvacs)]
        self.vacdf = pd.DataFrame(index = range(0, self.pipeline.source.num_frames))
        for vac_id in ids:
            self.vacdf[vac_id] = self.vacsvstime[:, vac_id - 1, 2]
        self.vaccols = list(self.vacdf)
        self.vacvariances = {}
        for col in self.vaccols:
            self.vacvariances[col] = self.vacdf.var()[col]
        # differenced dataframe (removes trends from time series data)
        self.vacdiffdf = self.vacdf.diff(axis = 0, periods = 1)
        # drop first row where diff results in NaN
        self.vacdiffdf.drop(labels = 0)
        self.vacdiffvariances = {}
        for col in self.vaccols:
            self.vacdiffvariances[col] = self.vacdiffdf.var()[col]

    # some methods to return the data calculated above
    def posvstime(self):
        return self.posvstime

    def atomsvstime(self):
        return self.atomsvstime

    def vacsvstime(self):
        return self.vacsvstime

    def trajs(self):
        return self.trajs

    def shortest(self):
        return self.res

    def timestep(self):
        return self.timestep

    def df(self):
        return self.df

    def diffdf(self):
        return self.diffdf

    def cols(self):
        return  self.cols

    def variances(self):
        return self.variances

    def diffvariances(self):
        return self.diffvariances

    def vacdf(self):
        return self.vacdf

    def vacdiffdf(self):
        return self.vacdiffdf

    def vacvariances(self):
        return self.vacvariances

    def vacdiffvariances(self):
        return self.vacdiffvariances

    def vaccols(self):
        return self.vaccols

    # keep variances above 0.1 threshold
    def keeps(self, threshold):
        self.keeps = {}
        for key in self.variances.keys():
            # only relatively high variances are important
            if self.variances[key] > threshold:
                self.keeps[key] = self.df[key]
        return self.keeps

    def vackeeps(self,threshold):
        self.vackeeps = {}
        for key in self.vacvariances.keys():
            if self.vacvariances[key] > threshold:
                self.vackeeps[key] = self.vacdf[key]
        return self.vackeeps

    def diffkeeps(self, threshold):
        self.diffkeeps = {}
        for key in self.diffvariances.keys():
            if self.diffvariances[key] > threshold:
                self.diffkeepsp[key] = self.diffdf[key]
        return self.diffkeeps

    # plot a sampling of the trajectories over time
    def sample_ztraj(self, n):
        samples = self.df.sample(n, axis = 1)
        legend = []
        for col in list(samples):
            plt.plot(list(range(self.pipeline.source.num_frames)), self.df[col])
            legend.append(col)

        plt.legend(legend, loc = 'upper right')
        plt.show()
        return None

    def sample_vacs_ztraj(self,n):
        samples = self.vacdf.sample(n, axis = 1)
        legend = []
        for col in list(samples):
            plt.plot(list(range(self.pipeline.source.num_frames)), self.df[col])
            legend.append(col)
        plt.legend(legend, loc = 'upper right')
        plt.show()
        return None


    # Ackland-Jones analysis from OVITO vs time
    def acklandplot(self):
        for i in range(5):
            plt.plot(np.arange(1, self.pipeline.source.num_frames + 1), self.countsvstime[:,i])
        plt.legend(['Surface', 'FCC', 'HCP', 'BCC', 'ICO'])
        plt.show()
        return None

    def plot_variances(self):
        # initial z position on x axis and variance on y axis
        plt.plot([self.df[col][0] for col in self.cols], list(self.variances.values()), 'o')
        plt.show()
        return None

    def thresh_variance(self):
        leg_list = []
        # plot the trajectories that remain after filtering
        for key in self.keeps.keys():
            plt.plot(np.arange(1, self.pipeline.source.num_frames + 1), self.keeps[key])
            leg_list.append(key)
        plt.legend(leg_list, loc = 'upper right')
        plt.title('Nickel Trajectories w/ High Variance')
        plt.xlabel('Timestep')
        plt.ylabel('Z-Coordinate')

    def thresh_vacvariance(self):
        leg_list = []
        for key in self.vackeeps.keys():
            plt.plot(np.arange(1, self.pipeline.source.num_frames + 1), self.vackeeps[key])
            leg_list.append(key)
        plt.legend(leg_list, loc ='upper right')
        plt.title('Vacancy Trajectories w/ High Variance')
        plt.xlabel('Timestep')
        plt.ylabel('Z-Coordinate')


    # raw cross correlation
    def cross(self, atomid1, atomid2, differenced = False):
        if differenced:
            d1 = self.diffdf[atomid1]
            d2 = self.diffdf[atomid2]
        else:
            d1 = self.df[atomid1]
            d2 = self.df[atomid2]
        seconds = 5
        fps = 10
        rs = [crosscorr(d1,d2, lag) for lag in range(-int(seconds*fps),int(seconds*fps+1))]
        offset = np.ceil(len(rs)/2)-np.argmax(rs)
        f,ax=plt.subplots(figsize=(14,3))
        ax.plot(rs)
        ax.axvline(np.ceil(len(rs)/2),color='k',linestyle='--',label='Center')
        ax.axvline(np.argmax(rs),color='r',linestyle='--',label='Peak synchrony')
        ax.set(title=f'Offset = {offset} frames\n Atom 1 leads <> Atom 2 leads', xlabel='Offset',ylabel='Pearson r')
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels([-50, -25, 0, 25, 50])
        plt.legend()
        plt.show()
        return None

    # Windowed, time-lagged cross correlation
    def windowedcross(self, atomid1, atomid2, differenced = False):
        no_splits = 20
        samples_per_split = thousand.df.shape[0]/no_splits
        rss=[]
        seconds = 5
        fps = 10
        if differenced:
            for t in range(0, no_splits):
                d1 = self.diffdf[atomid1].loc[(t)*samples_per_split:(t+1)*samples_per_split]
                d2 = self.diffdf[atomid2].loc[(t)*samples_per_split:(t+1)*samples_per_split]
                rs = [crosscorr(d1,d2, lag) for lag in range(-int(seconds*fps),int(seconds*fps+1))]
                rss.append(rs)
        else:
            for t in range(0, no_splits):
                d1 = self.df[atomid1].loc[(t)*samples_per_split:(t+1)*samples_per_split]
                d2 = self.df[atomid2].loc[(t)*samples_per_split:(t+1)*samples_per_split]
                rs = [crosscorr(d1,d2, lag) for lag in range(-int(seconds*fps),int(seconds*fps+1))]
                rss.append(rs)
        rss = pd.DataFrame(rss)
        f,ax = plt.subplots()
        sns.heatmap(rss,cmap='coolwarm',ax=ax)
        ax.set(title=f'Windowed Time-Lagged Cross Correlation',xlabel='Offset',ylabel='Window epochs')
        #ax.set_xlim(85, 215)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels([-50, -25, 0, 25, 50])
        plt.show()
        return None

    # Rolling window, time-lagged cross correlation
    def rollingcross(self, atomid1, atomid2, differenced = False):
        seconds = 5
        fps = 10
        window_size = 300 #samples, should be a pretty high number compared to fps*sec to get good rolling averages
        t_start = 0
        t_end = t_start + window_size
        step_size = 20
        rss=[]
        while t_end < self.pipeline.source.num_frames:
            if differenced:
                d1 = self.diffdf[atomid1].iloc[t_start:t_end]
                d2 = self.diffdf[atomid2].iloc[t_start:t_end]
            else:
                d1 = self.df[atomid1].iloc[t_start:t_end]
                d2 = self.df[atomid2].iloc[t_start:t_end]
            rs = [crosscorr(d1,d2, lag, wrap=False) for lag in range(-int(seconds*fps),int(seconds*fps+1))]
            rss.append(rs)
            t_start = t_start + step_size
            t_end = t_end + step_size
        rss = pd.DataFrame(rss)

        f,ax = plt.subplots()
        sns.heatmap(rss,cmap='coolwarm',ax=ax)
        ax.set(title=f'Rolling-Window Time-Lagged Cross Correlation', xlabel='Offset',ylabel='Epochs')
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels([-50, -25, 0, 25, 50])
        plt.show()
        return None

        # Rolling window, time-lagged cross correlation for dataframes in different classes
    def rollingcrossvacs(atomid1, atomid2, differenced = False):
        seconds = 5
        fps = 10
        window_size = 300 #samples, should be a pretty high number compared to fps*sec to get good rolling averages
        t_start = 0
        t_end = t_start + window_size
        step_size = 20
        rss=[]
        while t_end < self.pipeline.source.num_frames:
            if differenced:
                d1 = self.diffdf[atomid1].iloc[t_start:t_end]
                d2 = self.vacdiffdf[atomid2].iloc[t_start:t_end]
            else:
                d1 = self.df[atomid1].iloc[t_start:t_end]
                d2 = self.vacdf[atomid2].iloc[t_start:t_end]
            rs = [crosscorr(d1,d2, lag, wrap=False) for lag in range(-int(seconds*fps),int(seconds*fps+1))]
            rss.append(rs)
            t_start = t_start + step_size
            t_end = t_end + step_size
        rss = pd.DataFrame(rss)

        f,ax = plt.subplots()
        sns.heatmap(rss,cmap='coolwarm',ax=ax)
        ax.set(title=f'Rolling-Window Time-Lagged Cross Correlation', xlabel='Offset',ylabel='Epochs')
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels([-50, -25, 0, 25, 50])
        plt.show()
        return None

os.chdir("/mnt/d/Burke/Documents/Pitt/RESEARCH/NVT Simulations/Temperature")
atomid = 2
vacid = 3
thousand = trajstats('new1000.lmp', atomid, vacid)
#nine = trajstats('new900.lmp', atomid, vacid, fluctuating = False)
#eight = trajstats('new800.lmp', atomid, vacid, fluctuating = False)
#seven = trajstats('new700.lmp', atomid, vacid, fluctuating = False)
#six = trajstats('new600.lmp', atomid, vacid, fluctuating = False)
#sanity checkos.chdir("D:/Burke/Documents/Pitt/RESEARCH/NVT Simulations/Temperature")
##atomid = 2
#vacid = 3
#thousand = trajstats('new1000.lmp', atomid, vacid, fluctuating = True)
#nine = trajstats('new900.lmp', atomid, vacid, fluctuating = False)
#eight = trajstats('new800.lmp', atomid, vacid, fluctuating = False)
#seven = trajstats('new700.lmp', atomid, vacid, fluctuating = False)
#six = trajstats('new600.lmp', atomid, vacid, fluctuating = False)
#sanity check
thou_df = thousand.df
print(thou_df.index.to_list)
