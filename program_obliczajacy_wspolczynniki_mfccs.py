#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:20:17 2020

@author: Edyta Bosacka
"""

import os
import librosa.display
import matplotlib.pyplot as plt
import numpy
import librosa

directory = 'E:/Nagrania'
resultsDirectory = directory + "/MFCCsresult"

for filename in os.listdir(directory):
    if filename.endswith('.wav'):
        y,sr = librosa.load(directory + "/" + filename)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, win_length= 1024, hop_length = 512)             
        print(mfccs)
        plt.figure(figsize=(10,4))
        librosa.display.specshow(mfccs, x_axis = 'time')
        plt.colorbar()
        plt.title('MFCC')
        plt.tight_layout()
        plt.show()
    
        outputFile = resultsDirectory + "/" + "Mfcc" + ".csv"
        file = open(outputFile, 'a')
        file.write(filename+"\n")
        numpy.savetxt(file, mfccs, delimiter=",") #save MFCCs as .csv
        file.close() # close file