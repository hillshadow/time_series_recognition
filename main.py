# coding: utf-8

import matplotlib.pyplot as plt
import os.path
from storage import load as ld
from storage.save import save_segments

from utilities.variables import activities, intervalles
# 
# * Classification => a prior, no problem
#     * nearest_neighbor
#     * split_time_series
#     * label_majoritaire
#     * classification
# 
# * Time Series Segmentation
#     * Lissage and differentiation => to improve ? how ?
#     * Detection of the breaking points => to absolutely improve ! But how ?
#     * Segments segmentation_construction
#     * Computation of the local minimums/maximums
#     * Breaking points selection => to test
#     
# * Data preparation
#      * Computation of the average segments => to improve !
#      * Generation of the average segments
#      
# * Data recognition
#     * recognition => to improve
#     * selection when supperposition
# 
# 

        
