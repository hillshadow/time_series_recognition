'''
Created on 2 juil. 2017

@author: Philippenko
'''

from segmentation.segmentation_construction import union
from segmentation.segmentation import Segmentation
import storage.load as ld

def test_union():
    sgmtt1=''
    sgmtt2=ld.load_segmentation("USC-Activities\\JumpingUp\\manual")
    (absc,serie,order,activity,sd_serie,breaking_points,segments,average_segment,dispersion_segment)=union(sgmtt1,sgmtt2)
    sgmtt3=Segmentation(absc=absc,serie=serie,order=order,activity=activity,sd_serie=sd_serie,
                        breaking_points=breaking_points,segments=segments,
                        average_segment=average_segment,
                        dispersion_segment=dispersion_segment)
    sgmtt3.plot_breaking_points("",save=False)
    
    