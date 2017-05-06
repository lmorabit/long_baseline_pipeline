import os
from lofarpipe.support.data_map import DataMap
from lofarpipe.support.data_map import DataProduct
import numpy as np
import glob


# Leah Morabito, May 2017


def plugin_main(args, **kwargs):
    """
    Reads in closure phase file and returns the best delay calibrator mapfile

    Parameters
    ----------
    mapfile_dir : str
        Directory for output mapfile
    closurePhaseFile: str
        Name of output mapfile
    closurePhase_file: str
	Name of file with closure phase scatter

    Returns
    -------
    result : dict
        Output datamap closurePhaseFile

    """
    mapfile_dir 	= kwargs['mapfile_dir']
    mapfile_out 	= kwargs['mapfile_out']
    closurePhaseFile    = kwargs['closurePhaseFile']

    fileid    = os.path.join(mapfile_dir, mapfile_out)	           # this file holds all the output measurement sets

    # read the file
    with open( closurePhaseFile, 'r' ) as f:
	lines = f.readlines()
    f.close()

    ## get lists of directions and scatter
    direction = []
    scatter = []
    for l in lines:
        direction.append(l.split()[4])
        scatter.append(np.float(l.split()[6]))

    print direction, scatter

    ## convert to numpy arrays
    direction = np.asarray( direction )
    scatter = np.asarray( scatter )

    ## find the minimum scatter
    if len(scatter) > 1:
        min_scatter_index = np.where( scatter == np.min( scatter ) )[0]
        best_calibrator = direction[min_scatter_index][0]
    else:
        best_calibrator = direction[0][0]

    delayCal = os.getcwd() + '/' + best_calibrator + '*' + 'concat'

    delay_ms = glob.glob( delayCal )[0]

    map_out = DataMap([])
    map_out.append( DataProduct('localhost', delay_ms, False) )

    
    map_out.save(fileid)
    