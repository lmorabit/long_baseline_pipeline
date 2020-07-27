#!/usr/bin/env python2.7

"""A collection of functions for modifying HDF5 files. These functions form
loop 2 of the LOFAR long-baseline pipeline, which can be found at
https://github.com/lmorabit/long_baseline_pipeline.
"""

from __future__ import print_function
from multiprocessing import Pool
import argparse
import ast
import csv
import datetime
import fnmatch
import glob
import os
import subprocess
import numpy as np
from astropy.coordinates import SkyCoord
from scipy.interpolate import interp1d
from losoto.lib_operations import reorderAxes
import losoto.h5parm as lh5  # on CEP3, "module load losoto"
import pyrap.tables as pt
import logging

__author__ = 'Sean and Alex'

# short term ------------------------------------------------------------------
# TODO fix the big error on line 1480 and remove debugging print statements
# TODO run LB-Split-Cals on the full bandwidth with loop 2 on the end, giving
#      it 3 delay calibrations and 3 directions (2 of which are calibrators
#      and one of which is empty sky)
# TODO run loop 2 on someone else's field
# TODO put back in resiudal tec parallelisation

# middle term -----------------------------------------------------------------
# TODO switch to interpolating LoSoTo solutions with NaN
# TODO only uses default loop 3 parameters
# TODO put in a switch to tell loop 3 to take a point source if loop 3 fails
#      because PyBDSF returns a RuntimeError saying that all the pixels are
#      flagged
# TODO source selection function of what to do next, maybe sort targets by flux
#      and then take the x that are furthest away from each other but above
#      a certain statistic that Judith is working on.
# TODO ensure all hdf5s use the same reference station! It is critical that all
#      HDF5s use the same reference station. Otherwise, taking solutions for
#      different stations from different HDF5s would be incorrect. So a check
#      needs to be introduced for this (by seeing which station has solutions
#      of zero), where it raises an error if it is not the case. Better still,
#      write a function which converts the solutions from one reference station
#      to another, and do this if necessary. (First, make sure the new station
#      we are using as the reference is present in all HDF5s.)

# long term -------------------------------------------------------------------
# TODO python 3 compatibility?
# TODO from __future__ import division?
# TODO Do we use WEIGHT_SPECTRUM or WEIGHT? And does it make sense to include
#      this while adding things (not averaging)? e.g. When combining weights,
#      should I take their intersection or average? When adding data, what role
#      should the weights play?
# TODO refactor code
# TODO benchmark loop 2 and improve increase the performance
# TODO Some time axes come out with 1029 timesteps after interpolation and when
#      matched this other time axes. Is this correct? It looks suspiciously
#      like one was added by mistake but it might be to do with the fact that
#      not all time axes in solution tables are the same. Check it out.


def make_ds9_region_file(dir_dict, ds9_region_file='directions.reg',
                         radius=120, caldir=''):
    """Make a ds9 region file for the directions to be solved in and for the
    calibrators used.

    Parameters
    ----------
    dir_dict : dictionary
        Dictionary containing the directions.
    ds9_region_file : string
        Name for the region file that will be created.
    radius : float or integer, optional
        Radius of the circles to be drawn in the region file in arcseconds. The
        default is 120.
    calidr : string, optional
        Directory of the calibrator loop 3 folders. The default is ``. If
        empty, the same directory as the ds9_region_file is used.

    Returns
    -------
    None
        Nothing is returned.
    """
    if not os.path.isfile(ds9_region_file):  # if it does not already exist

        region_header = ('# Region file format: DS9 version 4.1\nglobal '
                         'color=green width=1 font="helvetica 10 normal roman"'
                         ' select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 '
                         'include=1 source=1\nfk5\n\n')

        logging.info('Creating the ds9 region file called {} for the '
                     'calibrators'.format(ds9_region_file))
        with open(ds9_region_file, 'w+') as the_file:
            the_file.write(region_header)

        # new directions
        with open(ds9_region_file, 'a+') as the_file:
            for ra, dec, unit in zip(dir_dict['ra'], dir_dict['dec'],
                                     dir_dict['unit']):
                if unit[:3] == 'rad':  # convert to degrees
                    ra = ra * 180 / np.pi
                    dec = dec * 180 / np.pi
                    unit = 'degrees'
                RA = str(ra)
                ra = str(np.round(ra, 3))
                DEC = str(dec)
                dec = str(np.round(dec, 3))
                radius = str(radius)
                the_file.write('circle(' + RA + unit[0] + ',' + DEC + unit[0] +
                               ',' + radius + '") # width=2 color=magenta text'
                               '={dir ' + ra + ', ' + dec + '}\n')

        # calibrators used
        if caldir == '':
            caldir = os.path.dirname(ds9_region_file) + '/loop3*.apply_tec'
        cals = glob.glob(caldir)
        cRA, cDEC, cUNIT = [], [], []
        for cal in cals:
            cal = cal.replace('loop3_', '')
            # taqlcmd = 'select DIRECTION from {}/POINTING limit 1'.format(cal)
            # ra, dec = pt.taql(taqlcmd)[0]['DIRECTION'].flatten()
            ra, dec = dir_from_ms(cal)
            cRA.append(ra)
            cDEC.append(dec)
            cUNIT.append('radians')

        with open(ds9_region_file, 'a+') as the_file:
            for ra, dec, unit in zip(cRA, cDEC, cUNIT):
                if unit[:3] == 'rad':  # convert to degrees
                    ra = ra * 180 / np.pi
                    dec = dec * 180 / np.pi
                    unit = 'degrees'
                RA = str(ra)
                ra = '{:.3f}'.format(ra)
                DEC = str(dec)
                dec = '{:.3f}'.format(dec)
                radius = str(radius)
                the_file.write('circle(' + RA + unit[0] + ',' + DEC + unit[0] +
                               ',' + radius + '") # width=2 color=cyan text'
                               '={cal ' + ra + ', ' + dec + '}\n')


def dir_from_ms(ms, verbose=False):
    """Gets the pointing centre (right ascension and declination) from the
    measurement set.

    Parameters
    ----------
    ms : string
        File path of the measurement set.
    verbose : boolean, optional
        If True, print information. The default is False.

    Returns
    -------
    list
        Pointing centre from the measurement set.
    """

    # previously using this casacore one liner
    # np.squeeze(tb.table(ms + '::POINTING')[0]['DIRECTION'].tolist())

    if verbose:
        logging.info('Getting direction from {}.'.format(ms))
    t = pt.table(ms, readonly=True, ack=False)
    field = pt.table(t.getkeyword('FIELD'), readonly=True, ack=False)
    directions = field.getcell('PHASE_DIR', 0)[0].tolist()  # radians
    field.close()
    t.close()

    return directions


def combine_h5s(phase_h5='', amplitude_h5='', tec_h5='', loop3_dir=''):
    """Combine two HDF5 files into one.

    A function that takes a HDF5 with phase000 and a HDF5 with amplitude000
    and phase000 for a particular direction and combines them into one HDF5.
    This is necessary for the way loop 2 works. The dir2phasesol function
    includes amplitude (and also TEC) but it reads the files from working_data
    which is the list of HDF5s that have the desired phase solutions. So this
    function allows the amplitudes to be brought into the fold. We only copy
    across the solution tables; the source and antenna tables should come along
    too.

    The next steps for expanding this function would be to take a list of HDF5
    files which can include TEC too (and anything else) and combines them. Then
    the thing to have would be for this to search the loop 3 output and take
    the latest phase and amplitude HDF5s and combine them (given just the
    directory), with any specified TEC. Finally, add a flag to say whether to
    take the solutions from loop 3 or the diagonal solutions from an initial
    calibrator.

    Parameters
    ----------
    phase_h5 : string, optional
        Name of the HDF5 file with the phase solutions. The default is ``.
    amplitude_h5 : string, optional
        Name of the HDF5 file with the diagonal solutions. The default is ``.
    tec_h5 : string, optional
        Name of the HDF5 file with the TEC solutions. The default is ``.
    loop3_dir : string, optional
        Instead of a phase_h5 and an amplitude_h5, the directory of the loop 3
        run can be given, and the function will use the phase and amplitude
        HDF5 files relating to the furthest progress. This loop 3 directory
        will take priority if it is given with phase_h5 and amplitude_h5. The
        default is ``.

    Returns
    -------
    string
        Name of the new HDF5 file containing both phase and amplitude/phase
        solutions.
    """

    if loop3_dir:
        loop3_files = [loop3_dir + '/' + f for f in os.listdir(loop3_dir) if
                       os.path.isfile(os.path.join(loop3_dir, f))]
        only_h5s = [f for f in loop3_files if f.endswith('.h5')]
        amplitude_h5s = fnmatch.filter(only_h5s, '*_A_*.h5')
        for h5 in amplitude_h5s:
            while h5 in only_h5s:
                only_h5s.remove(h5)
        phase_h5s = only_h5s
        amplitude_h5s.sort()
        phase_h5s.sort()
        amplitude_h5 = amplitude_h5s[-1]
        phase_h5 = phase_h5s[-1]
        logging.info('Using {} and {}.'.format(phase_h5, amplitude_h5))

    # get data from the h5parms
    p = lh5.h5parm(phase_h5)
    p_soltab = p.getSolset('sol000').getSoltab('phase000')
    p_source = p.getSolset('sol000').getSou().items()
    p_antenna = p.getSolset('sol000').getAnt().items()

    a = lh5.h5parm(amplitude_h5)
    a_soltab = a.getSolset('sol000').getSoltab('amplitude000')
    a_source = a.getSolset('sol000').getSou().items()
    a_antenna = a.getSolset('sol000').getAnt().items()

    t = lh5.h5parm(tec_h5)
    t_soltab_phase = t.getSolset('sol000').getSoltab('phase000')
    t_soltab_tec = t.getSolset('sol000').getSoltab('tec000')

    #t_source = t.getSolset('sol000').getSou().items()
    #t_antenna = t.getSolset('sol000').getAnt().items()

    # reorder axes
    desired_axesNames = ['time', 'freq', 'ant', 'pol'] # NOTE no 'dir' axis
    desired_tec_axesNames = ['time', 'freq', 'ant']  # NOTE no 'dir' and 'pol' axis
    p_val_reordered = reorderAxes(p_soltab.val, p_soltab.getAxesNames(),
                                  desired_axesNames)
    p_weight_reordered = reorderAxes(p_soltab.weight, p_soltab.getAxesNames(),
                                     desired_axesNames)

    a_val_reordered = reorderAxes(a_soltab.val, a_soltab.getAxesNames(),
                                    desired_axesNames)
    a_weight_reordered = reorderAxes(a_soltab.weight,
                                       a_soltab.getAxesNames(),
                                       desired_axesNames)

    t_phase_val_reordered = reorderAxes(t_soltab_phase.val,
                                        t_soltab_phase.getAxesNames(),
                                        desired_tec_axesNames)
    t_phase_weight_reordered = reorderAxes(t_soltab_phase.weight,
                                           t_soltab_phase.getAxesNames(),
                                           desired_tec_axesNames)

    t_tec_val_reordered = reorderAxes(t_soltab_tec.val,
                                        t_soltab_tec.getAxesNames(),
                                        desired_tec_axesNames)
    t_tec_weight_reordered = reorderAxes(t_soltab_phase.weight,
                                           t_soltab_tec.getAxesNames(),
                                           desired_tec_axesNames)

    # make new solution tables in the new h5parm
    new_h5 = phase_h5[:-3] + '_P_A.h5'  # lazy method
    new_h5 = phase_h5[:-3] + '_phase_diag_tec.h5'

    n = lh5.h5parm(new_h5, readonly=False)

    n_solset = n.makeSolset(solsetName='sol000')
    source_table = n_solset.obj._f_get_child('source')
    source_table.append(p_source)  # populate source and antenna tables
    antenna_table = n_solset.obj._f_get_child('antenna')
    antenna_table.append(p_antenna)

    n_solset.makeSoltab('phase',
                              axesNames=desired_axesNames,
                              axesVals=[p_soltab.time, p_soltab.freq,
                                        p_soltab.ant, p_soltab.pol],
                              vals=p_val_reordered,
                              weights=p_weight_reordered)

    n_solset.makeSoltab('amplitude',
                                  axesNames=desired_axesNames,
                                  axesVals=[a_soltab.time, a_soltab.freq,
                                            a_soltab.ant, a_soltab.pol],
                                  vals=a_val_reordered,
                                  weights=a_weight_reordered)

    n_solset.makeSoltab('tec_phase',
                                  axesNames=desired_tec_axesNames,
                                  axesVals=[t_soltab_phase.time, t_soltab_phase.freq,
                                            t_soltab_phase.ant],
                                  vals=t_phase_val_reordered,
                                  weights=t_phase_weight_reordered)

    n_solset.makeSoltab('tec',
                                  axesNames=desired_tec_axesNames,
                                  axesVals=[t_soltab_tec.time, t_soltab_tec.freq,
                                            t_soltab_tec.ant],
                                  vals=t_tec_val_reordered,
                                  weights=t_tec_weight_reordered)



    # n_amplitude_solset.makeSoltab('phase',
    #                               axesNames=desired_axesNames,
    #                               axesVals=[a_soltab_theta.time,
    #                                         a_soltab_theta.freq,
    #                                         a_soltab_theta.ant,
    #                                         a_soltab_theta.pol],
    #                               vals=a_theta_val_reordered,
    #                               weights=a_theta_weight_reordered)

    # if tec_h5:  # if a hdf5 with tec solutions is given too, put this in the
    #     #         new hdf5 also
    #     t = lh5.h5parm(tec_h5)
    #     t_soltab = t.getSolset('sol000').getSoltab('tec000')
    #
    #     t_val = reorderAxes(t_soltab.val, t_soltab.getAxesNames(),
    #                         ['time', 'freq', 'ant'])  # NOTE no 'dir' axis
    #     t_weight = reorderAxes(t_soltab.weight, t_soltab.getAxesNames(),
    #                            ['time', 'freq', 'ant'])  # NOTE no 'dir' axis
    #
    #     t_solset = n.makeSolset(solsetName='sol002')  # in new h5parm
    #     t_source = t_solset.obj._f_get_child('source')
    #     t_source.append(t.getSolset('sol000').getSou().items())
    #     t_antenna = t_solset.obj._f_get_child('antenna')
    #     t_antenna.append(t.getSolset('sol000').getAnt().items())
    #
    #     t_solset.makeSoltab('tec',
    #                         axesNames=['time', 'freq', 'ant'],
    #                         axesVals=[t_soltab.time, t_soltab.freq,
    #                                   t_soltab.ant],
    #                         vals=t_val,
    #                         weights=t_weight)



    # tidy up
    p.close()
    a.close()
    n.close()
    t.close()
    logging.info('Created', new_h5)
    return new_h5


def make_blank_mtf(mtf):
    """Create an empty master text file containing all of the LOFAR core,
    remote, and international stations, and ST001.

    Parameters
    ----------
    mtf : string
        The name of the master text file to be created.

    Returns
    -------
    String
        The name of the master text file.
    """

    mtf_header = ('# h5parm, ra, dec, ST001, RS106HBA, RS205HBA, RS208HBA, '
                  'RS210HBA, RS305HBA, RS306HBA, RS307HBA, RS310HBA, RS404HBA,'
                  ' RS406HBA, RS407HBA, RS409HBA, RS410HBA, RS503HBA, '
                  'RS508HBA, RS509HBA, DE601HBA, DE602HBA, DE603HBA, DE604HBA,'
                  ' DE605HBA, FR606HBA, SE607HBA, UK608HBA, DE609HBA, '
                  'PL610HBA, PL611HBA, PL612HBA, IE613HBA, '
                  'CS001HBA0, CS001HBA1, CS002HBA0, CS002HBA1, '
                  'CS003HBA0, CS003HBA1, CS004HBA0, CS004HBA1, '
                  'CS005HBA0, CS005HBA1, CS006HBA0, CS006HBA1, '
                  'CS007HBA0, CS007HBA1, CS011HBA0, CS011HBA1, '
                  'CS013HBA0, CS013HBA1, CS017HBA0, CS017HBA1, '
                  'CS021HBA0, CS021HBA1, CS024HBA0, CS024HBA1, '
                  'CS026HBA0, CS026HBA1, CS028HBA0, CS028HBA1, '
                  'CS030HBA0, CS030HBA1, CS031HBA0, CS031HBA1, '
                  'CS032HBA0, CS032HBA1, CS101HBA0, CS101HBA1, '
                  'CS103HBA0, CS103HBA1, CS201HBA0, CS201HBA1, '
                  'CS301HBA0, CS301HBA1, CS302HBA0, CS302HBA1, '
                  'CS401HBA0, CS401HBA1, CS501HBA0, CS501HBA1\n')
    if not os.path.isfile(mtf):  # if it does not already exist
        logging.info('Creating the master text file {}'.format(mtf))
        with open(mtf, 'w+') as the_file:
            the_file.write(mtf_header)
    else:
        logging.info('%s exists so it is not created or overwritten' % mtf)
    return mtf


def interpolate_nan(x_):
    """Interpolate NaN values.

    It uses this answer from Stack Overflow:
    https://stackoverflow.com/a/6520696/6386612. This works even if the first
    or last value is nan or if there are multiple nans. It raises an error if
    all values are nan.

    Parameters
    ----------
    x_ : list or NumPy array
        Values to interpolate.

    Returns
    -------
    NumPy array
        The interpolated values.
    """

    x_ = np.array(x_)
    if np.isnan(x_).all():  # if all values are nan
        raise ValueError('All values in the array are nan, so interpolation is'
                         ' not possible.')
    nans, x = np.isnan(x_), lambda z: z.nonzero()[0]
    x_[nans] = np.interp(x(nans), x(~nans), x_[~nans])

    return x_


def coherence_metric(xx, yy):
    """Calculates the coherence metric by comparing the XX and YY phases.

    Parameters
    ----------
    xx : list or NumPy array
        One set of phase solutions.
    yy : list or NumPy array
        The other set of phase solutions.

    Returns
    -------
    float
        The coherence metric.
    """
    try:
        xx, yy = interpolate_nan(xx), interpolate_nan(yy)
    except (ZeroDivisionError, ValueError, KeyError, IndexError, TypeError):
        # if the values are all nan, they cannot be interpolated so return a
        # coherence value of nan also (which exception is this?)
        return np.nan

    return np.nanmean(np.gradient(abs(np.unwrap(xx - yy))) ** 2)


def evaluate_solutions(h5parm, mtf, threshold=0.25, verbose=False):
    """Evaluate phase solutions in a h5parm.

    Get the direction from the h5parm. Evaluate the phase solutions in the
    h5parm for each station using the coherence metric. Determine the validity
    of each coherence metric that was calculated. Append the right ascension,
    declination, and validity to the master text file.

    Parameters
    ----------
    h5parm : string
        LOFAR HDF5 parameter file.
    mtf : string
        Master text file.
    threshold : float, optional
        Threshold to determine the goodness of the coherence metric. The
        default is 0.25.
    verbose : boolean, optional
        Whether to print extra information. The default is False.

    Returns
    -------
    dictionary
        The coherence metric for each station.
    """
    logging.info('Evaluating phase solutions in', h5parm)
    h = lh5.h5parm(h5parm)
    solname = h.getSolsetNames()[0]  # set to -1 to use only the last solset
    solset = h.getSolset(solname)
    soltab = solset.getSoltab('phase000')
    stations = soltab.ant
    source = solset.getSou()  # dictionary
    direction = np.degrees(np.array(source[list(source.keys())[0]]))  # degrees
    generator = soltab.getValuesIter(returnAxes=['freq', 'time'])
    evaluations, temporary = {}, {}  # evaluations holds the coherence metrics

    for g in generator:
        temporary[g[1]['pol'] + '_' + g[1]['ant']] = np.squeeze(g[0])

    for station in stations:
        xx = temporary['XX_' + station]
        yy = temporary['YY_' + station]
        # try/except block facilitates one or many frequency axes in the hdf5
        try:  # if there are multiple frequency axes
            cohs = []
            # this unpack will fail if there is only one frequency axis
            num_solns, num_freqs = xx.shape
            for i in range(num_freqs):
                cohs.append(coherence_metric(xx[:, i], yy[:, i]))

            coh = np.mean(cohs)
            if verbose:
                logging.info('{} {} coherence: {:.3f} ({} frequency'
                             ' axes)'.format(h5parm, station, coh, num_freqs))
            evaluations[station] = coh  # 0 = best

        except (ValueError, KeyError, IndexError, TypeError):
            # if there is one frequency axis only
            coh = coherence_metric(xx, yy)
            if verbose:
                logging.info('{} {} coherence: {:.3f}'.format(h5parm, station,
                                                              coh))
            evaluations[station] = coh  # 0 = best

    with open(mtf) as f:
        mtf_stations = list(csv.reader(f))[0][3:]  # get stations from the mtf

    with open(mtf, 'a') as f:
        f.write('{}, {}, {}'.format(h5parm, direction[0], direction[1]))

        for mtf_station in mtf_stations:
            # look up the statistic for a station and determine if it is good
            try:
                value = evaluations[mtf_station[1:]]
            except KeyError:
                value = float('nan')

            if np.isnan(value):
                f.write(', {}'.format('nan'))
            elif value < threshold:  # success
                f.write(', {}'.format(int(True)))
            else:  # fail
                f.write(', {}'.format(int(False)))

        f.write('\n')
    h.close()
    return evaluations


def dir2phasesol_wrapper(mtf, directions=[], cores=4):
    """Book-keeping to get the multiprocessing set up and running.

    Parameters
    ----------
    mtf : string
        The master text file.
    directions : list or NumPy array, optional
        Directions in radians in the form RA1, Dec1, RA2, Dec2, etc. The
        default is [].
    cores : float or integer, optional
        Number of cores to use. The default is 4.

    Returns
    -------
    list
        The names of the newly created h5parms in the directions specified.
    """
    mtf_list = []
    for i in range(int(len(directions) / 2)):
        mtf_list.append(mtf)

    directions_paired = list(zip(directions[::2], directions[1::2]))
    multiprocessing = list(zip(mtf_list, directions_paired))
    pool = Pool(cores)  # specify cores
    new_h5parms = pool.map(dir2phasesol_multiprocessing, multiprocessing)

    return new_h5parms


def interpolate_time(the_array, the_times, new_times, tec=False):
    """Given a h5parm array, it will interpolate the values in the time axis
    from whatever it is to a given value.

    Parameters
    ----------
    the_array : NumPy array
        The array of values or weights from the h5parm.
    the_times : NumPy array
        The 1D array of values along the time axis.
    new_times : NumPy array
        The 1D time axis that the values will be mapped to.
    tec : boolean, optional
        Whether TEC solutions are being used. The default is False.

    Returns
    -------
    NumPy array
        The values or weights for a h5parm expanded to fit the new time axis.
    """

    if tec:
        # get the original data
        time, freq, ant, dir_ = the_array.shape  # axes were reordered earlier

        # make the new array
        interpolated_array = np.ones(shape=(len(new_times), freq, ant, dir_))

        for a in range(ant):  # for one antenna only
            old_values = the_array[:, 0, a, 0]  # xx

            # calculate the interpolated values
            f = interp1d(the_times, old_values, kind='nearest',
                         bounds_error=False)

            new_values = f(new_times)

            # assign the interpolated values to the new array
            interpolated_array[:, 0, a, 0] = new_values  # new values

    else:
        # get the original data
        time, freq, ant, pol, dir_ = the_array.shape  # axes reordered earlier

        # make the new array
        interpolated_array = np.ones(shape=(len(new_times), freq, ant, pol,
                                            dir_))

        for a in range(ant):  # for one antenna only
            old_x_values = the_array[:, 0, a, 0, 0]  # xx
            old_y_values = the_array[:, 0, a, 1, 0]  # yy

            # calculate the interpolated values
            x1 = interp1d(the_times, old_x_values, kind='nearest',
                          bounds_error=False)
            y1 = interp1d(the_times, old_y_values, kind='nearest',
                          bounds_error=False)

            new_x_values = x1(new_times)
            new_y_values = y1(new_times)

            # assign the interpolated values to the new array
            interpolated_array[:, 0, a, 0, 0] = new_x_values  # new x values
            interpolated_array[:, 0, a, 1, 0] = new_y_values  # new y values

    return interpolated_array


def dir2phasesol_multiprocessing(args):
    """Wrapper to parallelise make_h5parm.

    Parameters
    ----------
    args : list or tuple
        Parameters to be passed to the dir2phasesol function.

    Returns
    -------
    string
        The output of the dir2phasesol function, which is the name of a new
        h5parm file.
    """

    mtf, directions = args
    return dir2phasesol(mtf=mtf, directions=directions)


def build_soltab(soltab, working_data, solset):
    """Creates a solution table from many h5parms using data from the
    temporary working file.

    Parameters
    ----------
    soltab : string
        The name of the solution table to copy solutions from.
    working_data : NumPy array
        Data providing the list of good and bad stations, which was taken from
        the temporary working file, and the goodness relates to the coherence
        metric on the phase solutions.
    solset : string
        The solution set in the HDF5 files given in the working_data that have
        the relevant solution tables. The stardard I am going with is to have
        phase000 in sol000, amplitude000/phase000 in sol001, and tec000 in
        sol002.

    Returns
    -------
    NumPy array
        Values to populate the solution table.
    NumPy array
        Weights to populate the solution table.
    NumPy array
        Time axis to populate the solution table.
    NumPy array
        Frequency axis to populate the solution table.
    NumPy array
        Antenna axis to populate the solution table.
    """

    time_mins, time_maxs, time_intervals, frequencies = [], [], [], []

    # looping through the h5parms to build a new time axis; this has to be done
    # before getting the solutions from the h5parms so they are being looped
    # over twice
    for my_line in range(len(working_data)):  # one line per station
        my_station = working_data[my_line][0]
        my_h5parm = working_data[my_line][len(working_data[my_line]) - 1]
        lo = lh5.h5parm(my_h5parm, readonly=False)
        tab = lo.getSolset(solset).getSoltab(soltab + '000')
        time_mins.append(np.min(tab.time[:]))
        time_maxs.append(np.max(tab.time[:]))
        time_intervals.append((np.max(tab.time[:]) - np.min(tab.time[:])) /
                              (len(tab.time[:]) - 1))
        frequencies.append(tab.freq[:])
        lo.close()

    # the time ranges from the lowest to the highest on the smallest interval
    num_of_steps = 1 + ((np.max(time_maxs) - np.min(time_mins)) /
                        np.min(time_intervals))
    new_time = np.linspace(np.min(time_mins), np.max(time_maxs), num_of_steps)

    # looping through the h5parms to get the solutions for the good stations
    val, weight = [], []
    for my_line in range(len(working_data)):  # one line per station
        my_station = working_data[my_line][0]
        my_h5parm = working_data[my_line][len(working_data[my_line]) - 1]
        lo = lh5.h5parm(my_h5parm, readonly=False)
        tab = lo.getSolset(solset).getSoltab(soltab + '000')
        axes_names = tab.getAxesNames()
        values = tab.val
        weights = tab.weight

        if 'dir' not in axes_names:  # add the direction dimension
            axes_names = ['dir'] + axes_names
            values = np.expand_dims(tab.val, 0)
            weights = np.expand_dims(tab.weight, 0)

        if soltab == 'tec000' or soltab == 'tec_phase000':  # tec will not have a polarisation axis
            reordered_values = reorderAxes(values, axes_names,
                                           ['time', 'freq', 'ant', 'dir'])
            reordered_weights = reorderAxes(weights, axes_names,
                                            ['time', 'freq', 'ant', 'dir'])

            for s in range(len(tab.ant[:])):  # stations
                if tab.ant[s] == my_station.strip():
                    v = reordered_values[:, :, s, :]  # time, freq, ant, dir
                    w = reordered_weights[:, :, s, :]
                    v_expanded = np.expand_dims(v, axis=2)
                    w_expanded = np.expand_dims(w, axis=2)
                    v_interpolated = interpolate_time(the_array=v_expanded,
                                                      the_times=tab.time[:],
                                                      new_times=new_time,
                                                      tec=True)
                    w_interpolated = interpolate_time(the_array=w_expanded,
                                                      the_times=tab.time[:],
                                                      new_times=new_time,
                                                      tec=True)
                    val.append(v_interpolated)
                    weight.append(w_interpolated)
            lo.close()

        else:
            reordered_values = reorderAxes(values, axes_names,
                                           ['time', 'freq', 'ant', 'pol',
                                            'dir'])
            reordered_weights = reorderAxes(weights, axes_names,
                                            ['time', 'freq', 'ant', 'pol',
                                             'dir'])

            for s in range(len(tab.ant[:])):  # stations
                if tab.ant[s] == my_station.strip():
                    v = reordered_values[:, :, s, :, :]  # tme,frq,ant,pol,dir
                    w = reordered_weights[:, :, s, :, :]
                    v_expanded = np.expand_dims(v, axis=2)
                    w_expanded = np.expand_dims(w, axis=2)
                    v_interpolated = interpolate_time(the_array=v_expanded,
                                                      the_times=tab.time[:],
                                                      new_times=new_time)
                    w_interpolated = interpolate_time(the_array=w_expanded,
                                                      the_times=tab.time[:],
                                                      new_times=new_time)
                    val.append(v_interpolated)
                    weight.append(w_interpolated)
            lo.close()

    vals = np.concatenate(val, axis=2)
    weights = np.concatenate(weight, axis=2)

    # if there is only one frequency, avearging will return a float, where we
    # want it as a list, but if there are >1 frequency it is fine
    if isinstance(np.average(frequencies, axis=0), float):
        my_freqs = [np.average(frequencies)]
    else:
        my_freqs = np.average(frequencies, axis=0)

    return vals, weights, new_time, my_freqs


def dir2phasesol(mtf, directions=[]):
    """Make a h5parm in a new direction.

    Get the directions of the h5parms from the master text file. Calculate
    the separation between a list of given directions and the h5parm
    directions. For each station, find the h5parm of smallest separation which
    has valid phase solutions. Create a new h5parm. Write these phase solutions
    to this new h5parm.

    Parameters
    ----------
    mtf : string
        Master text file with list of h5parms.
    directions : list, optional
        Right ascension and declination of one source in radians. The default
        is [].

    Returns
    -------
    string
        The new h5parm to be applied to the measurement set.
    """
    # get the direction from the master text file
    # HACK genfromtxt gives empty string for h5parms when names=True is used
    # importing them separately as a work around
    data = np.genfromtxt(mtf, delimiter=',', unpack=True, dtype=float,
                         names=True)
    h5parms = np.genfromtxt(mtf, delimiter=',', unpack=True, dtype=str,
                            usecols=0)

    # calculate the distance betweeen the ms and the h5parm directions
    # there is one entry in mtf_directions for each unique line in the mtf
    directions = SkyCoord(directions[0], directions[1], unit='rad')
    mtf_directions = {}

    if h5parms.size == 1:
        # to handle mtf files with one row which cannot be iterated over
        mtf_direction = SkyCoord(float(data['ra']), float(data['dec']),
                                 unit='deg')
        separation = directions.separation(mtf_direction)
        mtf_directions[separation] = h5parms

    else:
        for h5parm, ra, dec in zip(h5parms, data['ra'], data['dec']):
            mtf_direction = SkyCoord(float(ra), float(dec), unit='deg')
            separation = directions.separation(mtf_direction)
            mtf_directions[separation] = h5parm  # distances from ms to h5parms

    # read in the stations from the master text file
    with open(mtf) as f:
        mtf_stations = list(csv.reader(f))[0][3:]  # skip h5parm, ra, and dec
        mtf_stations = [x.lstrip() for x in mtf_stations]  # remove first space

    # find the closest h5parm which has an acceptable solution for each station
    # a forward slash is added to the ms name in case it does not end in one
    parts = {'prefix': os.path.dirname(os.path.dirname(mtf + '/')),
             'ra': directions.ra.deg,
             'dec': directions.dec.deg}

    working_file = '{prefix}/direction_{ra:.3f}_{dec:.3f}.txt'.format(**parts)
    f = open(working_file, 'w')
    successful_stations = []

    for mtf_station in mtf_stations:  # for each station
        for key in sorted(mtf_directions.keys()):  # shortest separation first
            # if there are multiple h5parms for one direction, the best
            # solutions will be at the bottom (but this should never be the
            # case)
            h5parm = mtf_directions[key]

            # this try/except block is necessary because otherwise this crashes
            # when the master text file only has one h5parm in it
            try:
                row = list(h5parms).index(h5parm)  # row in mtf
                value = data[mtf_station][row]  # boolean for h5parm + station

            except (ZeroDivisionError, ValueError, KeyError, IndexError,
                    TypeError):  # cannot remember exact error raised
                row = 0
                value = data[mtf_station]

            if value == 1 and mtf_station not in successful_stations:
                w = '{}\t{}\t{}\t{}\t{}'.format(mtf_station.ljust(8),
                                                round(key.deg, 6), int(value),
                                                row, h5parm)
                f.write('{}\n'.format(w))
                successful_stations.append(mtf_station)
    f.close()

    # create a new h5parm
    # ms = os.path.splitext(os.path.normpath(ms))[0]
    new_h5parm = ('{}/direction_{:.3f}_{:.3f}'
                  '.h5'.format(os.path.dirname(mtf),
                               np.round(directions.ra.deg, 3),
                               np.round(directions.dec.deg, 3)))
    # new_h5parm = '{}_{}_{}.h5'.format(ms, np.round(directions.ra.deg, 3),
    #                                   np.round(directions.dec.deg, 3))
    h = lh5.h5parm(new_h5parm, readonly=False)
    table = h.makeSolset()  # creates sol000
    solset = h.getSolset('sol000')  # on the new h5parm

    # get data to be copied from the working file
    working_data = np.genfromtxt(working_file, delimiter='\t', dtype=str)
    working_data = sorted(working_data.tolist())  # stations are alphabetised

    # working_data is the list of nearest stations with good solutions; if for
    # a station there is no good solution in any h5parm the new h5parm will
    # exclude that station
    val, weight = [], []
    time_mins, time_maxs, time_intervals = [], [], []
    frequencies = []

    # looping through the h5parms that will be used in the new h5parm to find
    # the shortest time interval of all h5parms being copied, and the longest
    # time span
    for my_line in range(len(working_data)):  # one line per station
        my_station = working_data[my_line][0]
        my_h5parm = working_data[my_line][len(working_data[my_line]) - 1]

        # use the station to get the relevant data to be copied from the h5parm
        lo = lh5.h5parm(my_h5parm, readonly=False)  # NB change this to True
        # NB combine_h5s puts phase in sol000 and amplitude in sol001
        phase = lo.getSolset('sol000').getSoltab('phase000')
        time = phase.time[:]
        time_mins.append(np.min(time))
        time_maxs.append(np.max(time))
        time_intervals.append((np.max(time) - np.min(time)) / (len(time) - 1))
        frequencies.append(phase.freq[:])
        lo.close()

    # the time ranges from the lowest to the highest on the smallest interval
    num_of_steps = 1 + ((np.max(time_maxs) - np.min(time_mins)) /
                        np.min(time_intervals))
    new_time = np.linspace(np.min(time_mins), np.max(time_maxs), num_of_steps)
    stations_in_correct_order = []

    # looping through the h5parms again to get the solutions for the good
    # stations needed to build the new h5parm
    for my_line in range(len(working_data)):  # one line per station
        my_station = working_data[my_line][0]
        my_h5parm = working_data[my_line][len(working_data[my_line]) - 1]

        # use the station to get the relevant data to be copied from the h5parm
        lo = lh5.h5parm(my_h5parm, readonly=False)  # NB change this to True
        phase = lo.getSolset('sol000').getSoltab('phase000')

        axes_names = phase.getAxesNames()
        values = phase.val
        weights = phase.weight

        if 'dir' not in axes_names:  # add the direction dimension
            axes_names = ['dir'] + axes_names
            values = np.expand_dims(phase.val, 0)
            weights = np.expand_dims(phase.weight, 0)

        reordered_values = reorderAxes(values, axes_names,
                                       ['time', 'freq', 'ant', 'pol', 'dir'])
        reordered_weights = reorderAxes(weights, axes_names,
                                        ['time', 'freq', 'ant', 'pol', 'dir'])

        for s in range(len(phase.ant[:])):  # stations
            if phase.ant[s] == my_station.strip():
                stations_in_correct_order.append(phase.ant[s])
                # copy values and weights
                v = reordered_values[:, :, s, :, :]  # time, freq, ant, pol, dr
                w = reordered_weights[:, :, s, :, :]  # same order as v
                v_expanded = np.expand_dims(v, axis=2)
                w_expanded = np.expand_dims(w, axis=2)
                v_interpolated = interpolate_time(the_array=v_expanded,
                                                  the_times=phase.time[:],
                                                  new_times=new_time)
                w_interpolated = interpolate_time(the_array=w_expanded,
                                                  the_times=phase.time[:],
                                                  new_times=new_time)
                val.append(v_interpolated)
                weight.append(w_interpolated)

        lo.close()

    # properties of the new h5parm
    freq = np.average(frequencies, axis=0)  # all items in the list are equal
    ant = stations_in_correct_order  # antennas that will be in the new h5parm
    pol = ['XX', 'YY']  # as standard
    dir_ = [str(directions.ra.rad) + ', ' + str(directions.dec.rad)]  # given

    vals = np.concatenate(val, axis=2)
    weights = np.concatenate(weight, axis=2)

    # write these best phase solutions to the new h5parm
    logging.info('Putting phase soltuions in sol000 in {}.'.format(new_h5parm))
    solset.makeSoltab('phase',
                      axesNames=['time', 'freq', 'ant', 'pol', 'dir'],
                      axesVals=[new_time, freq, ant, pol, dir_],
                      vals=vals,
                      weights=weights)  # creates phase000

    # copy source and antenna tables into the new h5parm
    source_soltab = {'POINTING':
                     np.array([directions.ra.rad, directions.dec.rad],
                              dtype='float32')}
    # the x, y, z coordinates of the stations should be in these arrays
    tied = {'ST001': np.array([3826557.5, 461029.06, 5064908],
                              dtype='float32')}

    core = {'CS001HBA0': np.array([3826896.235, 460979.455, 5064658.203],
                                  dtype='float32'),
            'CS001HBA1': np.array([3826979.384, 460897.597, 5064603.189],
                                  dtype='float32'),
            'CS002HBA0': np.array([3826600.961, 460953.402, 5064881.136],
                                  dtype='float32'),
            'CS002HBA1': np.array([3826565.594, 460958.110, 5064907.258],
                                  dtype='float32'),
            'CS003HBA0': np.array([3826471.348, 461000.138, 5064974.201],
                                  dtype='float32'),
            'CS003HBA1': np.array([3826517.812, 461035.258, 5064936.15],
                                  dtype='float32'),
            'CS004HBA0': np.array([3826585.626, 460865.844, 5064900.561],
                                  dtype='float32'),
            'CS004HBA1': np.array([3826579.486, 460917.48, 5064900.502],
                                  dtype='float32'),
            'CS005HBA0': np.array([3826701.16, 460989.25, 5064802.685],
                                  dtype='float32'),
            'CS005HBA1': np.array([3826631.194, 461021.815, 5064852.259],
                                  dtype='float32'),
            'CS006HBA0': np.array([3826653.783, 461136.440, 5064824.943],
                                  dtype='float32'),
            'CS006HBA1': np.array([3826612.499, 461080.298, 5064861.006],
                                  dtype='float32'),
            'CS007HBA0': np.array([3826478.715, 461083.720, 5064961.117],
                                  dtype='float32'),
            'CS007HBA1': np.array([3826538.021, 461169.731, 5064908.827],
                                  dtype='float32'),
            'CS011HBA0': np.array([3826637.421, 461227.345, 5064829.134],
                                  dtype='float32'),
            'CS011HBA1': np.array([3826648.961, 461354.241, 5064809.003],
                                  dtype='float32'),
            'CS013HBA0': np.array([3826318.954, 460856.125, 5065101.85],
                                  dtype='float32'),
            'CS013HBA1': np.array([3826402.103, 460774.267, 5065046.836],
                                  dtype='float32'),
            'CS017HBA0': np.array([3826405.095, 461507.460, 5064978.083],
                                  dtype='float32'),
            'CS017HBA1': np.array([3826499.783, 461552.498, 5064902.938],
                                  dtype='float32'),
            'CS021HBA0': np.array([3826463.502, 460533.094, 5065022.614],
                                  dtype='float32'),
            'CS021HBA1': np.array([3826368.813, 460488.057, 5065097.759],
                                  dtype='float32'),
            'CS024HBA0': np.array([3827218.193, 461403.898, 5064378.79],
                                  dtype='float32'),
            'CS024HBA1': np.array([3827123.504, 461358.861, 5064453.935],
                                  dtype='float32'),
            'CS026HBA0': np.array([3826418.227, 461805.837, 5064941.199],
                                  dtype='float32'),
            'CS026HBA1': np.array([3826335.078, 461887.696, 5064996.213],
                                  dtype='float32'),
            'CS028HBA0': np.array([3825573.134, 461324.607, 5065619.039],
                                  dtype='float32'),
            'CS028HBA1': np.array([3825656.283, 461242.749, 5065564.025],
                                  dtype='float32'),
            'CS030HBA0': np.array([3826041.577, 460323.374, 5065357.614],
                                  dtype='float32'),
            'CS030HBA1': np.array([3825958.428, 460405.233, 5065412.628],
                                  dtype='float32'),
            'CS031HBA0': np.array([3826383.037, 460279.343, 5065105.85],
                                  dtype='float32'),
            'CS031HBA1': np.array([3826477.725, 460324.381, 5065030.705],
                                  dtype='float32'),
            'CS032HBA0': np.array([3826864.262, 460451.924, 5064730.006],
                                  dtype='float32'),
            'CS032HBA1': np.array([3826947.411, 460370.066, 5064674.992],
                                  dtype='float32'),
            'CS101HBA0': np.array([3825899.977, 461698.906, 5065339.205],
                                  dtype='float32'),
            'CS101HBA1': np.array([3825805.288, 461653.869, 5065414.35],
                                  dtype='float32'),
            'CS103HBA0': np.array([3826331.59, 462759.074, 5064919.62],
                                  dtype='float32'),
            'CS103HBA1': np.array([3826248.441, 462840.933, 5064974.634],
                                  dtype='float32'),
            'CS201HBA0': np.array([3826679.281, 461855.243, 5064741.38],
                                  dtype='float32'),
            'CS201HBA1': np.array([3826690.821, 461982.139, 5064721.249],
                                  dtype='float32'),
            'CS301HBA0': np.array([3827442.564, 461050.814, 5064242.391],
                                  dtype='float32'),
            'CS301HBA1': np.array([3827431.025, 460923.919, 5064262.521],
                                  dtype='float32'),
            'CS302HBA0': np.array([3827973.226, 459728.624, 5063975.3],
                                  dtype='float32'),
            'CS302HBA1': np.array([3827890.077, 459810.483, 5064030.313],
                                  dtype='float32'),
            'CS401HBA0': np.array([3826795.752, 460158.894, 5064808.929],
                                  dtype='float32'),
            'CS401HBA1': np.array([3826784.211, 460031.993, 5064829.062],
                                  dtype='float32'),
            'CS501HBA0': np.array([3825568.82, 460647.62, 5065683.028],
                                  dtype='float32'),
            'CS501HBA1': np.array([3825663.508, 460692.658, 5065607.883],
                                  dtype='float32')}

    antenna_soltab = {'RS106HBA': np.array([3829205.598, 469142.533000,
                                            5062181.002], dtype='float32'),
                      'RS205HBA': np.array([3831479.67, 463487.529000,
                                            5060989.903], dtype='float32'),
                      'RS208HBA': np.array([3847753.31, 466962.809000,
                                            5048397.244], dtype='float32'),
                      'RS210HBA': np.array([3877827.56186, 467536.604956,
                                            5025445.584], dtype='float32'),
                      'RS305HBA': np.array([3828732.721, 454692.421000,
                                            5063850.334], dtype='float32'),
                      'RS306HBA': np.array([3829771.249, 452761.702000,
                                            5063243.181], dtype='float32'),
                      'RS307HBA': np.array([3837964.52, 449627.261000,
                                            5057357.585], dtype='float32'),
                      'RS310HBA': np.array([3845376.29, 413616.564000,
                                            5054796.341], dtype='float32'),
                      'RS404HBA': np.array([0.0, 0.0, 0.0],
                                           dtype='float32'),  # not operational
                      'RS406HBA': np.array([3818424.939, 452020.269000,
                                            5071817.644], dtype='float32'),
                      'RS407HBA': np.array([3811649.455, 453459.894000,
                                            5076728.952], dtype='float32'),
                      'RS409HBA': np.array([3824812.621, 426130.330000,
                                            5069251.754], dtype='float32'),
                      'RS410HBA': np.array([0.0, 0.0, 0.0],
                                           dtype='float32'),  # not operational
                      'RS503HBA': np.array([3824138.566, 459476.972,
                                            5066858.578], dtype='float32'),
                      'RS508HBA': np.array([3797136.484, 463114.447,
                                            5086651.286], dtype='float32'),
                      'RS509HBA': np.array([3783537.525, 450130.064,
                                            5097866.146], dtype='float32'),
                      'DE601HBA': np.array([4034101.522, 487012.757,
                                            4900230.499], dtype='float32'),
                      'DE602HBA': np.array([4152568.006, 828789.153,
                                            4754362.203], dtype='float32'),
                      'DE603HBA': np.array([3940295.706, 816722.865,
                                            4932394.416], dtype='float32'),
                      'DE604HBA': np.array([3796379.823, 877614.13,
                                            5032712.528], dtype='float32'),
                      'DE605HBA': np.array([4005681.02, 450968.643,
                                            4926458.211], dtype='float32'),
                      'FR606HBA': np.array([4324016.708, 165545.525,
                                            4670271.363], dtype='float32'),
                      'SE607HBA': np.array([3370271.657, 712125.881,
                                            5349991.165], dtype='float32'),
                      'UK608HBA': np.array([4008461.941, -100376.609,
                                            4943716.874], dtype='float32'),
                      'DE609HBA': np.array([3727217.673, 655109.175,
                                            5117003.123], dtype='float32'),
                      'PL610HBA': np.array([3738462.416, 1148244.316,
                                            5021710.658], dtype='float32'),
                      'PL611HBA': np.array([3850980.881, 1438994.879,
                                            4860498.993], dtype='float32'),
                      'PL612HBA': np.array([3551481.817, 1334203.573,
                                            5110157.41], dtype='float32'),
                      'IE613HBA': np.array([3801692.0, -528983.94,
                                            5076958.0], dtype='float32')}

    # delete a key, value pair from the antenna table if it does not exist in
    # the antenna axis
    keys_to_remove = []
    for key in antenna_soltab:
        if key not in ant:
            keys_to_remove.append(key)

    for k in keys_to_remove:
        antenna_soltab.pop(k, None)

    for a in ant:
        if a[:2] == 'ST':
            antenna_soltab.update(tied)  # there will only be the tied station
        if a[:2] == 'CS':
            antenna_soltab.update(core)
            break  # only add the core stations to the antenna table once

    source_table = table.obj._f_get_child('source')
    source_table.append(source_soltab.items())  # from dictionary to list
    antenna_table = table.obj._f_get_child('antenna')
    antenna_table.append(antenna_soltab.items())  # from dictionary to list

    try:  # bring across amplitude solutions if there are any
        vals, weights, time, freq = build_soltab(soltab='amplitude',
                                                 working_data=working_data,
                                                 solset='sol000')
        q = new_h5parm
        logging.info('Putting amplitude soltuions in sol001 in {}.'.format(q))
        amp_solset = h.makeSolset('sol001')
        amp_solset.makeSoltab('amplitude',
                              axesNames=['time', 'freq', 'ant', 'pol', 'dir'],
                              axesVals=[time, freq, ant, pol, dir_],
                              vals=vals,
                              weights=weights)  # creates amplitude000
        # amplitude solutions have a phase component too
        vals, weights, time, freq = build_soltab(soltab='phase',
                                                 working_data=working_data,
                                                 solset='sol000')
        amp_solset.makeSoltab('phase',
                              axesNames=['time', 'freq', 'ant', 'pol', 'dir'],
                              axesVals=[time, freq, ant, pol, dir_],
                              vals=vals,
                              weights=weights)  # creates phase000

        # make source and antenna tables
        # using the source and antenna tables from phase as they should be the
        # same (where a station is included is based on the phase coherences
        # per station)
        amp_source = amp_solset.obj._f_get_child('source')
        amp_source.append(source_soltab.items())  # from dictionary to list
        amp_antenna = amp_solset.obj._f_get_child('antenna')
        amp_antenna.append(antenna_soltab.items())  # from dictionary to list

    except (ZeroDivisionError, ValueError, KeyError, IndexError, TypeError):
        logging.info('No amplitude solutions found.')
        pass

    # try:  # bring across tec solutions if there are any
    vals, weights, time, freq = build_soltab(soltab='tec',
                                             working_data=working_data,
                                             solset='sol000')
    logging.info('Putting TEC soltuions in sol002 in {}.'.format(new_h5parm))
    tec_solset = h.makeSolset('sol002')
    tec_solset.makeSoltab('tec',
                          axesNames=['time', 'freq', 'ant', 'dir'],
                          axesVals=[time, freq, ant, dir_],
                          vals=vals,
                          weights=weights)  # creates tec000

    # make source and antenna tables
    tec_source = tec_solset.obj._f_get_child('source')
    tec_source.append(source_soltab.items())  # from dictionary to list
    tec_antenna = tec_solset.obj._f_get_child('antenna')
    tec_antenna.append(antenna_soltab.items())  # from dictionary to list

    h.close()  # close the new h5parm
    # os.remove(working_file)
    return new_h5parm


def residual_tec_solve(ms, column_out='DATA', solint=5, tidyup=False,
                       runnow=True, sourcedb=''):
    """Solve for TEC using NDPPP.

    Write a parset to solve for the residual TEC in the measurement set
    using Gaincal, then execute the parset using NDPPP. For information on
    NDPPP, see this URL:
    https://www.astron.nl/lofarwiki/doku.php?id=public:user_software:documentation:ndppp#gaincal
    This step will be built into loop 3 instead.

    Parameters
    ----------
    ms : string
        Filename of the measurement set.
    column_out : string, optional
        Name of the column in the measurement set to write the corrected data
        to (the default is DATA).
    solint : float, optional
        Solution interval in seconds for the TEC solve (the default is 5).
    tidyup : boolean, optional
        Whether to remove files at the end. The default is False.
    runnow : boolean, optional
        Whether to run the code now or just return the parset. The default is
        True.
    sourcedb : string, optional
        The name of the source model in sourcedb format. The default is ``, in
        which case it will look for the sky model outputted from loop 3.

    Returns
    -------
    string
        The name of the measurement set with the TEC solutions applied.
    string
        The HDF5 file containing the residual TEC solutions.
    """
    parset = os.path.dirname(ms + '/')[:-3] + '_tec.parset'
    h5parm = parset[:-11] + '_tec_00_c0.h5'
    msout = parset[:-11] + '_tec.MS'
    column_in = 'DATA'
    if sourcedb == '':
        sourcedb = ms + '_A_final-image.sky'
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logging.info('Solving for residual TEC on {}'.format(ms))
    with open(parset, 'w') as f:  # create the parset
        f.write('# created by residual_tec_solve at {}\n\n'.format(now))
        f.write('msin                       = {}\n'.format(ms))
        f.write('msin.datacolumn            = {}\n'.format(column_in))
        f.write('msout                      = {}\n'.format(msout))
        f.write('msout.datacolumn           = {}\n\n'.format(column_out))
        f.write('steps                      = [residual_tec]\n\n')
        f.write('residual_tec.type          = gaincal\n')  # ddecal either
        f.write('residual_tec.caltype       = tec\n')
        f.write('residual_tec.parmdb        = {}\n'.format(h5parm))
        f.write('residual_tec.applysolution = True\n')  # apply on the fly
        f.write('residual_tec.solint        = {}\n'.format(solint))
        f.write('residual_tec.sourcedb      = {}\n'.format(sourcedb))
    if runnow:
        subprocess.check_output(['NDPPP', parset])
        logging.info('Going to return this', msout)
        return msout, parset
    if tidyup:
        os.remove(parset)
    logging.info('MS out is:', msout, 'and parset is', parset, 'all ok?')
    return msout, parset


def apply_h5parm(h5parm, ms, col_out='DATA', solutions=['phase'], tidy=False,
                 time_step=4, freq_step=4, phase_center='', column_in='DATA',
                 phase_up="{ST001:'CS*'}", filter_cmd="'!CS*&*'",
                 execute=True):
    """Creates an NDPPP parset. Applies the output of make_h5parm to the
    measurement set.

    Parameters
    ----------
    h5parm : string
        The output of dir2phasesol.
    ms : string
        The measurement set for self-calibration.
    column_out : string, optional
        The column NDPPP writes to. The default is `DATA`.
    solutions : string, optional
        Which solutions to apply. For phase, amplitude and tec, pass ['phase',
        'amplitude', 'tec'] (where it assumes all solutions are in sol000).
        'tec_phase000' for phase, 'amplitude000' and 'phase000' for diagonal solutions
        and 'tec000' for tec solutions.
        The default is [`phase`].
    tidy : boolean, optional
        Whether to tidy up afterwards by deleting the parset. The default is
        False.
    time_step : float or int, optional
        The averaging time step. The default is 4.
    freq_step : float or int, optional
        The averaging frequency step. The default is 4.
    phase_center : string, optional
        The phase centre to shift to. The default is ``.
    column_in : string, optional
        The name of the column to use. The default is `DATA`.
    phase_up : string, optional
        The phase up command to use with NDPPP. The default is `{ST001:'CS*'}`.
        This phases the core stations up to ST001.
    filter_cmd : string, optional
        The filter command to use with NDPPP. The default is `'!CS*&*'`. This
        removes the core stations after phasing them up.
    execute : boolean, optional
        Whether to run NDPPP with the parset or just to make it and return it.
        The default is True, so it is executed.

    Returns
    -------
    string
        The name of the new measurement set.
    string
        Optionally return the name of the parset too, if execute is False.
    """

    # parset is saved in same directory as the h5parm
    parset = h5parm[:-2] + 'parset'
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    msout = h5parm[:-2] + 'MS'
    # msout looks like /data/scratch/sean/direction_133.404_20.111.MS

    with open(parset, 'w') as f:  # create the parset
        f.write('# created by apply_h5parm at {}\n\n'.format(now))
        f.write('msin                                = {}\n'.format(ms))
        f.write('msin.datacolumn                     = {}\n'.format(column_in))
        f.write('msout                               = {}\n'.format(msout))
        f.write('msout.datacolumn                    = {}\n\n'.format(col_out))


        logging.info('Applying phase, amplitude, and TEC solutions in'
                     ' %s to %s' % (h5parm, ms))
        f.write('steps                               = [phaseshift, '
                'average, stationadder, filter, apply_solutions]\n\n')


        f.write('phaseshift.type                     = phaseshift\n')
        f.write('phaseshift.phasecenter              = '
                '{}\n\n'.format(phase_center))
        f.write('average.type                        = average\n')
        f.write('average.timestep                    = {}\n'.format(time_step))
        f.write('average.freqstep                    = '
                '{}\n\n'.format(freq_step))
        f.write('stationadder.type                   = stationadder\n')
        f.write('stationadder.stations               = '
                '{}\n\n'.format(phase_up))
        f.write('filter.type                         = filter\n')
        f.write('filter.baseline                     = '
                '{}\n'.format(filter_cmd))
        f.write('filter.remove                        = True\n\n')
        f.write('apply_solutions.type                 = applycal\n')
        f.write('apply_solutions.parmdb               = {}\n'.format(h5parm))
        f.write('apply_solutions.solset               = sol000\n')
        f.write('apply_solutions.steps                = [tec_phase, amplitude '
                                                        'phase, tec]\n')
        f.write('apply_solutions.tec_phase.correction = tec_phase\n\n')
        f.write('apply_solutions.amplitude.correction = amplitude000\n\n')
        f.write('apply_solutions.phase.correction     = phase000\n\n')
        f.write('apply_solutions.tec.correction       = tec000\n\n')

    if execute:
        subprocess.check_output(['NDPPP', parset])
    else:
        return msout, parset
    if tidy:
        logging.info('Deleting the parset.')
        os.remove(parset)

    return msout


def add_amplitude_and_phase_solutions(diag_A_1, diag_P_1, diag_A_2, diag_P_2):
    """Add amplitude and phase solutions.

    Convert amplitude and phase solutions into complex numbers, add them,
    and return the amplitude and phase components of the result. The solutions
    must be on the same time axis. The solutions should just be given as a list
    (or one dimensional array) with one solutions per timestep, or as an array
    with one solution per timestep per frequency. But for XX and YY solutions,
    and for each antenna, this function should be called separately, and no
    direction axis is expected.

    Note on adding amplitudes and phases:
    If there is no amplitude, what do we do? We should not take A = 1 as this
    biases the sum. e.g. say
    [A1 = 1, P1 = 1], [A2 = 1000, P2 = 2], [A3 = 0.001, P3 = 2], then
    [A1, P1] + [A2, P2] ~ [A2, P2] and [A1, P1] + [A3, P3] ~ [A1, P1].
    i.e. the ratio of A1 to A2 or A3 biases the sum. The best course of action
    is to set the amplitude equal to the amplitude it is being added to, in the
    case there is none.

    Parameters
    ----------
    diag_A_1 : list or NumPy array
        Amplitude solutions.
    diag_P_1 : list or NumPy array
        Phase solutions.
    diag_A_2 : list or NumPy array
        Amplitude solutions.
    diag_P_2 : list or NumPy array
        Phase solutions.

    Returns
    -------
    NumPy array
        Summed amplitude solutions.
    NumPy array
        Summed phase solutions.
    """
    # convert nan to zero, otherwise nan + x = nan, not x

    diag_A_1 = np.nan_to_num(diag_A_1)
    diag_P_1 = np.nan_to_num(diag_P_1)
    diag_A_2 = np.nan_to_num(diag_A_2)
    diag_P_2 = np.nan_to_num(diag_P_2)

    if diag_A_1.ndim > 1:  # more than one frequency axis
        amplitude_final = np.zeros(diag_A_1.shape)
        phase_final = np.zeros(diag_P_1.shape)

        for i in range(diag_A_1.shape[1]):
            amplitude_1_2, phase_1_2 = [], []
            try:
                for A1, P1, A2, P2 in zip(diag_A_1[:, i], diag_P_1[:, i],
                                          diag_A_2[:, i], diag_P_2[:, i]):
                    complex_1 = A1 * complex(np.cos(P1), np.sin(P1))
                    complex_2 = A2 * complex(np.cos(P2), np.sin(P2))
                    complex_1_2 = complex_1 + complex_2

                    amplitude_1_2.append(abs(complex_1_2))
                    phase_1_2.append(np.arctan2(complex_1_2.imag,
                                                complex_1_2.real))
            except IndexError:
                logging.info('This is a hack! Fix it ASAP, it is wrong')  # NB
                # This gets an index error because the diag_A/P_1 have 6 freq
                # axes and diag_A/P_2 have 1 freq axis so defining i as the
                # range for diag_A/P_1 gives i up to 6, and at i == 2 i have
                # line 1267 saying diag_A_2[:,i] with i = 2, but i is 1 max! so
                # it fails. See https://i.imgur.com/9W1lTKo.png
                for A1, P1, A2, P2 in zip(diag_A_1[:, 0], diag_P_1[:, 0],
                                          diag_A_2[:, 0], diag_P_2[:, 0]):
                    complex_1 = A1 * complex(np.cos(P1), np.sin(P1))
                    complex_2 = A2 * complex(np.cos(P2), np.sin(P2))
                    complex_1_2 = complex_1 + complex_2

                    amplitude_1_2.append(abs(complex_1_2))
                    phase_1_2.append(np.arctan2(complex_1_2.imag,
                                                complex_1_2.real))

            amplitude_final[:, i] = amplitude_1_2
            phase_final[:, i] = phase_1_2

    else:  # only one frequency axis
        amplitude_final, phase_final = [], []

        for A1, P1, A2, P2 in zip(diag_A_1, diag_P_1, diag_A_2, diag_P_2):
            complex_1 = A1 * complex(np.cos(P1), np.sin(P1))
            complex_2 = A2 * complex(np.cos(P2), np.sin(P2))
            complex_1_2 = complex_1 + complex_2

            amplitude_final.append(abs(complex_1_2))
            phase_final.append(np.arctan2(complex_1_2.imag, complex_1_2.real))

        amplitude_final = np.array(amplitude_final)
        phase_final = np.array(phase_final)

    return amplitude_final, phase_final


def make_new_times(time1, time2):
    """Make a new time axis from two others, going from the minimum to the
    maximum with the smallest time step.

    Parameters
    ----------
    time1 : list or NumPy array
        Times.
    time2 : list or NumPy array
        Other times.

    Returns
    -------
    list
        New time axis.
    """
    times = [time1, time2]
    time_intervals = []
    for time in times:
        time_intervals.append((np.max(time) - np.min(time)) / (len(time) - 1))

    max_time = np.max([np.max(time1), np.max(time2)])
    min_time = np.min([np.min(time1), np.min(time2)])
    num_of_steps = 1 + (max_time - min_time) / np.min(time_intervals)
    new_time = np.linspace(min_time, max_time, num_of_steps)

    return new_time


def sort_axes(soltab, tec=False):
    """Add a direction axis if there is none and sort the axes into a
     predefined order.

    Parameters
    ----------
    soltab : Losoto object
        Solution table.
    tec : boolean, optional
        Specify whether or not there is a TEC axis included. The default is
        False.

    Returns
    -------
    NumPy array
        Values ordered (time, frequency, antennas, polarisation, and
        direction), where a direction axis is included.
    NumPy array
        Weights ordered (time, frequency, antennas, polarisation, and
        direction), where a direction axis is included.
    """
    axes_names = soltab.getAxesNames()
    if 'dir' not in axes_names:  # add the direction dimension
        axes_names = ['dir'] + axes_names
        values = np.expand_dims(soltab.val, 0)
        weights = np.expand_dims(soltab.weight, 0)
    else:
        values = soltab.val
        weights = soltab.weight

    if tec:
        reordered_values = reorderAxes(values, axes_names,
                                       ['time', 'freq', 'ant', 'dir'])
        reordered_weights = reorderAxes(weights, axes_names,
                                        ['time', 'freq', 'ant', 'dir'])
    else:
        reordered_values = reorderAxes(values, axes_names,
                                       ['time', 'freq', 'ant', 'pol', 'dir'])
        reordered_weights = reorderAxes(weights, axes_names,
                                        ['time', 'freq', 'ant', 'pol', 'dir'])

    return reordered_values, reordered_weights


def rejig_solsets(h5parm, is_tec=True, add_tec_to_phase=False):
    """This is a specific funtion to take a h5parm with three solsets, where
    sol000 has phase solutions (phase000), sol001 has diagonal solutions
    (amplitude000 and phase000), and sol002 has tec solutions (tec000). It adds
    the values in sol000 and sol001 and then outputs a h5parm that has one
    solset that has phase000, amplitude000 (the sum of the phase and diagonal
    solutions), and tec000.

    Parameters
    ----------
    h5parm : string
        The filename of the h5parm with the three solsets, including the
        filepath.
    is_tec : boolean, optional
        If true, the function exepcts TEC solutions to be in the HDF5 file too.
        The default value is True.
    add_tec_to_phase : boolean, optional
        If true, the function will convert the TEC solutions to phase solutions
        and add them to the phase solutions. The default is False.

    Returns
    -------
    string
        Filename (including the filepath) of the h5parm with the solutions in
        the format described above.
    """

    # get the old h5parm and create the new h5parm
    new_h5parm = h5parm[:-3] + '-ddf' + h5parm[-3:]  # name for new hdf5 file
    h1 = lh5.h5parm(h5parm)  # old h5parm
    h2 = lh5.h5parm(new_h5parm, readonly=False)  # new h5parm

    # get sol000/phase000 and sol001/phase000,amplitude000 from the old h5parm
    phase = h1.getSolset('sol000').getSoltab('tec_phase000')
    diagonal_amplitude = h1.getSolset('sol000').getSoltab('amplitude000')
    diagonal_phase = h1.getSolset('sol000').getSoltab('phase000')

    # use add_amplitude_and_phase_solutions to add sol000/phase000 to
    # sol001/phase000, amplitude000 (set the amplitude for the phase-only term
    # to the amplitude from the diagonal term)

    # get the phase and diagonal solutions on the same time axis
    time = make_new_times(phase.time, diagonal_phase.time)  # get new time axis

    # sort the soltab axes so they are the same before summing
    ph_val_srt, ph_wgt_srt = sort_axes(phase,tec=True)
    diag_A_val_srt, diag_A_wgt_srt = sort_axes(diagonal_amplitude)
    diag_P_val_srt, diag_P_wgt_srt = sort_axes(diagonal_phase)

    # interpolate solutions
    phase_val_new = interpolate_time(ph_val_srt, phase.time, time)
    phase_weight_new = interpolate_time(ph_wgt_srt, phase.time, time)
    diagonal_amplitude_val_new = interpolate_time(diag_A_val_srt,
                                                  diagonal_amplitude.time,
                                                  time)
    diagonal_amplitude_weight_new = interpolate_time(diag_A_wgt_srt,
                                                     diagonal_amplitude.time,
                                                     time)
    diagonal_phase_val_new = interpolate_time(diag_P_val_srt,
                                              diagonal_phase.time, time)
    diagonal_phase_weight_new = interpolate_time(diag_P_wgt_srt,
                                                 diagonal_phase.time, time)

    # get list of total combined antennas
    # this protects against the antennas not being in the order in each h5parm
    freq = np.mean([phase.freq, diagonal_phase.freq], axis=0)
    # get total unique list of antennas
    ant = sorted(list(set(phase.ant.tolist() + diagonal_phase.ant.tolist())))
    pol = ['XX', 'YY']
    try:
        dir_ = phase.dir  # assume phase and diagonal solutions in same dir
    except AttributeError:
        dir_ = [0]

    # add the solutions together
    default_shape = (len(time), len(phase.freq), 1, 1)  # time, freq, pol, dir
    empty_A_val = np.zeros((len(time), len(freq), len(ant), 2, 1))
    empty_A_wgt = np.zeros((len(time), len(freq), len(ant), 2, 1))
    empty_P_val = np.zeros((len(time), len(freq), len(ant), 2, 1))
    empty_P_wgt = np.zeros((len(time), len(freq), len(ant), 2, 1))

    for n in range(len(ant)):  # for each antenna in either h5parm
        antenna = ant[n]
        # set empty variables in case there is not data for all antennas
        ph_val_xx = np.zeros(default_shape)
        ph_val_yy = np.zeros(default_shape)
        ph_wgt_xx = np.zeros(default_shape)
        ph_wgt_yy = np.zeros(default_shape)
        diag_A_val_xx = np.zeros(default_shape)
        diag_A_val_yy = np.zeros(default_shape)
        diag_A_wgt_xx = np.zeros(default_shape)
        diag_A_wgt_yy = np.zeros(default_shape)
        diag_P_val_xx = np.zeros(default_shape)
        diag_P_val_yy = np.zeros(default_shape)
        diag_P_wgt_xx = np.zeros(default_shape)
        diag_P_wgt_yy = np.zeros(default_shape)

        # loop through combined antenna list adding phase & diagonal solutions
        for a in range(len(phase.ant)):
            # get values and weights from the phase soltab for this antenna
            if antenna == phase.ant[a]:
                ph_val_xx = phase_val_new[:, :, a, 0, 0]
                ph_val_yy = phase_val_new[:, :, a, 1, 0]
                ph_wgt_xx = phase_weight_new[:, :, a, 0, 0]
                ph_wgt_yy = phase_weight_new[:, :, a, 1, 0]

        # get values and weights from the diagonal soltabs for this antenna
        for a in range(len(diagonal_phase.ant)):
            if antenna == diagonal_phase.ant[a]:
                diag_A_val_xx = diagonal_amplitude_val_new[:, :, a, 0, 0]
                diag_A_val_yy = diagonal_amplitude_val_new[:, :, a, 1, 0]
                diag_A_wgt_xx = diagonal_amplitude_weight_new[:, :, a, 0, 0]
                diag_A_wgt_yy = diagonal_amplitude_weight_new[:, :, a, 1, 0]

        for a in range(len(diagonal_amplitude.ant)):
            if antenna == diagonal_amplitude.ant[a]:
                diag_P_val_xx = diagonal_phase_val_new[:, :, a, 0, 0]
                diag_P_val_yy = diagonal_phase_val_new[:, :, a, 1, 0]
                diag_P_wgt_xx = diagonal_phase_weight_new[:, :, a, 0, 0]
                diag_P_wgt_yy = diagonal_phase_weight_new[:, :, a, 1, 0]

        # convert nan to zero
        # do we want to covert nan to numbers for weights? just the weights
        # ph_val_xx = np.nan_to_num(ph_val_xx)
        # ph_val_yy = np.nan_to_num(ph_val_yy)
        # diag_A_val_xx = np.nan_to_num(diag_A_val_xx)
        # diag_A_val_yy = np.nan_to_num(diag_A_val_yy)
        # diag_P_val_xx = np.nan_to_num(diag_P_val_xx)
        # diag_P_val_yy = np.nan_to_num(diag_P_val_yy)
        ph_wgt_xx = np.nan_to_num(ph_wgt_xx)
        ph_wgt_yy = np.nan_to_num(ph_wgt_yy)
        diag_A_wgt_xx = np.nan_to_num(diag_A_wgt_xx)
        diag_A_wgt_yy = np.nan_to_num(diag_A_wgt_yy)
        diag_P_wgt_xx = np.nan_to_num(diag_P_wgt_xx)
        diag_P_wgt_yy = np.nan_to_num(diag_P_wgt_yy)

        # add them
        # here setting the amplitude of the phase only solutions to the
        # amplitude of the diagonal solutions
        amp_sum_xx, ph_sum_xx = add_amplitude_and_phase_solutions(
                                diag_A_1=diag_A_val_xx, diag_P_1=ph_val_xx,
                                diag_A_2=diag_A_val_xx, diag_P_2=diag_P_val_xx)
        amp_sum_yy, ph_sum_yy = add_amplitude_and_phase_solutions(
                                diag_A_1=diag_A_val_yy, diag_P_1=ph_val_yy,
                                diag_A_2=diag_A_val_yy, diag_P_2=diag_P_val_yy)

        # how do we want to add the weights?
        # averaging them here
        wgt_sum_xx = ph_wgt_xx + ((diag_A_wgt_xx + diag_P_wgt_xx) / 2) / 2
        wgt_sum_yy = ph_wgt_yy + ((diag_A_wgt_yy + diag_P_wgt_yy) / 2) / 2

        # populate the empty arrays with the new solutions
        empty_A_val[:, :, n, 0, 0] = amp_sum_xx
        empty_A_val[:, :, n, 1, 0] = amp_sum_yy
        empty_A_wgt[:, :, n, 0, 0] = wgt_sum_xx
        empty_A_wgt[:, :, n, 1, 0] = wgt_sum_yy
        empty_P_val[:, :, n, 0, 0] = ph_sum_xx
        empty_P_val[:, :, n, 1, 0] = ph_sum_yy
        empty_P_wgt[:, :, n, 0, 0] = wgt_sum_xx
        empty_P_wgt[:, :, n, 1, 0] = wgt_sum_yy

    amp_vals, phase_vals = empty_A_val, empty_P_val
    amp_weights, phase_weights = empty_A_wgt, empty_P_wgt

    # put the resulting amplitude and phase in sol000/amplitude000 and
    # sol000/phase000 in h5parm2 respectively
    sol000 = h2.makeSolset('sol000')
    sol000.makeSoltab('amplitude',
                      axesNames=['time', 'freq', 'ant', 'pol', 'dir'],
                      axesVals=[time, freq, ant, pol, dir_],
                      vals=amp_vals,
                      weights=amp_weights)

    sol000.makeSoltab('phase',
                      axesNames=['time', 'freq', 'ant', 'pol', 'dir'],
                      axesVals=[time, freq, ant, pol, dir_],
                      vals=phase_vals,
                      weights=phase_weights)

    # move sol002/tec000 from the h5parm to sol000/tec000 in the new h5parm
    if is_tec:
        tec = h1.getSolset('sol002').getSoltab('tec000')
        tec_srt_val, tec_srt_wgt = sort_axes(tec, tec=True)
        sol000.makeSoltab('tec',
                          axesNames=['time', 'freq', 'ant', 'dir'],
                          axesVals=[tec.time, tec.freq, tec.ant, dir_],
                          vals=tec_srt_val,
                          weights=tec_srt_wgt)

    # populate source and antenna tables
    # copy source and antenna tables into the new solution set
    source_soltab = h1.getSolset('sol000').getSou().items()  # dict to list
    antenna_soltab = h1.getSolset('sol000').getAnt().items()  # dict to list

    source_table = h2.getSolset('sol000').obj._f_get_child('source')
    source_table.append(source_soltab)
    antenna_table = h2.getSolset('sol000').obj._f_get_child('antenna')
    antenna_table.append(antenna_soltab)

    if add_tec_to_phase:  # convert tec to phase and add it to the phase
        # tec has no frequency axis so project it along the phase axis
        tec = sol000.getSoltab('tec000')  # this is the tec soltab
        phase = sol000.getSoltab('phase000')  # this is the phase soltab

        # sort as time, frequency, antenna, polarisation [and direction]
        tec_sort_value, tec_sort_weight = sort_axes(tec, tec=True)
        phase_sort_value, phase_sort_weight = sort_axes(phase, tec=True)

        # get my_phase and my_tec on the same time axis
        time_new = make_new_times(tec.time, phase.time)
        tec_interpolate_value = interpolate_time(tec_sort_value, tec.time,
                                                 time_new, tec=True)
        tec_interpolate_weight = interpolate_time(tec_sort_weight, tec.time,
                                                  time_new, tec=True)
        phase_interpolate_value = interpolate_time(phase_sort_value,
                                                   phase.time, time_new)
        phase_interpolate_weight = interpolate_time(phase_sort_weight,
                                                    phase.time, time_new)

        # get my_phase and my_tec on the same antenna axis
        antenna_new = sorted(list(set(tec.ant.tolist() +
                                      phase.ant.tolist())))
        # tec_phase_value will have the same antenna axis as tec.antenna
        # we will loop over antenna_new when adding the solutions

        # get the frequencies of the phase and feed those into tec_to_phase
        tec_phase_value, tec_phase_weight = tec_to_phase(
            tec=tec_interpolate_value, tec_weight=tec_interpolate_weight,
            frequency=phase.freq)
        # tec_phase_value is an array of phase values the same shape as tec

        # now we need to loop through antenna_new, the unique list of antennas
        # for which there are tec_phase_value or phase solutions, and add them
        empty_phase_value = np.zeros([len(time_new), len(phase.freq),
                                     len(antenna_new), 2, 1])  # pol, dir
        empty_phase_weight = np.zeros([len(time_new), len(phase.freq),
                                      len(antenna_new), 2, 1])  # pol, dir

        for i, antenna in enumerate(antenna_new):
            # initialise empty arrays in case we do not have both tec and phase
            # solutions for every antenna (time, frequency, polarisation, and
            # direction)
            tec_phase_value_xx = np.zeros([len(time_new), len(phase.freq), 1,
                                           1])
            tec_phase_value_yy = np.zeros([len(time_new), len(phase.freq), 1,
                                           1])
            tec_phase_weight_xx = np.zeros([len(time_new), len(phase.freq), 1,
                                            1])
            tec_phase_weight_yy = np.zeros([len(time_new), len(phase.freq), 1,
                                            1])
            phase_value_xx = np.zeros([len(time_new), len(phase.freq), 1, 1])
            phase_value_yy = np.zeros([len(time_new), len(phase.freq), 1, 1])
            phase_weight_xx = np.zeros([len(time_new), len(phase.freq), 1, 1])
            phase_weight_yy = np.zeros([len(time_new), len(phase.freq), 1, 1])

            for j, antenna_tec in enumerate(tec.ant):
                # there is a better way of doing this - check if it is in the
                # array and if so return the index - change it throughout
                if antenna == antenna_tec:  # we have tec solutions for it
                    tec_phase_value_xx = tec_phase_value[:, :, j, 0, 0]
                    tec_phase_value_yy = tec_phase_value[:, :, j, 1, 0]
                    tec_phase_weight_xx = tec_phase_weight[:, :, j, 0, 0]
                    tec_phase_weight_yy = tec_phase_weight[:, :, j, 1, 0]

            for j, antenna_phase in enumerate(phase.ant):
                if antenna == antenna_phase:  # we have phase solutions for it
                    phase_value_xx = phase_interpolate_value[:, :, j, 0, :]
                    phase_value_yy = phase_interpolate_value[:, :, j, 1, :]
                    phase_weight_xx = phase_interpolate_weight[:, :, j, 0, :]
                    phase_weight_yy = phase_interpolate_weight[:, :, j, 1, :]

            # set nan to zero so x + nan = x, not nan
            tec_phase_value_xx = np.nan_to_num(tec_phase_value_xx)
            tec_phase_value_xx = np.nan_to_num(tec_phase_value_yy)
            tec_phase_weight_xx = np.nan_to_num(tec_phase_weight_xx)
            tec_phase_weight_yy = np.nan_to_num(tec_phase_weight_yy)
            phase_value_xx = np.nan_to_num(phase_value_xx)
            phase_value_yy = np.nan_to_num(phase_value_yy)
            phase_weight_xx = np.nan_to_num(phase_weight_xx)
            phase_weight_yy = np.nan_to_num(phase_weight_yy)

            # now add the solutions
            # when adding solutions they should be multiplied by the weight so
            # if the weight is 0, the solution contributes 0, and 1 otherwise
            sum_value_xx = ((tec_phase_value_xx * tec_phase_weight_xx) +
                            (phase_value_xx[:, :, 0] *
                             phase_weight_xx[:, :, 0]))
            sum_value_yy = ((tec_phase_value_yy * tec_phase_weight_yy) +
                            (phase_value_yy[:, :, 0] *
                             phase_weight_yy[:, :, 0]))
            sum_weight_xx = tec_phase_weight_xx + phase_weight_xx[:, :, 0]
            sum_weight_yy = tec_phase_weight_yy + phase_weight_yy[:, :, 0]
            # if either are not zero then the new weight should not be zero
            sum_weight_xx = np.where(sum_weight_xx >= 1, 1, 0)
            sum_weight_yy = np.where(sum_weight_yy >= 1, 1, 0)

            # then populate the empty arrays
            empty_phase_value[:, :, i, 0, 0] = sum_value_xx
            empty_phase_value[:, :, i, 1, 0] = sum_value_yy
            empty_phase_weight[:, :, i, 0, 0] = sum_weight_xx
            empty_phase_weight[:, :, i, 1, 0] = sum_weight_yy
            # now the for loop moves on to the next antenna

        # at the end, reassign the now-full "empty" arrays
        phase_value_sum = empty_phase_value
        phase_weight_sum = empty_phase_weight

        # NOTE so much repeated code here...
        # write new tec + phase solutions to the solution set
        phase_sum = sol000.makeSoltab('newphase',
                                      axesNames=['time', 'freq', 'ant', 'pol',
                                                 'dir'],
                                      axesVals=[time_new, phase.freq,
                                                antenna_new, phase.pol,
                                                phase.dir],
                                      vals=phase_value_sum,
                                      weights=phase_weight_sum)

        # now remove phase000 and tec000 and rename phase001 to phase000
        logging.info('Converted TEC to phase and added it to phase.')
        tec.delete()
        phase.delete()
        phase_sum.rename('phase000')
        logging.info('Removed the TEC and phase soltabs and replaced phase000'
                     ' with the new solutions.')
        # NOTE interpolating weights, is that a good idea? change to
        # interpolating by setting things to nan instead like in the
        # calibration paper

        # to test on cep3:
        # module load lofar losoto && ipython
        # from hdf5_functions import update_list
        # update_list(initial_h5parm='/data020/scratch/sean/letsgetloopy/'
        #                            'init.h5',
        #             incremental_h5parm='/data020/scratch/sean/letsgetloopy/'
        #                                'increm.h5',
        #             mtf='/data020/scratch/sean/letsgetloopy/mtf.txt',
        #             threshold=0.25,
        #             tec_included=True)

        # it works but one issue: i map tec solutions to both xx and yy when
        # converting to phase, so now the coherence metric for those solutions
        # will be perfectly 0

    # close h5parms and delete the old h5parm
    h1.close()
    h2.close()
    os.remove(h5parm)

    return new_h5parm


def tec_to_phase(tec, tec_weight, frequency):
    """Convert TEC solutions to phase solutions. See equation 7.5 of
    https://imgur.com/95HukQw. For information on the H5parm format, see
    appendix C.1 of https://arxiv.org/pdf/1811.07954.pdf. A weight of zero
    indicates a flagged solution.

    Parameters
    ----------
    tec : array
        TEC solutions.

    tec_weight : array
        Weights for TEC solutions.

    frequency : float, int, or array
        Frequency axis. There can be a single or multiple axes.

    Returns
    -------
    array
        Phase solutions that the TEC has been converted to.
    array
        The weights associated with the TEC.
    """
    t, _, a, d = tec.shape  # direction axis but no polarisation axis
    f = 1 if type(frequency) is float else len(frequency)  # if an array
    tec_phases = np.zeros([t, f, a, 2, d])  # added polarisation axis for phase
    tec_phases_weights = np.zeros(tec_phases.shape)  # it will be the same size

    if type(frequency) is float:  # only one frequency axis
        # put the result into xx and yy
        tec_phases[:, :, :, 0, :] = -8.4479745e9 * tec[:, 0, :, :] / frequency
        tec_phases[:, :, :, 1, :] = -8.4479745e9 * tec[:, 0, :, :] / frequency
        tec_phases_weights[:, :, :, 0, :] = tec_weight[:, 0, :, :]  # xx
        tec_phases_weights[:, :, :, 1, :] = tec_weight[:, 0, :, :]  # yy

    elif type(frequency) is np.ndarray:  # eg 120 MHz, 140 MHz, and 160 MHz
        for f in range(len(frequency)):
            tec_phases[:, f, :, 0, :] = (-8.4479745e9 * tec[:, 0, :, :] /
                                         frequency[f])  # specify freq tec axis
            tec_phases[:, f, :, 1, :] = (-8.4479745e9 * tec[:, 0, :, :] /
                                         frequency[f])
            tec_phases_weights[:, f, :, 0, :] = tec_weight[:, 0, :, :]  # xx
            tec_phases_weights[:, f, :, 1, :] = tec_weight[:, 0, :, :]  # yy

    return tec_phases, tec_phases_weights


def update_list(initial_h5parm, incremental_h5parm, mtf, threshold=0.25,
                amplitudes_included=True, tec_included=True, cores=4):
    """Combine the phase solutions from the initial h5parm and the final
    h5parm. The initial h5parm contains the initial solutions and the final
    h5parm contains the incremental solutions so they need to be added to form
    the final solutions. Calls evaluate_solutions to update the master file
    with a new line appended.

    Parameters
    ----------
    initial_h5parm : string
        The initial h5parm (i.e. from dir2phasesol).
    incremental_h5parm : string
        The final h5parm from loop 3.
    mtf : string
        Master text file.
    threshold : float, optional
        Threshold determining goodness passed to evaluate_solutions. The
        default is 0.25.
    amplitude_included : boolean, optional
        Whether amplitude solutions are included. The default is True.
    tec_included : boolean, optional
        Whether TEC solutions are included. The default is True.
    cores : float or integer
        The number of cores to use. The default is 4.

    Returns
    -------
    string
        A new h5parm that is a combination of new_h5parm and loop3_h5parm.
    """

    # get solutions from new_h5parm and loop3_h5parm
    f = lh5.h5parm(initial_h5parm)  # from new_h5parm
    initial_phase = f.getSolset('sol000').getSoltab('phase000')
    try:  # h5parms from dir2phasesol have a direction, but in case not
        initial_dir = initial_phase.dir[:]
    except AttributeError:
        initial_dir = ['0']  # if it is missing

    initial_time = initial_phase.time[:]
    initial_freq = initial_phase.freq[:]
    initial_ant = initial_phase.ant[:]
    # initial_val = initial_phase.val[:]
    # initial_weight = initial_phase.weight[:]

    g = lh5.h5parm(incremental_h5parm)  # from loop3_h5parm
    # sol000 = g.getSolset('sol000')  # change to take highest solset?
    incremental_phase = g.getSolset('sol000').getSoltab('phase000')
    antenna_soltab = g.getSolset('sol000').getAnt().items()  # dict to list
    source_soltab = g.getSolset('sol000').getSou().items()  # dict to list

    try:  # may not contain a direction dimension
        dir_ = incremental_phase.dir[:]
    except AttributeError:
        dir_ = initial_dir  # if none, take it from the other h5
    incremental_time = incremental_phase.time[:]
    incremental_freq = incremental_phase.freq[:]
    incremental_ant = incremental_phase.ant[:]
    # incremental_val = incremental_phase.val[:]
    # incremental_weight = incremental_phase.weight[:]

    # for comined_h5parm
    # make val_initial and val_incremental on the same time axis
    # first, build the new time axis and order the array
    new_times = make_new_times(initial_time, incremental_time)
    initial_sorted_val, initial_sorted_weight = sort_axes(initial_phase)
    incremental_sorted_val, incremental_sorted_weight = sort_axes(
                                                        incremental_phase)

    # interpolate the solutions from both h5parms onto this new time axis
    initial_val_new = interpolate_time(initial_sorted_val, initial_time,
                                       new_times)
    initial_weight_new = interpolate_time(initial_sorted_weight, initial_time,
                                          new_times)
    incremental_val_new = interpolate_time(incremental_sorted_val,
                                           incremental_time, new_times)
    incremental_weight_new = interpolate_time(incremental_sorted_weight,
                                              incremental_time, new_times)

    # get total unique list of antennas
    # this protects against the antennas not being in the order in each h5parm
    A = list(set(initial_ant.tolist() + incremental_ant.tolist()))
    all_antennas = sorted(A)
    default_shape = (len(new_times), 1, 2, 1)
    summed_values, summed_weights = [], []

    for antenna in all_antennas:  # for each antenna in either h5parm
        # get values and weights from the first h5parm
        val1 = np.zeros(default_shape)
        wgt1 = np.zeros(default_shape)
        for ant1 in range(len(initial_ant)):
            if antenna == initial_ant[ant1]:
                val1 = initial_val_new[:, :, ant1, :, :]
                wgt1 = initial_weight_new[:, :, ant1, :, :]

        # get values and weights from the second h5parm
        val2 = np.zeros(default_shape)
        wgt2 = np.zeros(default_shape)
        for ant2 in range(len(incremental_ant)):
            if antenna == incremental_ant[ant2]:
                val2 = incremental_val_new[:, :, ant2, :, :]
                wgt2 = incremental_weight_new[:, :, ant2, :, :]

        # and add them, converting nan to zero
        val_new = np.expand_dims(np.nan_to_num(val1) + np.nan_to_num(val2),
                                 axis=2)
        wgt_new = np.expand_dims((np.nan_to_num(wgt1) +
                                  np.nan_to_num(wgt2)) / 2, axis=2)

        summed_values.append(val_new)
        summed_weights.append(wgt_new)

    vals = np.concatenate(summed_values, axis=2)
    weights = np.concatenate(summed_weights, axis=2)

    # handles multiple frequencies
    freq = np.average([initial_freq, incremental_freq], axis=0)
    pol = np.array(['XX', 'YY'])

    combined_h5parm = (os.path.splitext(initial_h5parm)[0] + '_final.h5')

    # write these best phase solutions to the combined_h5parm
    h = lh5.h5parm(combined_h5parm, readonly=False)
    table = h.makeSolset()  # creates sol000
    solset = h.getSolset('sol000')
    solset.makeSoltab('phase',
                      axesNames=['time', 'freq', 'ant', 'pol', 'dir'],
                      axesVals=[new_times, freq, all_antennas, pol, dir_],
                      vals=vals,
                      weights=weights)  # creates phase000

    # copy source and antenna tables into the new h5parm
    source_table = table.obj._f_get_child('source')
    source_table.append(source_soltab)
    antenna_table = table.obj._f_get_child('antenna')
    antenna_table.append(antenna_soltab)  # from dictionary to list

    if amplitudes_included:  # include amplitude solutions if they exist
        initial_diagonal_A = f.getSolset('sol000').getSoltab('amplitude000')
        initial_diagonal_P = f.getSolset('sol000').getSoltab('phase000')

        sol001 = g.getSolset('sol000')
        incremental_diagonal_A = sol001.getSoltab('amplitude000')
        incremental_diagonal_P = sol001.getSoltab('phase000')

        # get the two diagonal solution tables onto a new time axis
        new_diag_time = make_new_times(initial_diagonal_A.time,
                                       incremental_diagonal_A.time)

        # sort_axes adds the dir dimension and reorders the axes
        init_diag_A_val, init_diag_A_wgt = sort_axes(initial_diagonal_A)
        init_diag_P_val, init_diag_P_wgt = sort_axes(initial_diagonal_P)
        increm_diag_A_val, increm_diag_A_wgt = sort_axes(
                                               incremental_diagonal_A)
        increm_diag_P_val, increm_diag_P_wgt = sort_axes(
                                               incremental_diagonal_P)

        # interpolate the solutions in the initial and incremental tables
        init_diag_A_time = initial_diagonal_A.time
        init_diag_P_time = initial_diagonal_P.time
        incr_diag_A_time = incremental_diagonal_A.time
        incr_diag_P_time = incremental_diagonal_P.time
        incr_diag_A_val = increm_diag_A_val
        incr_diag_A_wgt = increm_diag_A_wgt
        incr_diag_P_val = increm_diag_P_val
        incr_diag_P_wgt = increm_diag_P_wgt

        init_diag_A_val_interp = interpolate_time(the_array=init_diag_A_val,
                                                  the_times=init_diag_A_time,
                                                  new_times=new_diag_time)
        init_diag_A_wgt_interp = interpolate_time(the_array=init_diag_A_wgt,
                                                  the_times=init_diag_A_time,
                                                  new_times=new_diag_time)
        init_diag_P_val_interp = interpolate_time(the_array=init_diag_P_val,
                                                  the_times=init_diag_P_time,
                                                  new_times=new_diag_time)
        init_diag_P_wgt_interp = interpolate_time(the_array=init_diag_P_wgt,
                                                  the_times=init_diag_P_time,
                                                  new_times=new_diag_time)
        increm_diag_A_val_interp = interpolate_time(the_array=incr_diag_A_val,
                                                    the_times=incr_diag_A_time,
                                                    new_times=new_diag_time)
        increm_diag_A_wgt_interp = interpolate_time(the_array=incr_diag_A_wgt,
                                                    the_times=incr_diag_A_time,
                                                    new_times=new_diag_time)
        increm_diag_P_val_interp = interpolate_time(the_array=incr_diag_P_val,
                                                    the_times=incr_diag_P_time,
                                                    new_times=new_diag_time)
        increm_diag_P_wgt_interp = interpolate_time(the_array=incr_diag_P_wgt,
                                                    the_times=incr_diag_P_time,
                                                    new_times=new_diag_time)

        # get the frequencies and the list of antennas for the new array
        new_diag_freq = np.mean([initial_diagonal_A.freq,
                                 incremental_diagonal_A.freq], axis=0)
        A = set(initial_diagonal_A.ant.tolist() +
                incremental_diagonal_A.ant.tolist())
        new_diag_ant = sorted(list(A))

        # add the diagonal solutions together
        default_shape = np.zeros((len(new_diag_time),
                                  len(new_diag_freq),
                                  1, 1))  # time, freq, pol, dir
        empty_diag_A_val = np.zeros((len(new_diag_time), len(new_diag_freq),
                                     len(new_diag_ant), 2, 1))
        empty_diag_A_wgt = np.zeros((len(new_diag_time), len(new_diag_freq),
                                    len(new_diag_ant), 2, 1))
        empty_diag_P_val = np.zeros((len(new_diag_time), len(new_diag_freq),
                                    len(new_diag_ant), 2, 1))
        empty_diag_P_wgt = np.zeros((len(new_diag_time), len(new_diag_freq),
                                    len(new_diag_ant), 2, 1))

        summed_values, summed_weights = [], []

        for n in range(len(new_diag_ant)):  # for each antenna in either h5parm
            antenna = new_diag_ant[n]
            # set empty variables in case there is not data for all antennas
            init_A_val_xx = default_shape
            init_A_val_yy = default_shape
            init_A_wgt_xx = default_shape
            init_A_wgt_yy = default_shape
            init_P_val_xx = default_shape
            init_P_val_yy = default_shape
            init_P_wgt_xx = default_shape
            init_P_wgt_yy = default_shape
            increm_A_val_xx = default_shape
            increm_A_val_yy = default_shape
            increm_A_wgt_xx = default_shape
            increm_A_wgt_yy = default_shape
            increm_P_val_xx = default_shape
            increm_P_val_yy = default_shape
            increm_P_wgt_xx = default_shape
            increm_P_wgt_yy = default_shape

            # get values and weights from the initial h5parm
            for ant in range(len(initial_diagonal_A.ant)):
                if antenna == initial_diagonal_A.ant[ant]:
                    # antenna will also equal initial_diagonal_P.ant[ant]
                    init_A_val_xx = init_diag_A_val_interp[:, :, ant, 0, 0]
                    init_A_val_yy = init_diag_A_val_interp[:, :, ant, 1, 0]
                    init_A_wgt_xx = init_diag_A_wgt_interp[:, :, ant, 0, 0]
                    init_A_wgt_yy = init_diag_A_wgt_interp[:, :, ant, 1, 0]

                    init_P_val_xx = init_diag_P_val_interp[:, :, ant, 0, 0]
                    init_P_val_yy = init_diag_P_val_interp[:, :, ant, 1, 0]
                    init_P_wgt_xx = init_diag_P_wgt_interp[:, :, ant, 0, 0]
                    init_P_wgt_yy = init_diag_P_wgt_interp[:, :, ant, 1, 0]

            # get values and weights from the incremental h5parm
            for ant in range(len(incremental_diagonal_A.ant)):
                if antenna == incremental_diagonal_A.ant[ant]:
                    increm_A_val_xx = increm_diag_A_val_interp[:, :, ant, 0, 0]
                    increm_A_val_yy = increm_diag_A_val_interp[:, :, ant, 1, 0]
                    increm_A_wgt_xx = increm_diag_A_wgt_interp[:, :, ant, 0, 0]
                    increm_A_wgt_yy = increm_diag_A_wgt_interp[:, :, ant, 1, 0]

                    increm_P_val_xx = increm_diag_P_val_interp[:, :, ant, 0, 0]
                    increm_P_val_yy = increm_diag_P_val_interp[:, :, ant, 1, 0]
                    increm_P_wgt_xx = increm_diag_P_wgt_interp[:, :, ant, 0, 0]
                    increm_P_wgt_yy = increm_diag_P_wgt_interp[:, :, ant, 1, 0]

            # add the diagonal solutions
            new_A_val_xx, new_P_val_xx = add_amplitude_and_phase_solutions(
                                         diag_A_1=init_A_val_xx,
                                         diag_P_1=init_P_val_xx,
                                         diag_A_2=increm_A_val_xx,
                                         diag_P_2=increm_P_val_xx)
            new_A_val_yy, new_P_val_yy = add_amplitude_and_phase_solutions(
                                         diag_A_1=init_A_val_yy,
                                         diag_P_1=init_P_val_yy,
                                         diag_A_2=increm_A_val_yy,
                                         diag_P_2=increm_P_val_yy)

            # combine the weights, using the commented out method (1 if all
            # weights are 1, else 0) or the below method, which takes the mean
            # calculate the new weights, where both have to be 1 for a 1,
            # otherwise set to 0
            # new_diag_wgt_xx, new_diag_wgt_yy = [], []
            # a = np.nan_to_num(init_A_wgt_xx)
            # b = np.nan_to_num(init_P_wgt_xx)
            # c = np.nan_to_num(increm_A_wgt_xx)
            # d = np.nan_to_num(increm_P_wgt_xx)
            # for init_A, init_P, increm_A, increm_P in zip(a, b, c, d):
            #     W = 1 if np.sum([init_A, init_P, increm_A, increm_P]) == 4
            #     else 0
            #     new_diag_wgt_xx.append(W)
            # a = np.nan_to_num(init_A_wgt_yy)
            # b = np.nan_to_num(init_P_wgt_yy)
            # c = np.nan_to_num(increm_A_wgt_yy)
            # d = np.nan_to_num(increm_P_wgt_yy)
            # for init_A, init_P, increm_A, increm_P in zip(a, b, c, d):
            #     W = 1 if np.sum([init_A, init_P, increm_A, increm_P]) == 4
            #     else 0
            #     new_diag_wgt_yy.append(W)

            new_A_wgt_xx = (np.nan_to_num(init_A_wgt_xx) +
                            np.nan_to_num(increm_A_wgt_xx)) / 2
            new_P_wgt_xx = (np.nan_to_num(init_P_wgt_xx) +
                            np.nan_to_num(increm_P_wgt_xx)) / 2
            new_A_wgt_yy = (np.nan_to_num(init_A_wgt_yy) +
                            np.nan_to_num(increm_A_wgt_yy)) / 2
            new_P_wgt_yy = (np.nan_to_num(init_P_wgt_yy) +
                            np.nan_to_num(increm_P_wgt_yy)) / 2

            # populate the empty arrays with the new solutions
            empty_diag_A_val[:, :, n, 0, 0] = new_A_val_xx
            empty_diag_A_val[:, :, n, 1, 0] = new_A_val_yy
            empty_diag_A_wgt[:, :, n, 0, 0] = new_A_wgt_xx
            empty_diag_A_wgt[:, :, n, 1, 0] = new_A_wgt_yy
            empty_diag_P_val[:, :, n, 0, 0] = new_P_val_xx
            empty_diag_P_val[:, :, n, 1, 0] = new_P_val_yy
            empty_diag_P_wgt[:, :, n, 0, 0] = new_P_wgt_xx
            empty_diag_P_wgt[:, :, n, 1, 0] = new_P_wgt_yy

        A_vals = empty_diag_A_val
        A_weights = empty_diag_A_wgt
        P_vals = empty_diag_P_val
        P_weights = empty_diag_P_wgt

        # write these best phase solutions to the combined_h5parm
        solset = h.makeSolset('sol001')  # creates sol001

        solset.makeSoltab('amplitude',
                          axesNames=['time', 'freq', 'ant', 'pol', 'dir'],
                          axesVals=[new_diag_time, new_diag_freq,
                                    new_diag_ant, pol, dir_],
                          vals=A_vals,
                          weights=A_weights)  # creates amplitude000

        solset.makeSoltab('phase',
                          axesNames=['time', 'freq', 'ant', 'pol', 'dir'],
                          axesVals=[new_diag_time, new_diag_freq,
                                    new_diag_ant, pol, dir_],
                          vals=P_vals,
                          weights=P_weights)  # creates phase000

        # copy source and antenna tables into the new solution set
        source_soltab = f.getSolset('sol000').getSou().items()  # dict to list
        antenna_soltab = f.getSolset('sol000').getAnt().items()  # dict to list

        source_table = solset.obj._f_get_child('source')
        source_table.append(source_soltab)
        antenna_table = solset.obj._f_get_child('antenna')
        antenna_table.append(antenna_soltab)

    if tec_included:  # include tec solutions if they exist
        # initial tec will be coming from initial_h5parm but the
        # incremental_tec will be from the residual tec solve, which will be
        # implemented into loop 3 soon

        # assign all the information to variables
        solset_tec, soltab_tec = 'sol000', 'tec000'

        initial_tec = f.getSolset(solset_tec).getSoltab(soltab_tec)
        try:  # may not contain a direction dimension
            initial_dir = initial_tec.dir[:]
        except AttributeError:
            initial_dir = ['0']  # if it is missing
        initial_time = initial_tec.time[:]
        initial_freq = initial_tec.freq[:]
        initial_ant = initial_tec.ant[:]
        # initial_val = initial_tec.val[:]
        # initial_weight = initial_tec.weight[:]

        incremental_tec = g.getSolset(solset_tec).getSoltab(soltab_tec)
        antenna_tec = g.getSolset(solset_tec).getAnt().items()
        source_tec = g.getSolset(solset_tec).getSou().items()

        try:
            incremental_dir = incremental_tec.dir[:]
        except AttributeError:
            incremental_dir = initial_dir  # if none, take it from the other h5
        incremental_time = incremental_tec.time[:]
        incremental_freq = incremental_tec.freq[:]
        incremental_ant = incremental_tec.ant[:]
        # incremental_val = incremental_tec.val[:]
        # incremental_weight = incremental_tec.weight[:]

        # for comined_h5parm, we want to get val_initial and val_incremental on
        # the same time axis, so first, build the new time axis and order the
        # array
        new_times = make_new_times(initial_time, incremental_time)
        # new_times go from the lowest minimum to the highest maximum on the
        # shortest interval
        initial_sorted_val, initial_sorted_weight = sort_axes(initial_tec,
                                                              tec=True)
        incremental_sorted_val, incremental_sorted_weight = sort_axes(
                                                            incremental_tec,
                                                            tec=True)
        # sort_axes sorts the axis into the order I want and adds a direction
        # axis if there is not one

        # interpolate the solutions from both h5parms onto this new time axis
        initial_val_new = interpolate_time(initial_sorted_val,
                                           initial_time, new_times, tec=True)
        initial_weight_new = interpolate_time(initial_sorted_weight,
                                              initial_time,
                                              new_times, tec=True)
        incremental_val_new = interpolate_time(incremental_sorted_val,
                                               incremental_time, new_times,
                                               tec=True)
        incremental_weight_new = interpolate_time(incremental_sorted_weight,
                                                  incremental_time, new_times,
                                                  tec=True)

        # protects against antennas not being in the same order in each h5parm
        A = list(set(initial_ant.tolist() + incremental_ant.tolist()))
        all_antennas = sorted(A)  # total unique list of antennas
        default_shape = (len(new_times), 1, 1)  # time, freq, dir
        summed_values, summed_weights = [], []

        # actually do the adding, where we go through each antenna and combine
        # the solutions
        for antenna in all_antennas:  # for each antenna in either h5parm
            # get values and weights from the first h5parm
            val1 = np.zeros(default_shape)
            wgt1 = np.zeros(default_shape)
            for ant1 in range(len(initial_ant)):
                if antenna == initial_ant[ant1]:
                    val1 = initial_val_new[:, :, ant1, :]
                    wgt1 = initial_weight_new[:, :, ant1, :]

            # get values and weights from the second h5parm
            val2 = np.zeros(default_shape)
            wgt2 = np.zeros(default_shape)
            for ant2 in range(len(incremental_ant)):
                if antenna == incremental_ant[ant2]:
                    val2 = incremental_val_new[:, :, ant2, :]
                    wgt2 = incremental_weight_new[:, :, ant2, :]

            # and add them, converting nan to zero
            # the values are simple addition and I avearge the weights, but
            # this is a WARNING that this may not be the desired behaviour
            val_new = np.expand_dims(np.nan_to_num(val1) +
                                     np.nan_to_num(val2), axis=2)
            wgt_new = np.expand_dims((np.nan_to_num(wgt1) +
                                      np.nan_to_num(wgt2)) / 2, axis=2)

            summed_values.append(val_new)
            summed_weights.append(wgt_new)

        # get the array into the right format
        vals = np.concatenate(summed_values, axis=2)
        weights = np.concatenate(summed_weights, axis=2)

        # handles multiple frequencies
        freq = np.average([initial_freq, incremental_freq], axis=0)

        # write these best phase solutions to the combined_h5parm
        # creates sol002 in h which is the new h5parm
        solset = h.makeSolset(solset_tec)
        solset.makeSoltab('tec',
                          axesNames=['time', 'freq', 'ant', 'dir'],
                          axesVals=[new_times, freq, all_antennas,
                                    incremental_dir],
                          vals=vals,
                          weights=weights)  # creates tec000

        # copy source and antenna tables into the new h5parm
        source_table = solset.obj._f_get_child('source')
        source_table.append(source_tec)
        antenna_table = solset.obj._f_get_child('antenna')
        antenna_table.append(antenna_tec)

    f.close()
    g.close()
    h.close()

    # now we have a h5parm with 3 solsets, sol000 has phase solutions
    # (phase000), sol001 has diagonal solutions (amplitude000 and phase000),
    # and sol002 has tec solutions (tec000) - however, we want to change this
    # to produce one hdf5 with 1 solset, which has phase000, amplitude000,
    # and tec000
    logging.info('Plotting solutions with LoSoTo, is that alright with you?')
    plot_h5(h5parm=combined_h5parm, ncpu=cores)
    # logging.info('Making final HDF5 file.')
    # rejigged_h5parm = rejig_solsets(h5parm=combined_h5parm,
    #                                 is_tec=tec_included,
    #                                 add_tec_to_phase=True)
    rejigged_h5parm = combined_h5parm

    # evaluate the solutions and update the master file
    evaluate_solutions(h5parm=rejigged_h5parm, mtf=mtf, threshold=threshold)

    return rejigged_h5parm


def plot_h5(h5parm, ncpu=4, phasesol='sol000', diagsol='sol000',
            tecsol='sol000'):
    """Make losoto plots for a h5parm.

    Parameters
    ----------
    h5parm : string
        Name of the h5parm file.
    ncpu : integer or float, optional
        The number of cores to use. The default is 4.
    phasesol : string, optional
        The name of the solset that contains the phase solutions. The default
        is `sol000`.
    diagsol : string, optional
        The name of the solset that contains the diagonal solutions. The
        default is `sol001`.
    tecsol : string, optional
        The name of the solset that contains the TEC solutions. The default is
        `sol002`.

    Returns
    -------
    NoneType
        Nothing is returned.
    """
    # e.g. h5parm = direction_133.305_19.515_final.h5
    parset = h5parm.replace('final.h5', 'losoto.parset')
    # e.g. parset = direction_133.305_19.515.MS_losoto.parset
    # print('h5parm:', h5parm)
    # print('parset:', parset)
    prefix = h5parm.replace('final.h5', '')
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(parset, 'w') as f:  # create the parset
        f.write('# created by plot_h5 in loop 2 at {}\n\n'.format(now))
        f.write('ncpu        = {}\n\n'.format(ncpu))

        f.write('[plotPhase]\n')
        f.write('plotFlag    = False\n')
        f.write('axesInPlot  = [time]\n')
        f.write('prefix      = {}_phase_\n'.format(prefix))
        f.write('soltab      = {}/tec_phase000\n'.format(phasesol))
        f.write('axisInTable = ant\n')
        f.write('operation   = PLOT\n')
        f.write('refAnt      = ST001\n')
        f.write('minmax      = [-3.14, 3.14]\n\n')

        f.write('[plotDiagonalPhase]\n')
        f.write('plotFlag    = False\n')
        f.write('axesInPlot  = [time]\n')
        f.write('prefix      = {}_diag_phase_\n'.format(prefix))
        f.write('soltab      = {}/phase000\n'.format(diagsol))
        f.write('axisInTable = ant\n')
        f.write('operation   = PLOT\n')
        f.write('refAnt      = ST001\n')
        f.write('minmax      = [-3.14, 3.14]\n\n')

        f.write('[plotDiagonalAmplitude]\n')
        f.write('plotFlag    = False\n')
        f.write('axesInPlot  = [time]\n')
        f.write('prefix      = {}_diag_amp_\n'.format(prefix))
        f.write('soltab      = {}/amplitude000\n'.format(diagsol))
        f.write('axisInTable = ant\n')
        f.write('operation   = PLOT\n\n')

        f.write('[plotTEC]\n')
        f.write('plotFlag    = False\n')
        f.write('axesInPlot  = [time]\n')
        f.write('prefix      = {}_tec_\n'.format(prefix))
        f.write('soltab      = {}/tec000\n'.format(tecsol))
        f.write('axisInTable = ant\n')
        f.write('operation   = PLOT\n')
        f.write('refAnt      = ST001\n')

    logging.info('Created {}'.format(parset))
    logging.info('Plotting solutions from {}'.format(h5parm))
    subprocess.check_output(['losoto', h5parm, parset])
    # move plots to new directory, e.g. direction_133.305_19.515_pngs
    dir_for_plots = parset.replace('_losoto.parset', '_pngs')
    os.mkdir(dir_for_plots)
    for png_image in glob.glob('direction_*.*_*.*_*_*.png'):
        png_base = os.path.basename(png_image)
        if 'XX_' in png_base:
            png_base.split('polXX_')[0] + 'XX.png'
        elif 'YY_' in png_base:
            png_base.split('polYY_')[0] + 'YY.png'
        png_base = png_base.replace(', ', '_')
        os.rename(png_image, dir_for_plots + '/' + png_base)
    add_history_to_h5parm(h5_file=h5parm)


def add_history_to_h5parm(h5_file, working_file=''):
    """Add to the phase solution table the origin of the initial phase
    solutions for any given direction.

    Parameters
    ----------
    h5_file : string
        Name of the h5parm.
    working_file : string, optional
        Name of the working file containing the information to be written to
        the phase solution table history. The file should contain the origin
        of the solutions for each antenna. The default is ``, which means the
        function will look for the file in the directory of the h5_file.

    Returns
    -------
    NoneType
        Nothing is returned.
    """
    h5 = lh5.h5parm(h5_file, readonly=False)
    sol000 = h5.getSoltset('sol000')
    soltab = sol000.getSoltab('phase000')
    working_file = h5_file.replace('final.h5', '.txt')
    with open(working_file, 'r') as f:
        data = f.read()
    print('Writing history to sol000/phase000 in {}'.format(h5_file))
    soltab.addHistory(data)
    h5.close()


def main(calibrators_ms, delaycal_ms='../L*_SB001_*_*_1*MHz.msdpppconcat',
         mtf='mtf.txt', threshold=0.25, cores=4, time_step=4, freq_step=4,
         loop3_script='../lofar-vlbi/bin/loop3B_v1.py', phase_up="{ST001:'CS*'}",
         filter_cmd="'!CS*&*'", suffix='.apply_tec.selfcal', column_in='DATA',
         directions_file='loop2_directions.csv'):
    """Run loop 2 of the LOFAR long-baseline pipeline, creating h5parms with
    solutions for any given directions.

    First, evaluate the h5parm phase solutions. Then for a given direction,
    make a new h5parm of acceptable solutions from the nearest direction for
    each station. Apply the solutions to the measurement set. Run loop 3 to
    image the measurement set in the given direction. Add the intiial solutions
    to the incremental loop 3 solutions. Update the master text file with these
    new best solutions.

    Parameters
    ----------
    calibrators_ms : string or list
        List of the calibrator measurement sets outputted by parallel_split.py.
    delaycal_ms : string, optional
        Filepath including wildcard of the LB-Delay-Calibrator.parset
        concatenated measurement sets. The default is
        `../L*_SB001_*_*_1*MHz.msdpppconcat`.
    mtf : string, optional
        Name of the master text file to be created. The default is `mtf.txt`.
    threshold : float, optional
        Threshold to determine goodness of solutions. The default is 0.25.
    cores : integer or float, optional
        Number of CPUs available when executing steps in parallel. The default
        is 4.
    time_step : string or float, optional
        Averaging step in time. For more see
        https://www.astron.nl/lofarwiki/doku.php?id=public:user_software:documentation:ndppp#averager
        The default is 4.
    freq_step : integer or float, optional
        Averaging step in frequency. For more see
        https://www.astron.nl/lofarwiki/doku.php?id=public:user_software:documentation:ndppp#averager
        The default is 4.
    loop3_script : string, optional
        Location of the loop 3 script. The default is `./loop3B_v1.py`.
    phase_up : string, optional
        Stations to phase up. For more see
        https://www.astron.nl/lofarwiki/doku.php?id=public:user_software:documentation:ndppp#stationadder
        The default is `{ST001:'CS*'}` which adds the core stations to form
        ST001.
    filter_cmd : string, optional
        Stations to filter. For more see
        https://www.astron.nl/lofarwiki/doku.php?id=public:user_software:documentation:ndppp#filter
        The default is `'!CS*&*'` which removes the core stations.
    suffix : str, optional
        String to locate measurement sets outputted by the apply_tec step in
        the LB-Split-Calibrators.parset. The default is `.apply_tec`.
    column_in : string, optional
        Name of the column in the measurement set where the data are. The
        default is `DATA`.
    directions_file : string, optional
        Filepath to the CSV containing the directions to create h5parms for.
        The file should have a header with Source_id, RA, Dec, and Units.
        Values should be comma separated. Units of radians or degrees are
        allowed. The default is `loop2_directions.csv`.

    Returns
    -------
    NoneType
        Nothing is returned.
    """
    # get the arguments passed to main() in the correct format for future use
    ms_list = ast.literal_eval(str(calibrators_ms))  # eg ['cl1.ms', 'cl2.ms']
    cores = int(cores)  # passed in as a string by default
    directions, rad_ra_list, rad_dec_list = [], [], []

    # read the directions csv and get them into a dictionary and a list
    with open(directions_file) as csvfile:
        reader = csv.DictReader(csvfile, skipinitialspace=True)
        dir_dict = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in reader.fieldnames:
                dir_dict[name].append(row[name])
    dir_dict = dict((k.lower() if isinstance(k, str) else k, v.lower()
                    if isinstance(v, str) else v) for k, v in
                    dir_dict.iteritems())

    # check if units are given in the directions file and assume radians if not
    if 'units' in dir_dict:
        dir_dict['unit'] = dir_dict.pop('units')
    rads = []
    for r in range(len(dir_dict['ra'])):
        rads.append('radians')
    if 'unit' not in dir_dict:
        dir_dict['unit'] = rads   # assume radians if no units given
    for ra, dec, unit in zip(dir_dict['ra'], dir_dict['dec'],
                             dir_dict['unit']):
        if unit[:3].lower() == 'rad':
            directions.append(float(ra))
            directions.append(float(dec))
            rad_ra_list.append(float(ra))
            rad_dec_list.append(float(dec))
        elif unit[:3].lower() == 'deg':
            directions.append(float(ra) * np.pi / 180)
            directions.append(float(dec) * np.pi / 180)
            rad_ra_list.append(float(ra) * np.pi / 180)
            rad_dec_list.append(float(dec) * np.pi / 180)
        else:
            raise NotImplementedError('Positions in {} must be in radians or '
                                      'degrees'.format(directions_file))
    dir_dict['ra'] = rad_ra_list
    dir_dict['dec'] = rad_dec_list
    dir_dict['unit'] = rads

    make_ds9_region_file(dir_dict=dir_dict,  # includes calibrators, directions
                         ds9_region_file=(os.path.dirname(os.path.dirname(mtf +
                                          '/')) + '/directions.reg'))
    make_blank_mtf(mtf=mtf)  # create the master text file if it does not exist
    sources = []
    for ms in ms_list:
        if ms.split('/')[-1][:5] != 'loop3':
            sources.append(ms.split('/')[-1][:-32])
    logging.info('Found', len(ms_list), 'sources:', ', '.join(sources))

    # for each calibrator source group the, phase, diagonal, and tec solutions
    # in one h5parm, and evaluate the goodness of the phase solutions
    for i, (ms, source) in enumerate(zip(ms_list, sources)):
        phase_h5 = glob.glob(ms[:-32]+'_sols.h5')[0]
        amplitude_h5 = glob.glob(ms[:-32]+'_sols.h5')[0]
        tec_h5 = glob.glob(ms.replace(suffix, '.ms_tec.h5'))[0]

        logging.info('Source {}/{}:'.format(i + 1, len(ms_list)), source)
        logging.info(source, 'MS:', ms)
        logging.info(source, 'phase h5parm:', phase_h5)
        logging.info(source, 'amplitude h5parm:', amplitude_h5)
        logging.info(source, 'TEC h5parm:', tec_h5, '\n')

        combined_h5 = combine_h5s(phase_h5=phase_h5,
                                  amplitude_h5=amplitude_h5,
                                  tec_h5=tec_h5)

        evaluate_solutions(h5parm=combined_h5, mtf=mtf, threshold=threshold)

    # create new h5parms with the nearest good solutions for each direction
    new_h5parms = dir2phasesol_wrapper(mtf=mtf,  # this is run in parallel
                                       directions=directions,
                                       cores=cores)

    logging.info('Built {} h5parms:'.format(len(new_h5parms)))
    for new_h5parm in new_h5parms:
        coords_str = ', '.join(new_h5parm.split('/')[-1][:-3].split('_')[-2:])
        logging.info('Direction {}: {}'.format(coords_str, new_h5parm))

    # output a measurement set per direction that is shifted and averaged; do
    # not execute the parset with ndppp when running apply_h5parm, just get the
    # newly created parset, and then run the processes in parallel
    parsets, msouts = [], []
    for new_h5parm, ra, dec in zip(new_h5parms, dir_dict['ra'],
                                   dir_dict['dec']):
        msout, parset = apply_h5parm(h5parm=new_h5parm, col_out='DATA',
                                     ms=delaycal_ms,
                                     time_step=time_step, freq_step=freq_step,
                                     phase_center=[ra, dec], phase_up=phase_up,
                                     filter_cmd=filter_cmd, tidy=False,
                                     column_in=column_in, execute=False,
                                     solutions=['phase', 'amplitude', 'tec'])
        parsets.append(parset)
        msouts.append(msout)
    logging.info('Running NDPPP in {} directions on {} CPUs in'
                 'parallel'.format(len(parsets), cores))
    processes = set()
    for name in parsets:
        processes.add(subprocess.Popen(['NDPPP', name]))
        if len(processes) >= cores:
            os.wait()
            processes.difference_update(
                [p for p in processes if p.poll() is not None])
    for p in processes:  # check if all the child processes were closed
        if p.poll() is None:
            p.wait()

    logging.info('Made {} new measurement sets:'.format(len(msouts)))
    for i, msout in enumerate(msouts):
        logging.info('{}/{}: {}'.format(i + 1, len(msouts), msout))

    # for each of measurement sets that were made, run loop 3 on them in
    # parallel
    logging.info('Running loop 3 in {} directions on {} CPUs in '
                 'parallel'.format(len(parsets), cores))
    processes = set()
    for name in msouts:
        processes.add(subprocess.Popen([loop3_script, name]))
        if len(processes) >= cores:
            os.wait()
            processes.difference_update(
                [p for p in processes if p.poll() is not None])
    for p in processes:  # check if all the child processes were closed
        if p.poll() is None:
            p.wait()

    # run combine_h5s again to put the loop 3 outputted solutions into one
    # h5parm
    msouts_tec, parsets_tec = [], []
    for ms in msouts:
        # NB check if loop 3 succeeded by checking the existence of the ms
        if os.path.exists(ms):
            # msout_tec, parset_tec = residual_tec_solve(ms=ms, runnow=False)
            msout_tec, parset_tec = residual_tec_solve(ms=ms, runnow=True)
            logging.info('Actually appending this:', msout_tec)
            msouts_tec.append(msout_tec)
            parsets_tec.append(parsets_tec)
        else:
            logging.info('Cannot find {} -  maybe loop 3 failed'.format(ms))

    logging.info('Full list looks like this:', msouts_tec, type(msouts_tec))
    try:
        logging.info('and len if a list:', len(msouts_tec))
        logging.info('And parsets! dont forget them!', type(parsets), parsets)
    except (ValueError, KeyError, IndexError, TypeError):
        pass
    # print('Solving for residual TEC in {} directions on {} CPUs in '
    #       'parallel'.format(len(parsets_tec), cores))
    # processes = set()
    # for parset in parsets_tec:
    #     processes.add(subprocess.Popen(['NDPPP', parset]))
    #     if len(processes) >= cores:
    #         os.wait()
    #         processes.difference_update(
    #             [p for p in processes if p.poll() is not None])
    # for p in processes:  # check if all the child processes were closed
    #     if p.poll() is None:
    #         p.wait()

    # put phase, amplitude and tec solutions for each direction into one h5parm
    logging.info('Collecting incremental solutions for each direction')
    combined_h5s = []
    for i, ms in enumerate(msouts_tec):
        logging.info('MS with TEC:', msouts_tec)
        phase_h5 = glob.glob(ms[:-7] + '.MS_??_c0.h5')[0]
        amplitude_h5 = glob.glob(ms[:-7] + '.MS_A_??_c0.h5')[0]
        tec_h5 = ms.replace('.MS', '_00_c0.h5')
        # dg = ', '.join(ms.split('_')[1:-1])
        logging.info('Direction {}/{}')  # {}d'.format(i+1,len(msouts_tec),dg))
        logging.info(source, 'MS:', ms)
        logging.info(source, 'phase h5parm:', phase_h5)
        logging.info(source, 'amplitude h5parm:', amplitude_h5)
        logging.info(source, 'TEC h5parm:', tec_h5, '\n')

        combined_h5 = combine_h5s(phase_h5=phase_h5,
                                  amplitude_h5=amplitude_h5,
                                  tec_h5=tec_h5)
        combined_h5s.append(combined_h5)

    # run update_list to add the incremental solutions from loop 3 to the
    # initial solutions that were used; update_list calls evaluate_solutions to
    # evaluate the goodness of these solutions
    logging.info('Updating the list')
    logging.info('msouts_tec:', msouts_tec)
    logging.info('combined_h5s:', combined_h5s)
    for msout, increm_h5 in zip(msouts_tec, combined_h5s):
        # crd = ', '.join(msout.split('_')[-3:-1])
        # print('Combining initial + incremental solutions for {}'.format(crd))
        # NB change msout[:-2] + 'h5' to increm_h5 + something, it should be eg
        # 'direction_133.305_19.515.h5'
        logging.info('starting with', msout[:-7] + '.h5', increm_h5)
        update_list(initial_h5parm=msout[:-7] + '.h5',
                    incremental_h5parm=increm_h5,
                    mtf=mtf, cores=cores,
                    threshold=threshold)

        # plot_h5(h5parm=increm_h5, ncpu=cores)  # plot solutions

    logging.info('Loop 2 is done')


if __name__ == '__main__':
    formatter_class = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=formatter_class)

    parser.add_argument('ms',
                        #required=False,
                        type=str,
                        default=('/data020/scratch/sean/letsgetloopy/SILTJ13' +
                                 '5044.06+544752.7_L693725_phasecal.' +
                                 'apply_tec'),
                        help='measurement set')

    parser.add_argument('-m',
                        '--mtf',
                        #required=False,
                        type=str,
                        default='/data020/scratch/sean/letsgetloopy/mtf.txt',
                        help='master text file')

    parser.add_argument('-t',
                        '--threshold',
                        #required=False,
                        type=float,
                        default=0.25,
                        help='threshold for the xx-yy statistic goodness')

    parser.add_argument('-n',
                        '--cores',
                        #required=False,
                        type=int,
                        default=4,
                        help='number of cores to use')

    parser.add_argument('-d',
                        '--directions_file',
                        type=str,
                        default="./sources.csv",
                        help='source positions file')

    args = parser.parse_args()
    ms = args.ms
    mtf = args.mtf
    threshold = args.threshold
    cores = args.cores
    directions_file = args.directions_file

    main(ms, delaycal_ms=ms, mtf=mtf, threshold=threshold, cores=cores,
         directions_file=directions_file)
