.. index:: Running

********************
Running the pipeline
********************
========
Overview
========

To produce high resolution images, there are four parsets you need to run. Two of these are from `prefactor`_ and two are from the `LOFAR-VLBI`_ pipeline. After you finish running the `prefactor`_ parsets you will need to assess the outputs before starting the `LOFAR-VLBI`_ parsets. These three sections are described in more detail below.

**Running Prefactor**
 * Pre-Facet-Calibrator 
 * Pre-Facet-Target

**Assessment of Prefactor output**
 * See guidance below

**Running the LOFAR-VLBI pipeline**
 * If necessary, configuring to use your own catalogue (see `Using your own catalogue`_ )
 * Delay-Calibration
 * Split-Directions

Prefactor in particular has many other steps you can run. These are not required for running the `LOFAR-VLBI`_ pipeline and therefore will *not* be covered here. 

=================
Running Prefactor
=================

The LOFAR-VLBI pipeline makes use of prefactor solutions to apply to the data. Therefore you must pre-process your data through prefactor, both the calibrator and target pipelines. For instructions how to run prefactor on your data, please look at the `prefactor documentation`_. For any issues you encounter with prefactor, please open an issue on the `prefactor`_ repository.


* The default is now for **Pre-Facet-Calibrator.parset** to process all stations, including the international stations. You can run this with the default settings. Please check the outputs to make sure they are sensible! 

.. note::
    If your standard calibrator is either 3C 295 or 3C 196, the standard models in the prefactor github repository do not have sufficiently high resolution, but high-resolution models do exist. Please contact the long baseline working group for help. 

.. note::
    Some versions of prefactor will incorrectly apply a 'reference' 3C48 bandpass which will result in an incorrect flux scale, and possible problems with self-calibration of the international stations. Please check the `Software Requirements`_ section of this documentation for more information.

* The **Pre-Facet-Target.parset** should be run with all the standard defaults. This will copy over the solutions from Pre-Facet-Calibrator and add the self-cal phase solutions for the core and remote stations, which are necessary for the LOFAR-VLBI pipeline. Please check the outputs to make sure they are sensible!  Also note any stations which were flagged as 'bad' as you will need to pre-flag these for the LOFAR-VLBI pipeline.

* There is a mismatch between the version of losoto in the Singularity image and the one expected by the pipeline. The result is that the ``-H`` flat is not recognized in the ``h5exp_gsm`` step of ``Pre-Facet-Target.parset``.  Before running, please change the following line::

        h5exp_gsm.argument.flags                                       =   [-q,-v,-H,h5in]

to::

        h5exp_gsm.argument.flags                                       =   [-q,-v,h5in]

otherwise you will get an error.

.. note::
    Processing of interleaved datasets is not currently supported.

==============================
Assessment of Prefactor output
==============================

**1.** In the parent directory for your *Pre-Facet-Target* directory, there should be a *Pre-Facet-Target.log* file. This will tell you if any stations were deemed to be bad and automatically flagged.
The `LOFAR-VLBI`_ pipeline doesn't have this information, so you have to manually set information in *Delay-Calibration.parset* manually to flag and filter any bad stations. The lines to change are ``! flag_baselines =`` and ``! filter_baselines =``. Here is an example of the expecte syntax for flagging CS013HBA and removing it from the data::

        ## Stations to flag and filter
        ! flag_baselines        = [ CS013HBA*&&* ]
        ! filter_baselines      = !CS013HBA*&&*

Please note that the syntax is different for *flagging*, which sets the FLAG value to 1 to indicate bad data; and *filtering*, which is a selection function for NDPPP.
In this example we flag CS013HBA and all auto- and cross-correlations to/from that antenna, and then filter, i.e., select every antenna that is not CS013HBA to remove it completely from the data.  The ``!`` in front of the CS013HBA on the filter line is a negation -- i.e., telling NDPPP to select everything *except* CS013HBA.

If you need to flag and filter multiple stations, here is an example of the syntax::

        ## Stations to flag and filter
        ! flag_baselines        = [ CS013HBA*&&*,RS508HBA*&&* ]
        ! filter_baselines      = !CS013HBA*&&*;!RS508HBA*&&*


===============================
Running the LOFAR-VLBI pipeline
===============================

The `LOFAR-VLBI`_ pipeline is broken into two steps: **Delay-Calibration.parset** and **Split-Directions.parset**. The first parset does all the heavy lifting; it applies the prefactor solutions, splits out best in-field calibrator candidate, performs the delay calibration on it, and applies these corrections back to the data. The second parset takes the resulting CORRECTED_DATA, splits out the directions in which you wish to image, and runs self-calibration on them. 

Before running the pipeline, you should check:

* If there are any bad stations flagged by prefactor. These will need to be manually input into the parsets. Follow exactly the syntax for the example given in the parset.

* Check the rest of the "Please update these parameters" section. Comments in the parset(s) describe what they are. 

* Optional: if you have run the ddf-pipeline, please update the DDF options as well. If you are only using the catalogue, update the lotss_skymodel parameter to point to your output file. 

Trying a different Delay Calibrator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you find that the delay calibrator selected by the pipeline is not adequate, or you wish to try a different LBCS source, you can do this with a little manual input. 

First, you will have to modify the statefile (see `prefactor documentation`_ ) to remove the *prep_delay_dir* step and everything afterwards. You will then be able to resume the pipeline after making your catalogue changes.

Next, you need to modify (or insert) the *best_delay_calibrators.csv* file, which can be found in the **results** directory of your runtime directory. The name of the file must be *best_delay_calibrators.csv* and the following comma separated values, including the header, are required::

        Source_id,RA_LOTSS,DEC_LOTSS,Date,Time,Goodness,Flags,FT_Goodness,Quality,FT_total,Radius,Total_flux,LGZ_Size,DC_Maj

The columns *LGZ_Size* and *DC_Maj* provide complementary information, and only one of these two columns is required. The rest of the columns must be present, and it does not matter if you have more columns than what is listed above. 

All of this inforamtion is available from a combination of LBCS and LoTSS queries, but if you do not have LoTSS information then you can input dummy information to fill out the required columns. The recommended way to do this is to start with LBCS information and do the following:

* Rename *Observation* to *Source_id* 

* Rename *RA* to *RA_LOTSS* and *DEC* to *DEC_LOTSS*

* Find the radius from the pointing centre and add this (in degrees) as the *Radius* information

* Input the *Total_flux* as *1.0*

* Add an *LGZ_Size* of *20.*

The last 2 parameters are read in by the pipeline, but can be dummy values. The *LGZ_Size* is used to calculate the imaging parameters, and should be sensible for LBCS calibrators.

You should have **ONLY ONE** source in your *best_delay_calibrators.csv* file. A good way to create this new file from LBCS data is to copy the *delay_calibrators.csv* file to *best_delay_calibrators.csv*, delete all the targets except the one you wish to try, and then add the extra columns necessary. 


Self-calibration or not?
^^^^^^^^^^^^^^^^^^^^^^^^

The pipeline by default will run self-calibration and imaging as described below. Currently this assumes a point source starting model, which may not be appropriate for all sources. In the case of fainter sources with lower signal to noise, this may drive the self-calibration to an incorrect source structure. You may therefore wish to adjust the **Split-Directions.parset** to only run the ``setup`` steps. This can be done by changing line 75 of the parset to::

        pipeline.steps = [ setup ]

The resulting measurement set will be appropriate to start imaging.


Selecting imaging parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, the pipeline will run self-calibration using difmap. This is an order of magnitude faster (usually ~30 min) than any self-calibration using native LOFAR tools, and already optimised for VLBI. Difmap operates on the XX and YY polarisations independently, but the self-calibration script converts these solutions to an h5parm, applies them, and makes a Stokes I image from the corrected data using wsclean. The final self-calibrated dataset will have TEC-corrected, un-self-calibrated data in the **DATA** column and TEC + self-cal corrected data in the **CORRECTED_DATA** column. The user is free to perform more self-calibration, or re-do the self-calibration, using any tools they wish. The data at this point is already corrected for beam effects (including the array factor), so you are free to use any imaging / gain calibration software you like.

The self-calibration script run by the pipeline has the following default parameters:
* Number of pixels = 512
* Pixel scale = 50 milli-arcsec

This gives an image which is 25.6 x 25.6 arcseconds. If your source is larger than this, you will need to adjust the number of pixels, following the convention of using powers of 2 (512,1024,2048,... etc.). 

======================
Pipeline Block Diagram
======================

To aid the user, below is a block diagram of the pipeline.

.. image:: images/LB_calibration-2.png
   :width: 800
   :alt: LOFAR-VLBI block diagram

   
.. _help:

.. _Software Requirements: installation.html#software-requirements-with-singularity
.. _Using your own catalogue: configuration.html#using-your-own-catalogue
.. _genericpipeline: https://www.astron.nl/citt/genericpipeline/
.. _Singularity: https://sylabs.io/guides/3.6/user-guide/
.. _LOFAR-VLBI: https://github.com/lmorabit/lofar-vlbi
.. _LoTSS catalogue server: https://vo.astron.nl/lofartier1/lofartier1.xml/cone/form
.. _LBCS catalogue server: https://lofar-surveys.org/lbcs.html
.. _Long Baseline Pipeline GitHub issues: https://github.com/lmorabit/lofar-vlbi/issues
.. _prefactor: https://github.com/lofar-astron/prefactor
.. _prefactor documentation: https://www.astron.nl/citt/prefactor/
.. _documentation: file:///media/quasarfix/media/cep3/prefactor/docs/build/html/parset.html
.. _ddf-pipeline: https://github.com/mhardcastle/ddf-pipeline
