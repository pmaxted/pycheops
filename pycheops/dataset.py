# -*- coding: utf-8 -*-
#
#   pycheops - Tools for the analysis of data from the ESA CHEOPS mission
#
#   Copyright (C) 2018  Dr Pierre Maxted, Keele University
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
"""
dataset
=======
 Object class for data access, data caching and data inspection tools

"""

from __future__ import (absolute_import, division, print_function,
                                unicode_literals)
import numpy as np
import tarfile
import re
from pathlib import Path
from .core import load_config
from astropy.io import fits
from astropy.table import Table, MaskedColumn
import matplotlib.pyplot as plt
from .instrument import transit_noise
from ftplib import FTP

_dataset_re = re.compile(r'PR(\d{2})(\d{4})_TG(\d{4})(\d{2})')

class dataset:
    """
    CHEOPS dataset object

    """

    def __init__(self, dataset_id, force_download=False, 
            configFile=None, verbose=True):

        self.dataset_id = dataset_id

        m = _dataset_re.search(dataset_id)
        if m is None:
            raise ValueError('Invalid dataset_id')
        l = [int(i) for i in m.groups()]
        self.progtype, self.prog_id, self.req_id, self.visitctr = l

        config = load_config(configFile)
        _cache_path = config['DEFAULT']['data_cache_path']
        tgzPath = Path(_cache_path,dataset_id).with_suffix('.tgz')
        self.tgzfile = str(tgzPath)

        if tgzPath.is_file():
            if verbose: print('Found archive tgzfile',self.tgzfile)
        else:
            ftp=FTP('obsftp.unige.ch')
            _ = ftp.login()
            ftp.cwd('pub/cheops/data/take_the_red_pill/ioc/ioc-c/repository/visit')
            if verbose: print('Downloading dataset from obsftp.unige.ch')
            cmd = 'RETR {}.tgz'.format(dataset_id)
            ftp.retrbinary(cmd, open(self.tgzfile, 'wb').write)
            ftp.quit()

        lisPath = Path(_cache_path,dataset_id).with_suffix('.lis')
        if lisPath.is_file():
            self.list = [line.rstrip('\n') for line in open(lisPath)]
        else:
            print('Creating dataset file list')
            tar = tarfile.open(self.tgzfile)
            self.list = tar.getnames()
            tar.close()
            with open(str(lisPath), 'w') as fh:  
                fh.writelines("%s\n" % l for l in self.list)

    def get_imagettes(self, verbose=True):
        imFile = "{}-Imagette.fits".format(self.dataset_id)
        imPath = Path(self.tgzfile).parent / imFile
        if imPath.is_file():
            with fits.open(imPath) as hdul:
                cube = hdul[1].data
                hdr = hdul[1].header
                meta = Table.read(hdul[2])
            if verbose: print ('Imagette data loaded from ',imPath)
        else:
            if verbose: print ('Extracting imagette data from ',self.tgzfile)
            tar = tarfile.open(self.tgzfile)
            r=re.compile('(.*SCI_RAW_Imagette.*.fits)' )
            datafile = list(filter(r.match, self.list))
            if len(datafile) == 0:
                raise Exception('Dataset does not contains imagette data.')
            if len(datafile) > 1:
                raise Exception('Multiple imagette data files in dataset')
            with tar.extractfile(datafile[0]) as fd:
                hdul = fits.open(fd)
                cube = hdul[1].data
                hdr = hdul[1].header
                meta = Table.read(hdul[2])
                hdul.writeto(imPath)
            if verbose: print('Saved imagette data to ',imPath)

        self.imagettes = (cube, hdr, meta)
        self.imagettes = {'data':cube, 'header':hdr, 'meta':meta}

        return cube

    def get_lightcurve(self, aperture='OPTIMAL',
            returnTable=False, reject_highpoints=True, verbose=True):

        if aperture not in ('OPTIMAL','RSUP','RINF','DEFAULT'):
            raise ValueError('Invalid aperture name')

        lcFile = "{}-{}.fits".format(self.dataset_id,aperture)
        lcPath = Path(self.tgzfile).parent / lcFile
        if lcPath.is_file(): 
            with fits.open(lcPath) as hdul:
                table = Table.read(hdul[1])
                hdr = hdul[1].header
            if verbose: print ('Light curve data loaded from ',lcPath)
        else:
            if verbose: print ('Extracting light curve from ',self.tgzfile)
            tar = tarfile.open(self.tgzfile)
            r=re.compile('(.*_SCI_COR_Lightcurve-{}_.*.fits)'.format(aperture))
            datafile = list(filter(r.match, self.list))
            if len(datafile) == 0:
                raise Exception('Dataset does not contain imagette data.')
            if len(datafile) > 1:
                raise Exception('Multiple imagette file in datset')
            with tar.extractfile(datafile[0]) as fd:
                hdul = fits.open(fd)
                table = Table.read(hdul[1])
                hdr = hdul[1].header
                hdul.writeto(lcPath)
            if verbose: print('Saved lc data to ',lcPath)

        ok = (table['EVENT'] == 0) | (table['EVENT'] == 100)
        bjd = np.array(table['BJD_TIME'][ok])
        bjd_ref = np.int(bjd[0])
        time = bjd-bjd_ref
        flux = table['FLUX'][ok]
        fluxerr = table['FLUXERR'][ok]
        fluxmed = np.median(flux)
        if reject_highpoints:
            C_cut = (2*np.median(flux)-np.min(flux))
            ok  = (flux < C_cut).nonzero()
            time = time[ok]
            flux = flux[ok]
            fluxerr = fluxerr[ok]
            N_cut = len(bjd) - len(time)
        if verbose:
            if reject_highpoints:
                print('C_cut = {:0.0f}'.format(C_cut))
                print('N(C > C_cut) = {}'.format(N_cut))
            print('Mean counts = {:0.1f}'.format(flux.mean()))
            print('Median counts = {:0.1f}'.format(fluxmed))
            print('RMS counts = {:0.1f} [{:0.0f} ppm]'.format(np.std(flux), 
                1e6*np.std(flux)/fluxmed))
            print('Median standard error = {:0.1f} [{:0.0f} ppm]'.format(
                fluxmed, 1e6*np.median(fluxerr)/fluxmed))

        flux = flux/fluxmed
        fluxerr = fluxerr/fluxmed
        self.lc = {'time':time, 'flux':flux, 'fluxerr':fluxerr,
                'bjd_ref':bjd_ref, 'table':table, 'header':hdr,
                'aperture':aperture}

        if returnTable:
            return table
        else:
            return time, flux, fluxerr

    def transit_noise_plot(self, width=3, steps=500,
            fname=None, figsize=(6,4), fontsize=10 , verbose=True):

        try:
            D = Table(self.lc['table'], masked=True)
        except AttributeError:
            raise AttributeError("Use get_lightcurve() to load data first.")
        time = self.lc['time']
        flux = self.lc['flux']
        flux_err = self.lc['fluxerr']
        T = np.linspace(np.min(time)+width/48,np.max(time)-width/48 , steps)

        Nsc = np.empty_like(T)
        Fsc = np.empty_like(T)
        Nmn = np.empty_like(T)

        for i,_t in enumerate(T):
            _n,_f = transit_noise(time, flux, flux_err, T_0=_t,
                              width=width, method='scaled')
            Nsc[i] = _n
            Fsc[i] = _f
            _n = transit_noise(time, flux, flux_err, T_0=_t,
                           width=width, method='minerr')
            Nmn[i] = _n

        if verbose:
            print('Scaled noise method')
            print('Mean noise = {:0.1f} ppm'.format(Nsc.mean()))
            print('Min. noise = {:0.1f} ppm'.format(Nsc.min()))
            print('Max. noise = {:0.1f} ppm'.format(Nsc.max()))
            print('Mean noise scaling factor = {:0.3f} '.format(Fsc.mean()))
            print('Min. noise scaling factor = {:0.3f} '.format(Fsc.min()))
            print('Max. noise scaling factor = {:0.3f} '.format(Fsc.max()))

            print('\nMinimum error noise method')
            print('Mean noise = {:0.1f} ppm'.format(Nmn.mean()))
            print('Min. noise = {:0.1f} ppm'.format(Nmn.min()))
            print('Max. noise = {:0.1f} ppm'.format(Nmn.max()))

        plt.rc('font', size=fontsize)    
        fig,ax=plt.subplots(2,1,figsize=figsize,sharex=True)

        ax[0].set_xlim(np.min(time),np.max(time))
        ax[0].plot(time, flux,'b.',ms=1)
        ax[0].set_ylabel("Flux ")
        ylo = np.min(flux) - 0.2*np.ptp(flux)
        ypl = np.max(flux) + 0.2*np.ptp(flux)
        yhi = np.max(flux) + 0.4*np.ptp(flux)
        ax[0].set_ylim(ylo, yhi)
        ax[0].errorbar(np.median(T),ypl,xerr=width/48,
               capsize=5,color='b',ecolor='b')
        ax[1].plot(T,Nsc,'b.',ms=1)
        ax[1].plot(T,Nmn,'g.',ms=1)
        ax[1].set_ylabel("Transit noise [ppm] ")
        ax[1].set_xlabel("Time");
        fig.tight_layout()
        if fname is None:
            plt.show()
        else:
            plt.savefig(fname)
        

    def diagnostic_plot(self, fname=None,
            figsize=(8,8), fontsize=10):
        try:
            D = Table(self.lc['table'], masked=True)
        except AttributeError:
            raise AttributeError("Use get_lightcurve() to load data first.")

        EventMask = (D['EVENT'] > 0) & (D['EVENT'] != 100)
        D['FLUX'].mask = EventMask
        D['FLUX_BAD'] = MaskedColumn(self.lc['table']['FLUX'], 
                mask = (EventMask == False))
        D['BACKGROUND'].mask = EventMask
        D['BACKGROUND_BAD'] = MaskedColumn(self.lc['table']['BACKGROUND'],
                mask = (EventMask == False))

        tjdb = D['BJD_TIME']
        flux = D['FLUX']
        flux_bad = D['FLUX_BAD']
        fluxerr = D['FLUXERR']
        back = D['BACKGROUND']
        back_bad = D['BACKGROUND_BAD']
        dark = D['DARK']
        contam = D['CONTA_LC']
        contam_err = D['CONTA_LC_ERR']
        rollangle = D['ROLL_ANGLE']
        xloc = D['LOCATION_X']
        yloc = D['LOCATION_Y']
        xcen = D['CENTROID_X']
        ycen = D['CENTROID_Y']

        plt.rc('font', size=fontsize)    
        fig, ax = plt.subplots(4,2,figsize=figsize)
        cgood = 'c'
        cbad = 'r'
        
        ax[0,0].scatter(tjdb,flux,s=2,c=cgood)
        ax[0,0].scatter(tjdb,flux_bad,s=2,c=cbad)
        ax[0,0].set_xlabel('BJD')
        ax[0,0].set_ylabel('Flux in ADU')
        
        ax[0,1].scatter(rollangle,flux,s=2,c=cgood)
        ax[0,1].scatter(rollangle,flux_bad,s=2,c=cbad)
        ax[0,1].set_xlabel('Roll angle in degrees')
        ax[0,1].set_ylabel('Flux in ADU')
        
        ax[1,0].scatter(tjdb,back,s=2,c=cgood)
        ax[1,0].scatter(tjdb,back_bad,s=2,c=cbad)
        ax[1,0].set_xlabel('BJD')
        ax[1,0].set_ylabel('Background in ADU')
        ax[1,0].set_ylim(0.9*np.quantile(back,0.005),1.1*np.quantile(back,0.995))
        
        ax[1,1].scatter(rollangle,back,s=2,c=cgood)
        ax[1,1].scatter(rollangle,back_bad,s=2,c=cbad)
        ax[1,1].set_xlabel('Roll angle in degrees')
        ax[1,1].set_ylabel('Background in ADU')
        ax[1,1].set_ylim(0.9*np.quantile(back,0.005),1.1*np.quantile(back,0.995))
        ax[2,0].scatter(xcen,flux,s=2,c=cgood)
        ax[2,0].scatter(xcen,flux_bad,s=2,c=cbad)
        ax[2,0].set_xlabel('Centroid x')
        ax[2,0].set_ylabel('Flux in ADU')
        
        ax[2,1].scatter(ycen,flux,s=2,c=cgood)
        ax[2,1].scatter(ycen,flux_bad,s=2,c=cbad)
        ax[2,1].set_xlabel('Centroid y')
        ax[2,1].set_ylabel('Flux in ADU')
        
        ax[3,0].scatter(contam,flux,s=2,c=cgood)
        ax[3,0].scatter(contam,flux_bad,s=2,c=cbad)
        
        ax[3,0].set_xlabel('Contamination estimate')
        ax[3,0].set_ylabel('Flux in ADU')
        ax[3,0].set_xlim(np.min(contam),np.max(contam))
        
        ax[3,1].scatter(rollangle,xcen,s=2,c=cgood)
        ax[3,1].scatter(rollangle,ycen,s=2,c=cbad)
        ax[3,1].set_xlabel('Roll angle in degrees')
        ax[3,1].set_ylabel('Centroid x (cyan), y (red)')

        fig.tight_layout()
        if fname is None:
            plt.show()
        else:
            plt.savefig(fname)

