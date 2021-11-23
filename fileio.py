from pathlib import Path

import numpy as np
import pandas as pd

## For manipulating file paths
STORM_filepaths = {'desmin_alphaactinin_600nm': '2020-08-25-12-24-36 sample2a/particles.dat'
, 'actin_desmin_600nm': '2017-03-21-11-31-56/particles.csv'
, 'actin_desmin_1.2um': '2020-07-28-12-00-44 sample1c desmin568 actin647/particles.dat'
, 'desmin_actin_2.5um': '2020-07-28-11-45-42 sample1b desmin568 actin647/particles.dat'}

def fetch_STORM_experiment_filepath(experiment_name, STORM_data_dir):
    assert experiment_name in STORM_filepaths.keys(), 'Experiment with specified name does not exist!'
    return Path(STORM_data_dir).joinpath(STORM_filepaths[experiment_name])

## For reading raw data files produced by Bruker Vutara352 microscope
def read_Vutara352_localization_data(filepath):
    path_obj = Path(filepath)
    file_ext = path_obj.suffix
    
    if file_ext == '.csv':
        localization_dataframe = _read_particles_csv(path_obj)
    elif file_ext == '.dat':
        localization_dataframe =  _read_particles_dat(path_obj)
    else:
        raise ValueError('Vutara output file is either in .dat or .csv format!')
    
    print(localization_dataframe.info())
    return localization_dataframe

def _read_particles_csv(path_csv):
    particles_dataframe = pd.read_csv(path_csv, header = 0)
    return particles_dataframe

def _read_particles_dat(path_dat):

    header_size = 572 # This was also reverse-engineered by manually decoding the particles.dat bytecode,
    # that is, by counting the number of bytes until the start of data was reached
    # Judging from Bruker's documentation, depending on what you do with Vutara before exporting(generate extra columns), this number could increase
    encoding_dtype = _get_particle_dat_encoding_datatype()
    
    with path_dat.open(mode = 'rb') as binary_file:
        header = binary_file.read(572)
        data = np.fromfile(binary_file, dtype = encoding_dtype)

    particles_dataframe = pd.DataFrame.from_records(data)
    return particles_dataframe

def _get_particle_dat_encoding_datatype():
    
    '''Returns the composite datatype that is used to encode the particles.dat files generated by newer versions of the Vutara software.
    This datatype was reversed-engineered through consulting Bruker's documentation for Vutara's file format, manually decoding the particles.dat bytecode
    , and comparing this to particles.csv produced by older versions of the Vutara software.'''
    
    integer_type = '<i4'
    float_type = '<f8'
    binary_type = '?'

    field_encoding_list = [('image-ID', integer_type), ('time-point', integer_type), ('cycle', integer_type), ('z-step', integer_type), ('frame', integer_type)
    , ('accum', integer_type), ('probe', integer_type), ('photon-count', float_type), ('photon-count11', float_type), ('photon-count12', float_type)
    , ('photon-count21', float_type), ('photon-count22', float_type), ('psfx', float_type), ('psfy', float_type), ('psfz', float_type)
    , ('psf-photon-count', float_type), ('x', float_type), ('y', float_type), ('z', float_type), ('stdev', float_type), ('amp', float_type)
    , ('background11', float_type), ('background12', float_type), ('background21', float_type), ('background22', float_type)
    , ('maxResidualSlope', float_type), ('chisq', float_type), ('log-likelihood', float_type), ('llr', float_type), ('accuracy', float_type)
    , ('precisionx', float_type), ('precisiony', float_type), ('precisionz', float_type), ('fiducial', binary_type), ('valid', binary_type)]

    return np.dtype(field_encoding_list)

## For preprocessing the dataframe produced by the read functions above
def drop_invalid_localizations(localization_dataframe):
    valid_inds = localization_dataframe['valid'] == True
    return localization_dataframe[valid_inds]

def get_unique_probe_ids(localization_dataframe):
    return np.sort(pd.unique(localization_dataframe['probe']))

def get_per_probe_localizations(localization_dataframe):
    probe_ids = get_unique_probe_ids(localization_dataframe)
    localization_dfs = [get_single_probe_localizations(localization_dataframe, id) for id in probe_ids]
    return probe_ids, localization_dfs

def get_single_probe_localizations(localization_dataframe, probe_id):
    selected_inds = (localization_dataframe['photon-count'] > 0) & (localization_dataframe['probe'] == probe_id)
    return localization_dataframe[selected_inds]