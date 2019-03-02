
This folder contains examples of input files for use with make_xml_files, e.g., 
 % make_xml_files NonTimeCriticalOnly.csv


NonTimeCriticalOnly.csv
 The terminal output shows some examples the Flags column being used to
 indicate errors/warnings, e.g., The flag for HD80607_2hr is 20480, which is
 the sum of the two following flags
  + 16384 = Acquisition error, brighter star within 51"
  +  4096 = Contamination error, Contam > 1
 Use "% make_xml_files -h" to see a full list of flag values.

TimeCriticalOnly_NoNRanges.csv

TimeCriticalOnly_NRanges.csv

All.csv
  Contains all the entries in the 3 previous files and shows how to combine a
  mixture of time-critical and non-time-critical observation requests. 
  Note that if you have previously run make_xml_files on one of the previous 
  files you will either need to delete the exisitng .xml files or use the -f
  flag to overwrite them.

NoGaiaID.csv
  Same as NonTimeCriticalOnly.csv but with no column Gaia_DR2. Use the flag 
  --ignore-gaia-id-check to automatically insert the Gaia DR2 ID for each
  target, i.e., 
  % make_xml_files --ignore-gaia-id-check NoGaiaID.csv
  ** The PI is responsible to check the DR2 ID is correct ** 

NoTexp.csv
  Same as NonTimeCriticalOnly.csv but with no column Texp. Use the flag 
  --auto-expose (-a for short) to automatically insert the exposure time.
  % make_xml_files -a NoTexp.csv
  ** The PI is responsible to check the DR2 ID is correct ** 

WrongGaiaID.csv
 An example of catching a mismatch between the user's input values of Gaia_DR2
 and the value from Gaia DR2 catalogue. This file can either be run without
 the flag --ignore-gaia-id-check, in which case an error message is printed,
 or with the flag, in which case the mismatch can be detected from the value of 
 "Flags" in the terminal output (see make_xml_files -h for further
 information).


