# DAISYcorrection

This code is provided as supporting material for the manuscript [Cabriel et al., 'Combining 3D single molecule localization strategies for reproducible bioimaging', bioRxiv 385799 (2018); doi: https://doi.org/10.1101/385799]. Please refer to the manuscript for more details about the content and the use of the code.

Python version: 2.7.10 (compatible with 32 bits and 64 bits)

The algorithm consists of several steps:
- First the data is filtered
- Then the optimal plane is fitted to the (y,x,Delta_z) (where Delta_z is the difference between the astigmatic and SAF axial positions) data to calculate the optimal coverslip plane. The data is thus corrected.
- Later, the axial drift is calculated by distributing the localizations in two temporally-separated series of 3D (y,x,z) matrices (one for the astigmatism, the other for the SAF). For each temporal slice, the two 3D matrices are cross-correlated allowing axial displacements only. This gives the axial correction to be applied to the astigmatic axial positions for the slice. Then the axial drift curve is obtained from the concatenation of all these correction values. The drift is subtracted from the axial astigmatic positions.
- Finally the output data is exported, as well as the correction parameters found.

Input format for the data:
Column-wise: y, x, z_SAF, z_astigmatism, number of the image (all values in nm except the number of the image)
Line-wise are all the localizations
Note: for optimal performance, make sure that pre-filtered data are used. In particular, saturated values should be removed if necessary. We also advise to eliminate unastigmatic PSFs having a width (i.e. the standard deviation of the spatial distribution) below 100 nm or above 300 nm. The datasets provided as test samples are pre-filtered.

The code is provided in the file "DAISY_correction.py".
The file "DAISY_data_test_COS7-alphatubulin-AF647_1.txt" is provided as a test sample (COS-7 cells, alpha-tubulin labelled with AF647). This is a localization list obtained from a COS-7 sample with alpha-tubulin labelled with Alexa Fluor 647.
Please download this file and place it in the same folder as the code to run it on the data.
Other data samples are provided in the "Datasets_DAISYcorrection.zip" file:
- "DAISY_data_test_COS7-alphatubulin-AF647_2.txt" -> COS-7 cells, alpha-tubulin labelled with AF647
- "DAISY_data_test_COS7-HCLCclathrin-AF647_1.txt" -> COS-7 cells, clathrin heavy chain and light chain labelled with AF647
- "DAISY_data_test_Ecoli-LPS-AF647_1.txt", "DAISY_data_test_Ecoli-LPS-AF647_2.txt" -> E. coli bacteria, lipopolysaccharide layer labelled with AF647
- "DAISY_data_test_Neurons-adducin-AF647_1.txt" -> rat hippocampal neurons, adducin labelled with AF647

The calculation can be parallelized on the different threads of the CPU to increase speed, but this might cause the script to crash on some computers (for unknown reasons). Please use the corresponding input option to set this parameter. Note that no GPU parallelization is implemented.
