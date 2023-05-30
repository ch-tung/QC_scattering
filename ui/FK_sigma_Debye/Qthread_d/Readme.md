Please refer to the provided GitHub link to access the source code.
https://github.com/ch-tung/QC_scattering/tree/main/ui/FK_sigma_Debye/Qthread_d

Please refer to the provided OSF link to access the packed executables.
https://osf.io/7qkt4/?view_only=f0e66d958c3a40729ac465c42d3dc27f


## Description

This project calculates and evaluates the scattering function of FK sigma phase based on user inputs. The scattering function is computed using two-point correlation and radial distribution function (RDF) calculations. The resulting scattering function is then plotted and saved as an output.

## Usage

1. Run FK_sigma_Debye.exe.

2. The application window will appear with input fields and buttons.

3. Fill in the required input parameters:
   - `r_ca`: c/a ratio of the material
   - `nx`, `ny`, `nz`: dimensions of the material
   - `n_bins`: number of bins for the radial distribution function
   - `filepath`: path to save the output files
   - `d_c`: interplanar spacing value

4. Click the "Start" button to initiate the calculation and evaluation of the scattering function.

5. Check the desired options in the UI to display specific scattering peaks.

6. The progress bar will show the progress of the calculation.

7. Once the calculation is complete, the scattering function will be plotted on the canvas.

8. Click the "Save Plot" button to save the plotted figure.