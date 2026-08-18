In this example chain C of the GroEL/ES chaperonin system (PDB entry 3zpz) was fitted into the 
corresponding cryo-EM density map of the full complex (EMDB entry 2325 - 8.9 Å resolution) with
a rotational sampling interval of 5 degrees.

In the `input.zip` you can find the following files:
- `3zpz_C.pdb`
- `EMD-2325.map`
- `input_parameters.txt`

The command in `input_parameters.txt` is needed to generate the output in `output.zip`

Please refer to the [manual](https://www.bonvinlab.org/powerfit/manual.html#output) for an explanation
of all the files present in `output.zip`

You can visualize the fits by downloading the output files and opening the result page with
`python3 -m http.server -d .` and clicking `report.html`

While Powerfit clearly favors one location based on cross correlation score and sigma difference, 
there is still a major break in sigma difference between the 7 symetric orientations and the 
next best fit (Fit 8).