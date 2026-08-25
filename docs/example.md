# Powerfit Example

This example contains a sample input and ouput generated with PowerFit release version 5.0.2.
The data is hosted on the [powerfit-examples github repository](https://github.com/haddocking/powerfit-examples/).

## Running the example

First download the input files and install PowerFit:

```shell
# Downloads the input files
mkdir powerfit-example-data/
cd powerfit-example-data
curl -L -o 3zpz_C.cif.gz https://github.com/haddocking/powerfit-examples/raw/refs/heads/main/3zpz_C.cif.gz
curl -L -o EMD-2325.map.gz https://github.com/haddocking/powerfit-examples/raw/refs/heads/main/EMD-2325.map.gz
# Create an environment, on Windows use WSL
python3 -m venv .venv
.venv/bin/activate # or activate equivalent for your OS/shell
# Install PowerFit release version 5.0.2
pip install powerfit-em==5.0.2
```

In this example chain C of the GroEL/ES chaperonin system (PDB entry 3zpz) was fitted into the 
corresponding cryo-EM density map of the full complex (EMDB entry 2325 - 8.9 Å resolution) with
a rotational sampling interval of 5 degrees.

In the repository you find the following files:
- [3zpz_C.cif.gz](https://github.com/haddocking/powerfit-examples/raw/refs/heads/main/3zpz_C.cif.gz)
- [EMD-2325.map.gz](https://github.com/haddocking/powerfit-examples/raw/refs/heads/main/EMD-2325.map.gz)
- [output.zip](https://github.com/haddocking/powerfit-examples/raw/refs/heads/main/output.zip)

The following command can be used to generate the output present in `output.zip`:

```shell
# Run the example
powerfit EMD-2325.map.gz 8.9 3zpz_C.cif.gz --angle 5 --directory output --report --delimiter , 
```

Dependent on your system and the amount of CPUs used to run the example, this run might take ~30 minutes

If you want to quickly check the results of the run, you can directly downloads `output.zip`
```shell
# Download ouput.zip
curl -L -o output.zip https://github.com/haddocking/powerfit-examples/raw/refs/heads/main/output.zip
```

Please refer to the [manual](https://www.bonvinlab.org/powerfit/manual.html#output) for an explanation
of all the files present in `output.zip`

You can visualize the fits by downloading the output files and opening the result page with
`python3 -m http.server -d .` and clicking `report.html`

While Powerfit clearly favors one location based on cross correlation score and sigma difference, 
there is still a major break in sigma difference between the 7 symetric orientations and the 
next best fit (Fit 8).
