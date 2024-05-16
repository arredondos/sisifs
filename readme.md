# Symmetrical Islands Site Frequency Spectrum (SISiFS)

This repository contains an implementation of the methods described in the paper: [Exact calculation of the expected SFS in structured populations](https://www.biorxiv.org/content/10.1101/2023.05.10.540112v1).

## Description

The Site Frequency Spectrum (SFS) is a summary statistic of the distribution of derived allele frequencies in a sample of DNA sequences. It provides information about genetic variation and can be used to make population inferences. The exact calculation of the expected SFS in a panmictic population under the infinite-site model of mutation has been known in the Markovian coalescent theory for decades, but its generalization to the structured coalescent is hampered by the almost exponential growth of the states space. 

The paper shows how to obtain this expected SFS as the solution of a linear system.
More precisely, it proposes a complete algorithmic procedure: building a suitable and ordered state space, taking advantage of the sparsity of the rate matrix, and solving numerically the linear system by an iterative method. 

The simplest case of the symmetrical n-island is showcased in this implementation.

## Quick start

Pre-compiled binaries are available for Windows and Linux in the [releases section](https://github.com/arredondos/sisifs/releases). Download your preferred version and run it from a terminal. 

*Linux:*
``` bash
$ ./SISiFS
```
*Windows:*
``` powershell
> .\SISiFS.exe
```
This invocation will run the program with default values for all the parameters, and you should see an output similar to the following:
```
reps = 1, e = 1e-06, d = 1, omega = 1.25, threads = 15, verbosity = 0.

k = 12, n = 10, M = 1, c = 1, sv = [12].
========================================
esfs = [0.210995, 0.129149, 0.098345, 0.0822614, 0.0727061, 0.0668195, 0.0634349, 0.062168, 0.063267, 0.0681307, 0.0827243]
```

## Compiling from source

If you want to compile the program yourself, you'll need the following prerequisites:

### Dependencies
*Linux:*
 * **LDC - The LLVM-based D Language Compiler.** Usually the following command will install the latest version: 
 
    `curl -fsS https://dlang.org/install.sh | bash -s ldc`.

    If you encounter issues, or want more information, please visit the [official downloads page](https://dlang.org/download.html).

 * **OpenBLAS.** On Ubuntu, the following command should work:

    `sudo apt-get install libopenblas-dev`.

*Windows:*
 * **LDC - The LLVM-based D Language Compiler.** Precompiled binaries for Windows can be downloaded from the [LDC GitHub releases page](https://github.com/ldc-developers/ldc/releases) (you will have to manually add the `/bin` directory to you PATH environment variable). 
 
    For more options/information, please visit the [official downloads page](https://dlang.org/download.html).

 * **Intel Math Kernel Library.** You can get this by installing the [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#base-kit). Please make a note of the location where the libraries were installed, as you may need to pass that as a linker flag (details in the next session).

### Building

The files `build_ldc_debug.cmd` and `build_ldc_release.cmd` contain the build commands used for  generating debug and release binaries, respectively. The commands are platform independent.

**Note:**  If you get a linker error on Windows (for example `LINK : fatal error LNK1181: cannot open input file 'mkl_core.lib'`) it's likely that  the linker didn't find the location where the MKL libraries were installed. To resolve  this, please edit the `dub.json` file, and specify the location of the libraries in the `lflags-ldc` list as a doubly-quoted string preceded by `-L` with escaped backslashes. For example: `"lflags-ldc": ["-LC:\\Program Files (x86)\\Intel\\oneAPI\\mkl\\2024.1\\lib"]`.

## Usage

The program receives the specification of an n-island model (number of islands with `-n`, migration rate with `-M`, and deme size with `-c`) and a sampling configuration (total number of lineages with `-k` and where they are located in the islands with `--sv`) and returns the (normalized) expected SFS under that model. 

### Example
The following command computes the expected SFS of a sample of 8 diploid organisms (16 haploids) under a stationary n-island model with 30 islands and a migration rate of 2.25, where initially 2 of the organisms were in one of the islands and the other 6 were in another. 
``` bash
$ ./SISiFS -n 30 -M 2.25 -k 16 --sv [4,12]
```
From the output we can see that the expected SFS for that scenario is:
```
esfs = [0.207535, 0.134109, 0.102504, 0.0832587, 0.0658089, 0.057087, 0.0509493, 0.0462981, 0.0426491, 0.0397682, 0.0376025, 0.0367528, 0.0329779, 0.0311733, 0.0315262]
```

### Notes
- The flags `-k` and `--sv` (sampling vector) will generally have to be set simultaneously, since all program flags that are not set will take their default values. The sampling vector does not have to be as long as the number of islands, since it will be padded by zeros. In fact, in the previous example, the specified sampling vector was [4, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], since there were 30 demes. Do not insert spaces when specifying the sampling vector with the `--sv` flag.

- Additional options are available that control the numerical behavior  (`-r`, `-e`, `-d`, `-omega` and `-threads`). Please consult the paper to learn more about them. 

- The `-v` flag is useful for benchmarking purposes since it reports time spent at the various computation steps.

- During  the usage of the program, one `.qmat` file will be generated for every unique combination of `-n` and `-k` values that the program is invoked with. These are cached symbolic representations of the rate matrices that are expensive to compute for large values of k. The files can be deleted, but then they will have to be re-generated when needed.

### Advanced usage

Any of the flags `-n`, `-M`, `-c` or `-k` can receive multiple values in the form of a vector. When multiple values are passed, the program computes one expected SFS for every combination of values of these 4 parameters. For example, the command:
```
./SISiFS -n [2,50,100] -M [0.5,1,10] -c [1,10]
```
results in 3 * 3 * 2 = 18 computations of an expected SFS.


Specifying multiple values for the number of samples `-k` is incompatible with exactly specifying the sampling vector `--sv`. However, in order to support this usecase to some extent, when multiple values are specified for `-k`, the vector passed to the `--sv` flag will be interpreted as a probability distribution of the samples over the demes. For example, the command:
```
./SISiFS -n 20 -k [4,8,12] --sv [0.5,0.5]
```
will result in three computations of an expected SFS, where the sampling vectors will be [2, 2], [4, 4] and [6, 6].

**Known Issue:** The program currently does not enforce that all possible combinations of sampling vectors and number of demes make sense, so please keep this in mind when specifying multiple parameters. For example, the command `./SISiFS -n 2 -k [4,8,12] --sv [0.25,0.25,0.25,0.25]` will result in nonsensical results for all three expected SFS computations, and other similar cases may result in a segmentation fault.

### Reference
Following is a brief description of all the supported flags and their default values:
``` 
 --help, -h      Prints this help.
 --samples, -k <string>
                 Total number of haploid lineages in the sample. May be vectorized. Default is 12.
 --demes, -n <string>
                 Number of demes or islands in the model. May be vectorized. Default is 10.
 --migration, -M <string>
                 Migration rate M of an n-island model. May be vectorized. Default is 1.0.
 --size, -c <string>
                 Relative deme size c in an n-island model. May be vectorized. Default is 1.0.
 --sv <string>   Sampling vector or initial distribution of sampled lineages across the demes.
                 Default is [12].
 --repetitions, -r <string>
                 Number of times each SFS computation is performed. Default is
                 1.
 --epsilon, -e <string>
                 Maximum absolute error tolerance for the normalized expected
                 SFS. Default is 1e-6.
 --steps, -d <string>
                 Approximate number of error introspection steps during
                 computation. Default is 1.
 --omega <string>
                 Relaxation parameter for the SOR method. Default is 1.25.
 --threads <string>
                 Number of additional execution threads in parallel workloads.
                 Default is one less than the total system threads as reported
                 by the OS.
 --verbosity, -v <int>
                 Controls the level of reporting detail. The three possible
                 values are 0, 1 and 2. Default is 0
```