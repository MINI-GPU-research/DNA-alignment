# DNA-alignment
Long nanopore DNA sequence alignment

### Building
To build the project run command ***make all*** in the *Debug* folder. Everything should compile succesfully and you should get a ***nanopore*** executive file in *Debug* folder.

### Usage
Simplest version:

nanopore \[path to file to process\]

If you want to catche processed file to RAM add third argument - it can be any string

nanopore \[path to file to process\] \[if you want to use RAM cache\]

*Disclaimer: This option lets program run much faster as it needs to read file from disk only once, but it uses humunguous amounts of RAM, it is best to have 2 times more ram than the size of processed file.*

### Output
As an output program prints weights of existing edges in DBG
