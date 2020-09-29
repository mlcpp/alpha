# alpha

## make commands 

##### *To be executed from project root*

1. Files under ./examples/

   1.  for a particular file

      `make <file-name>`

   2. for all the files

      `make all`

2. Files under ./examples/
   1.  for a particular file

      `make -f Makefile.ext <file-name>`

   2. for all the files

      `make -f Makefile.ext all`

The compiled binaries will be placed in ./build/

Execute them using `./build/<file-name>`

The build folder can be cleaning by using `make clean` for a clean rebuild e.g. `make clean && make all`