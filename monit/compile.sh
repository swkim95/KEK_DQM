#! /bin/bash

# To compile TBxxx.cc, type the commnand below
# `bash compile.sh TBxxx.cc` (or `bash compile.sh TBxxx.cc` for short)
# This will generate TBxxx executable

ext=${1##*.}
fname=`basename ${1} .${ext}`

echo "Compiling $fname.cc to $fname"

# yaml-cpp path - try Homebrew first, then CVMFS
if [ -d "/opt/homebrew/opt/yaml-cpp" ]; then
  YAMLPATH=/opt/homebrew/opt/yaml-cpp
elif [ -d "/Users/Shared/cvmfs/sft.cern.ch/lcg/releases/yamlcpp/0.6.3-d05b2/arm64-mac15-clang170-opt" ]; then
  YAMLPATH=/Users/Shared/cvmfs/sft.cern.ch/lcg/releases/yamlcpp/0.6.3-d05b2/arm64-mac15-clang170-opt
else
  echo "Error: yaml-cpp not found!"
  exit 1
fi

g++ \
-I../install/include \
-I$YAMLPATH/include \
-L../install/lib \
-L$YAMLPATH/lib \
../install/lib/libdrcTB.dylib \
-lyaml-cpp \
`root-config --cflags --libs` \
${fname}.cc -o ${fname}
echo "Done!"