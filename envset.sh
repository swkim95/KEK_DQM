#! /bin/bash

# When using CVMFS (no X11)
# source /cvmfs/sft.cern.ch/lcg/views/LCG_102/arm64-mac12-clang131-opt/setup.sh

# When using local ROOT (X11 activated)
source /Users/yhep/root-6.36.00/install/bin/thisroot.sh
source /Users/yhep/AI/ai/bin/activate

export INSTALL_DIR_PATH=$PWD/install

export PATH=$PATH:$INSTALL_DIR_PATH/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$INSTALL_DIR_PATH/lib
export DYLD_LIBRARY_PATH=$INSTALL_DIR_PATH/lib
export PYTHONPATH=$PYTHONPATH:$INSTALL_DIR_PATH/lib

# Set YAMLPATH - try Homebrew first, then CVMFS
if [ -d "/opt/homebrew/opt/yaml-cpp" ]; then
  export YAMLPATH=/opt/homebrew/opt/yaml-cpp/lib
elif [ -d "/Users/Shared/cvmfs/sft.cern.ch/lcg/releases/yamlcpp/0.6.3-d05b2/arm64-mac15-clang170-opt" ]; then
  export YAMLPATH=/Users/Shared/cvmfs/sft.cern.ch/lcg/releases/yamlcpp/0.6.3-d05b2/arm64-mac15-clang170-opt/lib
fi

export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$YAMLPATH
