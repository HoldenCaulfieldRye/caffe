#!/usr/bin/env sh

cd ../..
./build/tools/caffe train \
    --solver=task/soil_high_blue/solver.prototxt \
    --snapshot=task/soil_high_blue/none/fine_iter_2000.solverstate
