#!/usr/bin/env sh

cd ../..
./build/tools/caffe train \
    --solver=task/water_high_blue/solver.prototxt \
    --snapshot=task/water_high_blue/none/fine_iter_1000.solverstate
