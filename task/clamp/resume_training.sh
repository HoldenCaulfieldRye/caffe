#!/usr/bin/env sh

cd ../..
./build/tools/caffe train \
    --solver=task/clamp/solver.prototxt \
    --snapshot=task/clamp/none/_iter_750.solverstate
