#!/bin/zsh
cd "${0:A:h}/.."
echo "Watching shape-retrieval training. Press Ctrl-C to stop watching; training will continue."
touch results/shape_retrieval/training.log
tail -f results/shape_retrieval/training.log
