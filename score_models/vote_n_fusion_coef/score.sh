#!/bin/bash
#victor 2019.1.12

mutation=$1


mkdir -p $(dirname $0)/temp
mkdir -p $(dirname $0)/submission_file


python $(dirname $0)/score.py $mutation || exit 1


rm -rf $(dirname $0)/temp #清空temp文件夹