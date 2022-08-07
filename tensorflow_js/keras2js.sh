#! /bin/bash

input=$1
output=$2


tensorflowjs_converter --input_format keras $input $output