#!/bin/bash

make clean
make qemu SCHEDULER=$1
