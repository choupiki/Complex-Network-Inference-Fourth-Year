#!/bin/bash
while getopts ":i:e" opt; do
    case $opt in
    i)
      echo "-i install: $OPTARG" >&2
      if [ "$OPTARG" == "editable" ]; then
        pip install -e .
      elif [ "$OPTARG" == "fixed" ]; then
        pip install .
      else
        echo "Invalid install option chose. Valid choices: editable or fixed"
      fi
      
      ;;
    e)
      echo "-e: Copying executables to ~/bin" >&2
      mkdir -p ~/bin
      chmod u+x executables/*
      cp executables/* ~/bin/.
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done


