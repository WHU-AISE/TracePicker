#!/bin/bash

# dataSets=("A" "B" "C" "media" "hipster-more" "sockshop" "trainticket"  "socialNetwork")
dataSets=("trainticket")
methods=("TracePicker")

dataDir=$(pwd)/data
outputDir=$(pwd)/output
logDir=$(pwd)/logs
baselineDir=$(pwd)/others
sampleRate=0.1

if [ ! -d "$logDir" ]; then
  mkdir "$logDir"
fi

echo "[Run baselines]"
for d in ${dataSets[*]}
do
  echo "============For dataSet $d============"
  for m in ${methods[*]}
  do
    echo "Execute method $m"
    if [ "$m" = "TracePicker" ]; then
      python "main.py" --dataDir "$dataDir" --dataSet "$d" --saveDir "$outputDir" --sampleRate $sampleRate > "$logDir/$d-$m.log" &
    else
      python "$baselineDir/$m/run.py" --dataDir "$dataDir" --dataSet "$d" --saveDir "$outputDir"  --sampleRate $sampleRate > "$logDir/$d-$m.log" &
    fi
  done
  wait
  echo "Evaluate baselines"
  python "$(pwd)"/eval.py --saveDir "$outputDir" --dataDir "$dataDir" --dataSet "$d" --methods ${methods[*]} --sampleRate $sampleRate
done