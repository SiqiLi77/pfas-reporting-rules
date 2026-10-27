#!/bin/bash
c=$1
f=data/usdm/$c.json
[ -s "$f" ] && exit 0
for i in 1 2 3; do
  curl -sS -m 60 -H "Accept: application/json" "https://usdmdataservices.unl.edu/api/CountyStatistics/GetDSCI?aoi=$c&startdate=1/1/2023&enddate=6/30/2026&statisticsType=1" -o "$f" && [ -s "$f" ] && exit 0
  sleep $((i*3))
done
