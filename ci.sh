#!/usr/bin/env bash
function run {
  # Update pylec <-- CHANGE NAME
  $SSH git -C ./pylec_forecast_paper pull origin master

  # Run simulation <-- CHANGE NAME
  $SSH docker run -t -d --rm --name $1 --mount type=bind,source=/home/jon/pylec_forecast_paper,target=/usr/src/app python_box_jon $1 $2 $EMAIL
}

function getall {
  $SSH docker ps | grep jonathan | awk '{print $0}'
}

function status {
  $SSH docker logs --tail 100 $1
}

function stop {
  $SSH docker stop $1
}

source .env
case "$1" in
    run) run $2 $3;;
    getall) getall;;
    status) status $2;;
    stop) stop $2;;
esac
