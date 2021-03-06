#!/usr/bin/env bash
function run {
  # Update pylec <-- CHANGE NAME
  $SSH git -C ./fppylec pull origin master

  # Run simulation <-- CHANGE NAME and Image !
  $SSH docker run -t -d --rm --name $1 --mount type=bind,source=/home/jon/fppylec,target=/usr/src/app python_box2_jonathan $1 $2 $EMAIL
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
