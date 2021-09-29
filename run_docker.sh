#!/bin/bash
#
# Docker build image, run container, execute last container.
#
# - Author: Jongkuk Lim
# - Contact: limjk@jmarple.ai

xhost +

if [ "$1" = "build" ]; then
    docker build . -t jmarpledev/ayolov2
elif [ "$1" = "run" ]; then
    docker run -tid --privileged --gpus all \
        -e DISPLAY=${DISPLAY} \
        -e TERM=xterm-256color \
        -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
        -v /dev:/dev \
        -v $PWD:/home/user/yolo \
        --network host \
        jmarpledev/ayolov2 /bin/bash

    last_cont_id=$(docker ps -qn 1)
    echo $(docker ps -qn 1) > $PWD/.last_exec_cont_id.txt

    docker exec -ti $last_cont_id /bin/bash
elif [ "$1" = "exec" ]; then
    last_cont_id=$(tail -1 $PWD/.last_exec_cont_id.txt)
    docker start ${last_cont_id}
    docker exec -ti ${last_cont_id} /bin/bash
else
    echo ""
    echo "============= $0 [Usages] ============"
    echo "1) $0 build - build docker image"
    echo "2) $0 run - launch a new docker container"
    echo "3) $0 exec - execute last container launched"
fi

