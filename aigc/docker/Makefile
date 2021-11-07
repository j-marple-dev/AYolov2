container_name=agc2021
image_name=agc2021
volume=${CURDIR}

build-docker:
	cd ..;docker build -f docker/Dockerfile -t $(image_name) . --no-cache

run-docker:
	docker run -i -t -d -shm-size=8G --init --gpus all --name $(container_name) $(image_name)

exec-docker:
	docker exec -it $(container_name) /bin/bash

rm-docker:
	docker stop $(container_name) && docker rm $(container_name)
