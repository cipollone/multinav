#!/bin/bash

if [[ $# -eq 0 ]] ; then
	echo 'Arguments: image-name container-name [option]'

else
	name=$2
	docker run -dit --rm \
		--user=$UID:`id -g` \
		-v /home/cipollor:/home/cipollor \
		--name=$2 \
		-p 52123:6006 \
		-p 52124:6007 \
		${@:3} $1

	sleep 3s
	docker exec -it $name bash -c "tmux attach -t main"
fi
