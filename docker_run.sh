set -x

PROJECT_PATH=$(pwd)
IMAGE_NAME="nvcr.io/nvidia/pytorch:23.12-py3"

docker run --runtime=nvidia -it --rm --shm-size="10g" --cap-add=SYS_ADMIN \
		-v $PROJECT_PATH:/RLHF -v  $HOME/.cache:/root/.cache -v  $HOME/.bash_history2:/root/.bash_history \
		$IMAGE_NAME bash