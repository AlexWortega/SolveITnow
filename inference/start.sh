
#!/bin/bash
#start
model=alexw/Experements/SolveITnow/results-Instrucr/checkpoint-1500
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus 0 --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:1.0.3 --model-id $model