# DOCKER INSTALLATION README

1. Install docker
2. Using the 'Dockerfile' provided by this repo, create a docker container permanently running in background, which you can bash into. To do so, run 
```python
docker build . -t "uitb-sim2vr"
docker images
docker run -it --gpus all -d --name uitb uitb-sim2vr /bin/bash
```
3. Ensure that the docker container is running in background by calling
```python
docker ps -a
```
4. To bash into the running docker container, run:
```python
docker exec -it uitb /bin/bash
```

