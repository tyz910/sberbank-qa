IMAGE=tyz910/sdsj2017

run:
	docker run --rm -it -v ${CURDIR}:/app -w /app -p 8000:8000 ${IMAGE} python3 prediction_server.py

run-jupyter:
	docker run --rm -it -v ${CURDIR}:/app -w /app -p 8888:8888 ${IMAGE} jupyter notebook --ip=0.0.0.0 --no-browser --allow-root  --NotebookApp.token='' --NotebookApp.password=''

docker-build:
	docker build -t ${IMAGE} . && (docker ps -q -f status=exited | xargs docker rm) && (docker images -qf dangling=true | xargs docker rmi) && docker images

docker-push:
	docker push ${IMAGE}
