#!/bin/bash
if [ -d "docker_dir" ]; then
  rm -rf docker_dir
fi
mkdir docker_dir
cd docker_dir

download_url="https://www.docker.com/products/docker-desktop/" # Path to download docker desktop
output_file="docker-desktop.dmg" # Specify the output file name
curl -o "$output_file" "$download_url" # Download Docker Desktop
if [ $? -eq 0 ]; then
  echo "Docker Desktop downloaded successfully."
else
  echo "Failed to download Docker Desktop."
  exit 1
fi
echo "Installing Docker Desktop..."
hdiutil attach "$output_file"
cp -R /Volumes/Docker\ Desktop/Docker\ Desktop.app /Applications/
hdiutil detach /Volumes/Docker\ Desktop/
echo "Docker Desktop installation completed."
echo "Running Docker Desktop..."
open -a Docker
echo "Docker Desktop is now running."
git clone -b apm https://git.soma.salesforce.com/Business-Data-Science/gsi-predictive-enablement.git
cd gsi-predictive-enablement/src/sql/production/apm-as-a-service/docker

docker build -f "m1-m2-m4.Dockerfile" -t apmaas-m1-m2-m4 .
if [ $? -ne 0 ]; then
  echo "Docker build failed. Exiting."
  exit 1
fi
port=3002
while lsof -i :$port >/dev/null 2>&1; do
  ((port++))
done
docker stop apmaas-m1-m2-m4 >/dev/null 2>&1
docker rm apmaas-m1-m2-m4 >/dev/null 2>&1
current_dir=$(pwd)
parent_dir=$(dirname "$current_dir")
new_path="$parent_dir/apm_code"
config_file_path=${parent_dir%/*docker_dir*}
config_file_path_="$config_file_path/config.py"
cp "$config_file_path_" "$new_path/"

docker run -d --rm \
  -p $port:3001 \
  -e PATH_APMAAS_ROOT=/home/jovyan/work/projects/gsi-predictive-enablement/src/sql/production/apm-as-a-service \
  -e PATH_PE_PYTHON=/home/jovyan/work/projects/gsi-predictive-enablement/src/python-modules \
  -v "$new_path":"/home/jovyan/work" \
  --name apmaas-m1-m2-m4 apmaas-m1-m2-m4

if [ $? -ne 0 ]; then
  echo "Docker run failed. Exiting."
  exit 
  1
fi

docker exec -it apmaas-m1-m2-m4 bash -c  'rm -rf /home/jovyan/work/artifacts'
docker exec -it apmaas-m1-m2-m4 bash -c 'cd /home/jovyan/work && pip install -r requirements.txt'
docker exec -it apmaas-m1-m2-m4 bash -c 'cd /home/jovyan/work && python data_ingestion.py'
docker build -f "m3.Dockerfile" -t apmaas-m3 .
if [ $? -ne 0 ]; then
  echo "Docker build failed. Exiting."
  exit 1
fi
get_available_port() {
  local port=$1
  while nc -z 127.0.0.1 "$port"; do
    ((port++))
  done
  echo "$port"
}
app_port=$(get_available_port 8787)
docker run -d --rm -p 127.0.0.1:"$app_port":8787 \
  -e PATH_APMAAS_ROOT=/home/rstudio/mac/projects/gsi-predictive-enablement/src/sql/production/apm-as-a-service \
  -v $new_path:/home/rstudio/mac \
  -e DISABLE_AUTH=true -e ROOT=TRUE -e PASSWORD=p@ssw0rd \
  --name apmaas-m3 \
  apmaas-m3
container_id=$(docker ps -q -f name=apmaas-m3)
if [ -z "$container_id" ]; then
  echo "Docker container failed to start. Exiting."
  exit 1
fi
docker exec -it apmaas-m3 Rscript /home/rstudio/mac/module_3.r
docker stop apmaas-m3 >/dev/null 2>&1
docker rm apmaas-m3 >/dev/null 2>&1
docker rmi apmaas-m3 >/dev/null 2>&1
docker exec -it apmaas-m1-m2-m4 bash -c 'cd /home/jovyan/work && python module_4.py'
docker stop apmaas-m1-m2-m4 >/dev/null 2>&1
docker rm apmaas-m1-m2-m4 >/dev/null 2>&1
docker rmi apmaas-m1-m2-m4 >/dev/null 2>&1

current_dir=$(pwd)
parent_dir=$(dirname "$current_dir")
new_path="$parent_dir/apm_code"
config_file_path=${parent_dir%/*docker_dir*}

cp -r "$new_path/artifacts" "$config_file_path/"
rm -rvf "$config_file_path/docker_dir"
