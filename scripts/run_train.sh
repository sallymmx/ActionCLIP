#!/usr/bin/env bash
if [ -f $1 ]; then
  config=$1
else
  echo "need a config file"
  exit
fi

type=$(python -c "import yaml;print(yaml.load(open('${config}'))['network']['type'])")
arch=$(python -c "import yaml;print(yaml.load(open('${config}'))['network']['arch'])")
dataset=$(python -c "import yaml;print(yaml.load(open('${config}'))['data']['dataset'])")
now=$(date +"%Y%m%d_%H%M%S")
mkdir -p exp/${type}/${arch}/${dataset}/${now}
python -u train.py  --config ${config} --log_time $now 2>&1|tee exp/${type}/${arch}/${dataset}/${now}/$now.log
# --mail-user=mengmengwang@zju.edu.cn --mail-type=ALL -x node86 