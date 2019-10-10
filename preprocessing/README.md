#### N4 Bias Field correction implementation

Prerequsits: ensure doker is install on your local machine

`git clone https://github.com/jamesowler/final-year-project.git .`
`cd final-year-project/preprocessing`
`docker build -t python-n4:0.1 .`
`docker run --rm -u $(id -u):$(id -g) -v </path/to/image/directory>:/data python-n4:0.1 /data/<image-file-name.tif> -n4`