Test app for file processing
============================

try adding .mp3 files to a ./input/ directory


Usage
-----

It's easiest to run using Docker:

```shell
docker build -t file-tests . 

docker run --rm -v "$(pwd)/input:/app/input" -v "$(pwd)/output:/app/output" file-tests
```