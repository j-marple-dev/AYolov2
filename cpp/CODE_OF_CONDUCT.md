# 1. Style guide
- We follow [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) guideline.

# 2. Code check tool
## Environment setup
- `clang-format` v6.0.0
- `cpplint` v1.5.5
- `cppcheck` v1.8.2
- `doxygen` v1.9.2
- `cmake` >= 3.20.5
### Install CMake >= 3.20.5
* Download CMake from https://cmake.org/download/ for the suitable platform
``` shell
# ex) Current Make version is 3.10.2
cd ~
wget https://github.com/Kitware/CMake/releases/download/v3.21.0/cmake-3.21.0-linux-x86_64.tar.gz
tar -xzvf cmake-3.21.0-linux-x86_64.tar.gz

# Backup previous CMake just in case
sudo mv /usr/bin/cmake /usr/bin/cmake.3.10.2
sudo mv /usr/bin/ctest /usr/bin/ctest.3.10.2
sudo mv /usr/bin/cpack /usr/bin/cpack.3.10.2

# Make softlink to /usr/bin
sudo ln -s ${HOME}/cmake-3.21.0-linux-x86_64/bin/cmake /usr/bin/cmake
sudo ln -s ${HOME}/cmake-3.21.0-linux-x86_64/bin/ctest /usr/bin/ctest
sudo ln -s ${HOME}/cmake-3.21.0-linux-x86_64/bin/cpack /usr/bin/cpack
```


## 2.1. Formating
```shell
./run_check.sh format
```

## 2.2. Linting
```shell
./run_check.sh lint
```

## 2.3. Documentation
```shell
./run_check doc_check
```

## 2.4. Formating, Linting, and documentation checking all at once
```shell
./run_check.sh all
```

# 3. Unit testing
```shell
mkdir build
cd build
cmake .. && make && make test
```

# 4. Commit
* **DO NOT** commit on `main` branch. Make a new branch and commit and create a PR request.
* Formatting and linting is auto-called when you `commit` and `push` but we advise you to run `./run_check all` occasionally.

# 5. Documentation
## Install Doxygen
```shell
git clone -b Release_1_9_2 https://github.com/doxygen/doxygen.git
cd doxygen
mkdir build
cd build
cmake -G "Unix Makefiles" .. 
make
make install
```

## 5.1. Generate API document
```shell
doxygen
```

## 5.2. Run local documentation web server
```shell
cd html
python -m http.server
```
