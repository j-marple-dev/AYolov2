# 1. Style guide
- We follow conventional [PEP8](https://www.python.org/dev/peps/pep-0008/) guideline.

# 2. Code check tool
## 2.1. Formating
```shell
./run_check.sh format
```

## 2.2. Linting
```shell
./run_check.sh lint
```

## 2.3. Unit test
```shell
./run_check.sh test
```

## 2.4. Formating, Linting, and unit testing all at once
```shell
./run_check.sh all
```

# 3. Commit
* **DO NOT** commit on `main` branch. Make a new branch and commit and create a PR request.
* Formatting and linting is auto-called when you `commit` and `push` but we advise you to run `./run_check all` occasionally.

# 4. Documentation
## 4.1. Generate API document
```shell
./run_check doc
```

## 4.2. Run local documentation web server
```shell
./run_check doc-server
```
