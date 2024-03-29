<h1 align="center">
  <b>Multinav</b>
</h1>

<p align="center">
  <a href="https://pypi.org/project/multinav">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/multinav">
  </a>
  <a href="https://pypi.org/project/multinav">
    <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/multinav" />
  </a>
  <a href="">
    <img alt="PyPI - Status" src="https://img.shields.io/pypi/status/multinav" />
  </a>
  <a href="">
    <img alt="PyPI - Implementation" src="https://img.shields.io/pypi/implementation/multinav">
  </a>
  <a href="">
    <img alt="PyPI - Wheel" src="https://img.shields.io/pypi/wheel/multinav">
  </a>
  <a href="https://github.com/marcofavorito/multinav/blob/master/LICENSE">
    <img alt="GitHub" src="https://img.shields.io/github/license/marcofavorito/multinav">
  </a>
</p>
<p align="center">
  <a href="">
    <img alt="test" src="https://github.com/marcofavorito/multinav/workflows/test/badge.svg">
  </a>
  <a href="">
    <img alt="lint" src="https://github.com/marcofavorito/multinav/workflows/lint/badge.svg">
  </a>
  <a href="">
    <img alt="docs" src="https://github.com/marcofavorito/multinav/workflows/docs/badge.svg">
  </a>
  <a href="https://codecov.io/gh/marcofavorito/multinav">
    <img alt="codecov" src="https://codecov.io/gh/marcofavorito/multinav/branch/master/graph/badge.svg?token=FG3ATGP5P5">
  </a>
</p>
<p align="center">
  <a href="https://img.shields.io/badge/flake8-checked-blueviolet">
    <img alt="" src="https://img.shields.io/badge/flake8-checked-blueviolet">
  </a>
  <a href="https://img.shields.io/badge/mypy-checked-blue">
    <img alt="" src="https://img.shields.io/badge/mypy-checked-blue">
  </a>
  <a href="https://img.shields.io/badge/code%20style-black-black">
    <img alt="black" src="https://img.shields.io/badge/code%20style-black-black" />
  </a>
  <a href="https://www.mkdocs.org/">
    <img alt="" src="https://img.shields.io/badge/docs-mkdocs-9cf">
  </a>
</p>


## Preliminaries

System dependencies
```
sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev libgl1-mesa-glx graphviz
```

### Normal install

This package can be installed normally with `pip install <src>[tf]`, where `<src>` can be this cloned repository or [https://github.com/cipollone/multinav.git](https://github.com/cipollone/multinav.git).
If you omit the extra `[tf]`, TensorFlow is not installed as a dependency. This might be useful if you have a compatible Tensorflow installation already.

### Development mode install.

- Clone this repo with:
```
git clone --recurse-submodules https://github.com/cipollone/multinav
```

- Install Poetry, if you don't have it already:
```
pip install poetry
```

- Set up the virtual environment. 

```
poetry install -E tf
```
or omit the extra `-E ..` if you have a compatible Tensorflow already.

- Now you can run parts of this software:
```
poetry run python -m multinav train ...
```

- Note: each time you're working with local dependencies, and you modify them, install the new versions with the script `./scripts/update-local-dependencies.sh`.


## Tests

To run tests: `tox`

To run only the code tests: `tox -e py3.7`

To run only the linters: 
- `tox -e flake8`
- `tox -e mypy`
- `tox -e black-check`
- `tox -e isort-check`

Please look at the `tox.ini` file for the full list of supported commands. 

## Docs

To build the docs: `mkdocs build`

To view documentation in a browser: `mkdocs serve`
and then go to [http://localhost:8000](http://localhost:8000)

## License

TBD

## Authors

- [Roberto Cipollone](https://github.com/cipollone)
- [Marco Favorito](https://marcofavorito.me/)
