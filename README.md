# boid-model-visualization-matplotlib
![demo](https://github.com/estshorter/boid-model-visualization-matplotlib/raw/master/movie.gif)

A boid model is visualized by using matplotlib as shown above.

`boid.py` and part of `model.py` are from [here](https://github.com/projectmesa/mesa/tree/master/examples/boid_flockers/boid_flockers).

## Requirement
- matplotlib
- mesa
- numpy
- toml
- tqdm
- ffmpeg (when generating movie)

## How to use
```
python src/run.py
```

If you want to create a movie file, execute `python src/generate_movie`.
Note that `ffmpeg` is required.
