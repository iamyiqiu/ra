# Robo-advising: automated quantitative algorithms for risk profile estimation and portfolio optimization

This repository hosts the source code and experimental results for my honours year project on robo-advisors. The implementation mostly follows [Capponi et al (2022)](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.2021.4014) with a particular focus to close the gaps where the original paper ignores or oversimplifies, such as the numerical algorithm to calculate optimal investment strategy based on discretization of low-dimensional state tuples and estimation for client's personalized parameters.

The file structure of this project is as follows.
```
- ra
  - notebooks: Python notebooks for experiments and results demonstration
  - output: output directory to store saved objects and output figures
    -figures: stores output figures
  - src: source code directory, including a RoboAdvisor class and a ParamEstimator class
```

Feel free to cite my work via
```bibtex
@mastersthesis{liu2023ra,
    type={Bachelor's Thesis},
    author = {Yiqiu Liu},
    title = {Robo-advising: automated quantitative algorithms for risk profile estimation and portfolio optimization},
    publisher = {National University of Singapore Department of Mathematics},
    address = {Singapore},
    year = {2023}
}
```