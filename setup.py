from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension("idrpred.predictors.smoothing",
              ["src/idrpred/predictors/smoothing.pyx"]),
    Extension("idrpred.predictors.disembl",
              ["src/idrpred/predictors/disembl.pyx"]),
    Extension("idrpred.predictors.iupred",
              ["src/idrpred/predictors/iupred.pyx"]),
    Extension("idrpred.predictors.seg",
              ["src/idrpred/predictors/seg.pyx"]),
    Extension("idrpred.predictors.espritz",
              ["src/idrpred/predictors/espritz.pyx"]),
]

setup(
    ext_modules=cythonize(extensions,
                          compiler_directives={"language_level": "3"}),
)
