from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Compiler import Options
Options.language_level=3

extensions = [
    Extension(name='utils.build_path',  # using dots!
              sources=['utils/build_path.py']),
    Extension(name='utils.artifact_manager',  # using dots!
              sources=['utils/artifact_manager.py']),
    Extension(name='utils.misc',  # using dots!
              sources=['utils/misc.py']),
    Extension(name='utils.cass_configured',  # using dots!
              sources=['utils/cass_configured.py']),
    Extension(name='constants.ui_constants',  # using dots!
              sources=['constants/ui_constants.py']),
    Extension(name='constants.app_constants',  # using dots!
              sources=['constants/app_constants.py']),
    Extension(name='constants.game_constants',  # using dots!
              sources=['constants/game_constants.py']),
    Extension(name='train_model.model',  # using dots!
              sources=['train_model/model.py']),
    Extension(name='train_model.network',  # using dots!
              sources=['train_model/network.py']),
    Extension(name='train_model.input_vector',  # using dots!
              sources=['train_model/input_vector.py']),
    Extension(name='utils.heavy_imports',  # using dots!
              sources=['utils/heavy_imports.py']),
    Extension(name='main',  # using dots!
              sources=['main.py'])
]

setup(
  ext_modules = cythonize(extensions, build_dir="tmp_build/tmp", compiler_directives={'language_level' : "3"})
)