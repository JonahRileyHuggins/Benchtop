"""Load benchmark YAML and referenced PEtab TSV/SBML files."""

import json
import os
from types import SimpleNamespace
from typing import Union

import pandas as pd
import yaml


class FileLoader:
    """Parse benchmark YAML and load companion data files into memory."""

    def __init__(self, config_path: Union[str, os.PathLike]):
        self.config_path = config_path
        self.config = Config.file_loader(self.config_path)
        self.problems = []
        self.parameter_file = None

    def _petab_files(self) -> SimpleNamespace:
        """Load parameter TSV and per-problem condition/measurement/observable/SBML files."""
        yaml_dir = os.path.dirname(self.config_path)

        param_fp = os.path.join(yaml_dir, self.config.parameter_file)
        self.parameter_file = pd.read_csv(param_fp, sep="\t")

        for index, problem in enumerate(self.config.problems):
            p = SimpleNamespace()
            p.name = getattr(problem, "name", None) or f"problem_{index + 1}"
            p.cell_count = problem.cell_count

            for attr in (
                "condition_files",
                "measurement_files",
                "observable_files",
                "sbml_files",
                "visualization_df",
            ):
                file_list = getattr(problem, attr, None)
                if file_list is None:
                    continue

                loaded = []
                for rel in file_list:
                    fp = os.path.join(yaml_dir, rel)
                    ext = os.path.splitext(fp)[1].lower()

                    if ext in (".sbml", ".xml", ".bngl", ".net"):
                        loaded.append(fp)  # path only; simulator loads SBML
                    else:
                        loaded.append(pd.read_csv(fp, sep="\t"))

                setattr(p, attr, loaded)

            self.problems.append(p)

        del self.config_path

    def _extract_model_build_files(self) -> SimpleNamespace:
        """Load compilation input files as indexed DataFrames."""
        model_files = SimpleNamespace()
        yaml_dir = os.path.dirname(self.config_path)
        data_dir = os.path.join(yaml_dir, self.config.compilation.directory)

        for key, value in self.config.compilation.files.items():
            file_path = os.path.join(data_dir, value)
            setattr(
                model_files,
                key,
                pd.read_csv(file_path, sep="\t", index_col=0, header=0),
            )

        return model_files


class Config:
    """Dispatch file loading by extension."""

    @staticmethod
    def file_loader(file_path: os.PathLike, **kwargs):
        ext = os.path.splitext(file_path)[1].lower()

        loader_class = {
            ".yml": YAML,
            ".yaml": YAML,
            ".json": JSON,
            ".csv": CSV,
            ".tsv": CSV,
            ".txt": CSV,
        }.get(ext)

        if loader_class is None:
            raise ValueError(f"Unsupported file type: {ext}")

        file_instance = loader_class(file_path)
        try:
            return file_instance.loader(**kwargs)
        except TypeError:
            return file_instance.loader()


class File:
    def __init__(self, file_path: os.PathLike):
        self.file_path = file_path

    def loader(self):
        raise NotImplementedError("Subclasses must implement loader()")


class YAML(File):
    def loader(self):
        with open(self.file_path, encoding="utf-8") as file:
            return DotDict(yaml.safe_load(file))


class JSON(File):
    def loader(self):
        with open(self.file_path, encoding="utf-8") as file:
            return DotDict(json.load(file))


class CSV(File):
    def loader(self, **kwargs):
        kwargs.setdefault("sep", "\t")
        return pd.read_csv(filepath_or_buffer=self.file_path, engine="python", **kwargs)


class DotDict(dict):
    """Dict with attribute access for nested YAML/JSON config."""

    def __getattr__(self, attr):
        val = self.get(attr)
        if isinstance(val, dict):
            return DotDict(val)
        if isinstance(val, list):
            return [DotDict(x) if isinstance(x, dict) else x for x in val]
        return val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
