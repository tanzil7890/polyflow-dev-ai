from typing import Any, Callable

import pandas as pd

import polyflow
from polyflow.models import LanguageProcessor
from polyflow.templates import task_instructions
from polyflow.types import LMOutput, SemanticExtractOutput, SemanticExtractPostprocessOutput
from polyflow.utils import show_safe_mode

from .postprocessors import extract_postprocess


def sem_extract(
    docs: list[dict[str, Any]],
    model: LanguageProcessor,
    output_cols: dict[str, str | None],
    extract_quotes: bool = False,
    postprocessor: Callable[[list[str]], SemanticExtractPostprocessOutput] = extract_postprocess,
    safe_mode: bool = False,
    progress_bar_desc: str = "Extracting",
) -> SemanticExtractOutput:
    """
    Extracts attributes and values from a list of documents using a model.

    Args:
        docs (list[dict[str, Any]]): The list of documents to extract from.
        model (polyflow.models.LanguageProcessor): The model to use.
        output_cols (dict[str, str | None]): A mapping from desired output column names to optional descriptions.
        extract_quotes (bool): Whether to extract quotes for the output columns. Defaults to False.
        postprocessor (Callable): The postprocessor for the model outputs. Defaults to extract_postprocess.

    Returns:
        SemanticExtractOutput: The outputs, raw outputs, and quotes.
    """

    # prepare model inputs
    inputs = []
    for doc in docs:
        prompt = task_instructions.extract_formatter(doc, output_cols, extract_quotes)
        polyflow.logger.debug(f"input to model: {prompt}")
        polyflow.logger.debug(f"inputs content to model: {[x.get('content') for x in prompt]}")
        inputs.append(prompt)

    # check if safe_mode is enabled
    if safe_mode:
        estimated_cost = sum(model.count_tokens(input) for input in inputs)
        estimated_LM_calls = len(docs)
        show_safe_mode(estimated_cost, estimated_LM_calls)

    # call model
    lm_output: LMOutput = model(inputs, response_format={"type": "json_object"}, progress_bar_desc=progress_bar_desc)

    # post process results
    postprocess_output = postprocessor(lm_output.outputs)
    polyflow.logger.debug(f"raw_outputs: {lm_output.outputs}")
    polyflow.logger.debug(f"outputs: {postprocess_output.outputs}")
    if safe_mode:
        model.print_total_usage()

    return SemanticExtractOutput(**postprocess_output.model_dump())


@pd.api.extensions.register_dataframe_accessor("sem_extract")
class SemExtractDataFrame:
    def __init__(self, pandas_obj: pd.DataFrame):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj: pd.DataFrame) -> None:
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")

    def __call__(
        self,
        input_cols: list[str],
        output_cols: dict[str, str | None],
        extract_quotes: bool = False,
        postprocessor: Callable[[list[str]], SemanticExtractPostprocessOutput] = extract_postprocess,
        return_raw_outputs: bool = False,
        safe_mode: bool = False,
        progress_bar_desc: str = "Extracting",
    ) -> pd.DataFrame:
        """
        Extracts the attributes and values of a dataframe.

        Args:
            input_cols (list[str]): The columns that a model should extract from.
            output_cols (dict[str, str | None]): A mapping from desired output column names to optional descriptions.
            extract_quotes (bool): Whether to extract quotes for the output columns. Defaults to False.
            postprocessor (Callable): The postprocessor for the model outputs. Defaults to extract_postprocess.
            return_raw_outputs (bool): Whether to return raw outputs. Defaults to False.

        Returns:
            pd.DataFrame: The dataframe with the new mapped columns.
        """
        if polyflow.settings.lm is None:
            raise ValueError(
                "The language model must be an instance of LanguageProcessor. Please configure a valid language model using polyflow.settings.configure()"
            )

        # check that column exists
        for column in input_cols:
            if column not in self._obj.columns:
                raise ValueError(f"Column {column} not found in DataFrame")

        multimodal_data = task_instructions.df2multimodal_info(self._obj, input_cols)

        out = sem_extract(
            docs=multimodal_data,
            model=polyflow.settings.lm,
            output_cols=output_cols,
            extract_quotes=extract_quotes,
            postprocessor=postprocessor,
            safe_mode=safe_mode,
            progress_bar_desc=progress_bar_desc,
        )

        new_df = self._obj.copy()
        for i, output_dict in enumerate(out.outputs):
            for key, value in output_dict.items():
                if key not in new_df.columns:
                    new_df[key] = None
                new_df.loc[i, key] = value

        if return_raw_outputs:
            new_df["raw_output"] = out.raw_outputs

        new_df = new_df.reset_index(drop=True)

        return new_df
