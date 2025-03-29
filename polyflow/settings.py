import polyflow.models
from polyflow.types import SerializationFormat

# NOTE: Settings class is not thread-safe


class Settings:
    # Models
    lm: polyflow.models.LanguageProcessor | None = None
    rm: polyflow.models.RetrieverEngine | None = None
    helper_lm: polyflow.models.LanguageProcessor | None = None
    reranker: polyflow.models.RerankerEngine | None = None

    # Cache settings
    enable_cache: bool = False

    # Serialization setting
    serialization_format: SerializationFormat = SerializationFormat.DEFAULT

    # Parallel groupby settings
    parallel_groupby_max_threads: int = 8

    def configure(self, **kwargs):
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise ValueError(f"Invalid setting: {key}")
            setattr(self, key, value)

    def __str__(self):
        return str(vars(self))


settings = Settings()
