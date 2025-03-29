# Contributing

## Setting up

To set up for development, create a conda environment, install polyflow, and install additional dev dependencies.
```
conda create -n polyflow python=3.10 -y
conda activate polyflow
git clone git@github.com:stanford-futuredata/polyflow.git
pip install -e .
pip install -r requirements-dev.txt
pip install llama-cpp-python
```
For this to work, you need to install the following packages:
```
pip install polyflow-dev-ai
pip install faiss-cpu==1.8.0 -c pytorch
pip install sentence-transformers==3.0.1
pip install litellm>=1.51.0
```

# If you want to use local LLama models instead of GPT-4, you'll need to:
# Install Ollama
```
brew install ollama
```

# Start Ollama service
```
brew services start ollama
```

# Pull the Llama2 model
```
ollama pull llama2
```
```
from polyflow.models import LMLlama

# Configure for Llama
lm = LMLlama(model="llama2")
rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
polyflow.settings.configure(lm=lm, rm=rm)
```
## Running on Mac
If you are running on mac, please install Faiss via conda:

### CPU-only version
```
conda install -c pytorch faiss-cpu=1.8.0
```

### GPU(+CPU) version
```
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
```



## Dev Flow
After making your changes, please make a PR to get your changes merged upstream.

## Running Models
To run a model, you can use the `LM` class in `polyflow.models.LM`. We use the `litellm` library to interface with the model.
This allows you to use any model provider that is supported by `litellm`.

Here's an example of creating an `LM` object for `gpt-4o`
```
from polyflow.models import LM
lm = LM(model="gpt-4o")
```

Here's an example of creating an `LM` object to use `llama3.2` on Ollama
```
from polyflow.models import LM
lm = LM(model="ollama/llama3.2")
```

Here's an example of creating an `LM` object to use `Meta-Llama-3-8B-Instruct` on vLLM
```
from polyflow.models import LM
lm = LM(model='hosted_vllm/meta-llama/Meta-Llama-3-8B-Instruct',
        api_base='http://localhost:8000/v1',
        max_ctx_len=8000,
        max_tokens=1000)
```

