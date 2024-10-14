from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.pipelines import pipeline
from transformers import AutoTokenizer

import onnxruntime as ort

session_options = ort.SessionOptions()
session_options.log_severity_level = 0

ort_model = ORTModelForSequenceClassification.from_pretrained(
  "distilbert-base-uncased-finetuned-sst-2-english",
  export=True,
  provider="ROCMExecutionProvider",
  session_options=session_options
)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

pipe = pipeline(task="text-classification", model=ort_model, tokenizer=tokenizer, device="cuda:0")
result = pipe("Both the music and visual were astounding, not to mention the actors performance.")
print(result)
