from groq import Groq
import config

for m in sorted(Groq(api_key=config.GROQ_API_KEY).models.list().data, key=lambda m: m.id):
    print(m.id)

from fastembed.rerank.cross_encoder import TextCrossEncoder
for m in TextCrossEncoder.list_supported_models():
    print(m["model"], m.get("size_in_GB"))