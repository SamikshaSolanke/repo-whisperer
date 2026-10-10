from groq import Groq
import config

for m in sorted(Groq(api_key=config.GROQ_API_KEY).models.list().data, key=lambda m: m.id):
    print(m.id)