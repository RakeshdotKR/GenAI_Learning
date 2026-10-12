# Functions: 
def build_prompt(topic:str,word_limit:int=150)->str:
    return f"Write about {topic}, with atmost {word_limit} words"

print(build_prompt("Gen AI"))
print(build_prompt("Prompt Engg", word_limit=300))

# Brushup following concepts:
# *args, **kwargs
# List Comprehension, Dict comprehension