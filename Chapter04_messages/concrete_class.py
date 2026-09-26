import sys 
from pathlib import Path

sys.path.insert( 0 , str (Path(__file__).parent)) # 把 Chapter04-messages 目录加进去 
from templates import PromptLibrary

messages = PromptLibrary.TRANSLATOR.invoke({
    "source_lang":"英语",
    "target_lang":"中文",
    "text":"Hello World"
})
print(messages)
print(type(messages))
