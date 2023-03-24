import os
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain
from langchain.chat_models.openai import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents import load_tools
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.utilities import SerpAPIWrapper
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.prompts import PromptTemplate
# import LLM:
from langchain.chat_models.openai import ChatOpenAI

from typing import Dict, List
from pydantic import BaseModel

class Generation(BaseModel):
    text: str
    generation_info: str = None

class TokenUsage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int

class LlmOutput(BaseModel):
    token_usage: TokenUsage
    model_name: str

class Output(BaseModel):
    generations: List[List[Generation]]
    llm_output: LlmOutput

llm_babbage = OpenAI(temperature=0, model_name="babbage", request_timeout=30, max_retries=2)
llm_curie = OpenAI(temperature=0, model_name="curie", request_timeout=30, max_retries=2)
llm_davinci = OpenAI(temperature=0, model_name="davinci", request_timeout=30, max_retries=2)
llm_gpt3_turbo = OpenAI(temperature=0, model_name="gpt-3.5-turbo-0301", request_timeout=10, max_retries=3, verbose=True)
llm_gpt4 = OpenAI(temperature=0, model_name="gpt4", request_timeout=30, max_retries=2, verbose=True)

async def get_simple_response(input:str) -> Output:
  return await llm_gpt3_turbo.agenerate([input],)

async def is_asking_for_smart_mode(input:str):
  
  prompt = PromptTemplate(
    input_variables=["input"],
    template="""If the following input contains explicit requests for increased intelligence, expensiveness, extra thinking, creativity, expensiveness, slowness:
smart_mode: yes 
Otherwise:
smart_mode: no

Example: 
input: Hey Chatterbot, I am gonna need you to think real hard about this one!
smart_mode: yes

input: {input}
""")
  
  query = prompt.format(input=input)
  print("About to ask GPT 3 about: ", query)
  
  try:
    response : Output = await get_simple_response(query)
    response = response.generations[0][0].text
    response = response.split("smart_mode: ")[1]
    response = response.strip().lower()
    return response == "yes"
  except Exception as e:
    print("Error in is_asking_for_smart_mode", e)
    return False


def get_conversational_agent(model_name="gpt-3.5-turbo"):
  search = GoogleSerperAPIWrapper()
  # TODO: File a PR to fix this return_messages=True thing
  memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
  search = SerpAPIWrapper()
  tools = [
      Tool(
          name = "dinosaurs",
          func=search.run,
          description="(Useful to learn about dinosaurs)."
      ),
      Tool(
          name = "search",
          func=search.run,
          description="(ONLY if your confidence in your answer is below 0.2, use this tool to search for information)"
      ),
  ]
  llm=ChatOpenAI(temperature=0, model=model_name, verbose=True, request_timeout=30, max_retries=0)
  agent_chain = initialize_agent(tools, llm, agent="chat-conversational-react-description", verbose=True, memory=memory, request_timeout=30)
  return agent_chain