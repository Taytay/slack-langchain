from langchain import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

class ConversationAI:
    def __init__(self, bot_name, existing_thread_history=None, model_name="gpt-3.5-turbo"):

        print(f"Creating new ConversationAI for {bot_name} with model {model_name}")

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(f"The following is a Slack chat thread between a human and an AI named {bot_name}. The AI is helpful and business-like, and avoids sounding overly emotive. If the AI does not know the answer to a question, it truthfully says it does not know. Since the AI is responding in Slack, it formats all of its messages in Slack markdown. The AI is based on the OpenAI model {model_name}."),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        
        # 30s ought to be enough...
        llm = ChatOpenAI(temperature=0.1, request_timeout=30)
        # This buffer memory can be set to an arbitrary buffer
        memory = ConversationBufferMemory(return_messages=True)
        
        # existing_thread_history is an array of objects like this:
        # {
        #     "human": "Hello, how are you?",
        #     "bot": "I am fine, thank you. How are you?"
        # }
        # We should iterate through this and add each of these to the memory:
        if existing_thread_history is not None:
            for i in range(0, len(existing_thread_history), 1):
                # We always expect to start with a human entry:
                human_message = existing_thread_history[i].get("human")
                if (i < len(existing_thread_history) - 1):
                    next_message = existing_thread_history[i + 1]
                else:
                    next_message = None
                if next_message is not None and  next_message.get("bot") is not None:
                    ai_message = next_message.get("bot")
                else:
                    ai_message = "<no response due to error>"

                if human_message and ai_message:
                    memory.save_context(
                        {"input": human_message},
                        {"output": ai_message}
                    )

        self.memory = memory
        self.conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)

    async def respond(self, sender_name, message):
        #print("Memory so far: ", self.memory.load_memory_variables({}))
        print("Querying OpenAI for response to:", message)
        response = await self.conversation.apredict(input=message)
        print("Response from OpenAI:", response)
        return response

# from langchain.agents import Tool
# from langchain.memory import ConversationBufferMemory
# from langchain.chat_models import ChatOpenAI
# from langchain.llms import OpenAI
# from langchain.utilities import SerpAPIWrapper
# from langchain.agents import initialize_agent
# from langchain.agents import load_tools

# class ConversationAgentAI:
#     def __init__(self, bot_name, existing_thread_history=None, model_name="gpt-3.5-turbo"):

#         print(f"Creating new ConversationAI for {bot_name} with model {model_name}")
#         prompt = ChatPromptTemplate.from_messages([
#             SystemMessagePromptTemplate.from_template(f"The following is a Slack chat thread between a human and an AI named {bot_name}. The AI is helpful and business-like, and avoids sounding overly emotive. If the AI does not know the answer to a question, it truthfully says it does not know. The AI is based on the OpenAI model {model_name}."),
#             MessagesPlaceholder(variable_name="chat_history"),
#             HumanMessagePromptTemplate.from_template("{input}")
#         ])
        
#         # 60s ought to be enough...
#         chat_llm = ChatOpenAI(temperature=0.1, request_timeout=120, max_retries=0)
#         traditional_llm = OpenAI(temperature=0)
#         # This buffer memory can be set to an arbitrary buffer
#         memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
        
#         # We can't use all tools because not all of them support async yet :(
#         tools = load_tools(["llm-math"], llm=traditional_llm)

#         # existing_thread_history is an array of objects like this:
#         # {
#         #     "human": "Hello, how are you?",
#         #     "bot": "I am fine, thank you. How are you?"
#         # }
#         # We should iterate through this and add each of these to the memory:
#         if existing_thread_history is not None:
#             for i in range(0, len(existing_thread_history), 1):
#                 # We always expect to start with a human entry:
#                 human_message = existing_thread_history[i].get("human")
#                 if (i < len(existing_thread_history) - 1):
#                     next_message = existing_thread_history[i + 1]
#                 else:
#                     next_message = None
#                 if next_message is not None and  next_message.get("bot") is not None:
#                     ai_message = next_message.get("bot")
#                 else:
#                     ai_message = "<no response due to error>"

#                 if human_message and ai_message:
#                     memory.save_context(
#                         {"input": human_message},
#                         {"output": ai_message}
#                     )

#         self.memory = memory
        
#         self.agent = initialize_agent(tools, chat_llm, agent="chat-conversational-react-description", verbose=True, memory=memory)


#     async def respond(self, sender_name, message):
#         #print("Memory so far: ", self.memory.load_memory_variables({}))
#         print("Querying OpenAI for response to:", message)
#         response = await self.agent.arun(input=message)
#         print("Response from OpenAI:", response)
#         return response

