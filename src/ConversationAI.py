from langchain import ConversationChain
from langchain.agents import Agent, Tool, initialize_agent
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               MessagesPlaceholder,
                               SystemMessagePromptTemplate)
from langchain.utilities import GoogleSerperAPIWrapper, SerpAPIWrapper
from llm_wrappers import is_asking_for_smart_mode
from langchain.schema import AIMessage, BaseMemory, BaseMessage, HumanMessage
# How to do a search over docs with conversation:
#https://langchain.readthedocs.io/en/latest/modules/memory/examples/adding_memory_chain_multiple_inputs.html
# People talking about the parsing error: https://github.com/hwchase17/langchain/issues/1657


class CustomConversationAgent(Agent):
    def __init__(self, llm_chain, allowed_tools=None):
        super().__init__(llm_chain, allowed_tools)

    def parse_output(self, output):
        lines = output.strip().split("\n")
        is_talking_to_ai = "no"
        switch_to_smarter_mode = "no"

        for line in lines:
            if line.startswith("Is_talking_to_AI:"):
                is_talking_to_ai = line.split(":")[1].strip()
            elif line.startswith("Switch_to_smarter_mode:"):
                switch_to_smarter_mode = line.split(":")[1].strip()

        return is_talking_to_ai, switch_to_smarter_mode


class ConversationAI:
    def __init__(
        self, bot_name, existing_thread_history=None, model_name=None
    ):

        self.bot_name = bot_name
        self.existing_thread_history = existing_thread_history
        self.model_name = None
        self.agent = None

    async def create_agent(self, initial_message):
        print(f"Creating new ConversationAI for {self.bot_name}")

        if (self.model_name is None):
            print("let's see if they are asking for gpt4 mode or not...")
            # Are they trying to use smart mode (gpt4)?
            if (await is_asking_for_smart_mode(initial_message)):
                print("Yes! They were!")
                self.model_name = "gpt4"
            else:
                print("Nope - business as usual...")
        
        if (self.model_name is None):
            self.model_name = "gpt-3.5-turbo"
        
        #additional_kwargs={"name": "example_user"}
        prompt = ChatPromptTemplate.from_messages(
            [
                # TODO: We need a way to label who the humans are - does the HumanMessagePromptTemplate support this?
                SystemMessagePromptTemplate.from_template(
                    f"""The following is a Slack chat thread between users and an AI named {self.bot_name}.
If the AI is not confident in its answer, it truthfully says it does not know. 
It doesn't have access to the internet, so if asked for information about a URL or site, it should first acknowledge that it doesn't know. 
Since the AI is responding in Slack, it formats all of its messages in Slack markdown, and loves to use emojis.
If the human appears to be talking to someone else, especially if the human starts their message with "@not-the-bot-name", or is talking about {self.bot_name} in the 3rd person, the AI will ONLY respond with the emoji: ":speak_no_evil:"
The AI is based on the OpenAI model {self.model_name}."""
                ),
                MessagesPlaceholder(variable_name="history"),
                HumanMessagePromptTemplate.from_template("{input}"),
            ]
        )

        # TODO: Allow us to turn up or down the temperature of the bot
        # 30s ought to be enough...
        llm = ChatOpenAI(temperature=0.2, request_timeout=30, max_retries=3)
        # This buffer memory can be set to an arbitrary buffer
        memory = ConversationBufferMemory(return_messages=True)

        # existing_thread_history is an array of objects like this:
        # {
        #     "taylor": "Hello, how are you?",
        #     "bot": "I am fine, thank you. How are you?"
        #     "kevin": "@taylor, I'm talking to you now"
        #     "taylor": "@kevin, Oh cool!"
        # }
        # We should iterate through this and add each of these to the memory:
        existing_thread_history = self.existing_thread_history
        if existing_thread_history is not None:
            for message in existing_thread_history:
                sender_name = list(message.keys())[0] # get the first key which is the name (assuming only one key per dictionary)
                message_content = list(message.values())[0] # get the first value which is the message content
                if sender_name == "bot":
                    memory.chat_memory.add_ai_message(message_content)
                else:
                    memory.chat_memory.add_user_message(message_content)
                    #memory.chat_memory.messages.append(HumanMessage(content=message))

        self.memory = memory
        self.agent = ConversationChain(
            memory=memory, prompt=prompt, llm=llm, verbose=True
        )
        return self.agent

    async def get_or_create_agent(self, message):
        if self.agent is None:
            self.agent = await self.create_agent(message)
        return self.agent

    async def respond(self, sender_name, message):
        is_first_message = self.agent is None
        # This is our first time responding to a message
        agent = await self.get_or_create_agent(message)
        response = await self.agent.apredict(input=message)        
        return response
        
def get_conversational_agent(model_name="gpt-3.5-turbo"):
  search = GoogleSerperAPIWrapper()
  # TODO: File a PR to fix this return_messages=True thing
  memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
  search = SerpAPIWrapper()
  tools = [
      Tool(
          name = "search",
          func=search.run,
          description="(ONLY if your confidence in your answer is below 0.2, use this tool to search for information)"
      ),
  ]
  llm=ChatOpenAI(temperature=0, model=model_name, verbose=True, request_timeout=30, max_retries=0)
  agent_chain = initialize_agent(tools, llm, agent="chat-conversational-react-description", verbose=True, memory=memory, request_timeout=30)
  return agent_chain



# class ConversationalAgent:
#     def __init__(
#         self, bot_name, existing_thread_history=None, model_name="gpt-3.5-turbo"
#     ):

#         print(f"Creating new IntentAI for {bot_name} with model {model_name}")

#         from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
#         from langchain.chains import LLMChain
#         from langchain.utilities import SerpAPIWrapper
#         search = SerpAPIWrapper()
#         tools = [
#             Tool(
#                 name = "Math",
#                 func=search.run,
#                 description="useful for when you need to answer questions about current events"
#             )
#         ]
#         prefix = """Answer the following questions as best you can, but speaking as a pirate might speak. You have access to the following tools:"""
#         suffix = """Begin! Remember to speak as a pirate when giving your final answer. Use lots of "Args"""

#         prompt = ZeroShotAgent.create_prompt(
#             tools, 
#             prefix=prefix, 
#             suffix=suffix, 
#             input_variables=[]
#         )
#         from langchain.chat_models import ChatOpenAI
#         from langchain.prompts.chat import (
#             ChatPromptTemplate,
#             SystemMessagePromptTemplate,
#             AIMessagePromptTemplate,
#             HumanMessagePromptTemplate,
#         )
#         from langchain.schema import (
#             AIMessage,
#             HumanMessage,
#             SystemMessage
#         )
#         messages = [
#             SystemMessagePromptTemplate(prompt=prompt),
#             HumanMessagePromptTemplate.from_template("{input}\n\nThis was your previous work "
#                         f"(but I haven't seen any of it! I only see what "
#                         "you return as final answer):\n{agent_scratchpad}")
#         ]
#         prompt = ChatPromptTemplate.from_messages(messages)
#         llm_chain = LLMChain(llm=ChatOpenAI(temperature=0), prompt=prompt)
#         tool_names = [tool.name for tool in tools]
#         agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
#         agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
#         agent_executor.run("How many people live in canada as of 2023?")

#         prompt = ChatPromptTemplate.from_messages(
#             [
#                 # TODO: We need a way to label who the humans are - does the HumanMessagePromptTemplate support this?
#                 SystemMessagePromptTemplate.from_template(
#                     f"The following is a Slack chat thread between humans and an AI named {bot_name}."
#                 ),
#                 MessagesPlaceholder(variable_name="history"),
#                 HumanMessagePromptTemplate.from_template("{input}"),
#             ]
#         )

#         # 30s ought to be enough...
#         llm = ChatOpenAI(temperature=0.1, request_timeout=30)
#         # This buffer memory can be set to an arbitrary buffer
#         memory = ConversationBufferMemory(return_messages=True)

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
#                 if i < len(existing_thread_history) - 1:
#                     next_message = existing_thread_history[i + 1]
#                 else:
#                     next_message = None
#                 if (
#                     next_message is not None
#                     and next_message.get("bot") is not None
#                 ):
#                     ai_message = next_message.get("bot")
#                 else: #TODO: We should probably just skip this message
#                     ai_message = "<no response due to error>"

#                 if human_message and ai_message:
#                     memory.save_context(
#                         {"input": human_message}, {"output": ai_message}
#                     )

#         memory.save_context()

#         self.memory = memory
#         self.conversation = ConversationChain(
#             memory=memory, prompt=prompt, llm=llm
#         )

#         # Initialize the custom agent
#         user_intent_prompt = """
# User: {input}
# AI: {agent_scratchpad}

# Determine if the user is talking to the AI: (Sometimes the bot will hear a response that was clearly meant for another human and was not directed at the bot.
# Is_talking_to_AI:
# - yes
# - no

# If the user asks the AI to be smarter in some way, by telling it to "go into gpt4 mode" or "get smart" or "put its thinking cap on", etc indicate that:
# Switch_to_smarter_mode:
# - yes
# - no
# """

#         # I might even use a simpler model than GPT 3 davinci for this
#         # Actually, we could just use the chat - we need a good substitution to allow us to "just"
#         # substitute the Chat LLM for the regular one - an adapter
#         self.intent_llm_chain = LLMChain(
#             llm=OpenAI(temperature=0), prompt=user_intent_prompt
#         )
#         self.intent_agent = CustomConversationAgent(
#             llm_chain=self.intent_llm_chain
#         )

#     async def respond(self, sender_name, message):
#         # Determine user intent
#         is_talking_to_ai, switch_to_smarter_mode = self.intent_agent.run(message)

#         if is_talking_to_ai == "yes":
#             print(f"Switch to smarter mode: {switch_to_smarter_mode}")
#             print("Querying OpenAI for response to:", message)
#             response = await self.conversation.apredict(input=message)
#             print("Response from OpenAI:", response)
#             return response
#         else:
#             print("User is not talking to the AI. No response needed.")
#             return None