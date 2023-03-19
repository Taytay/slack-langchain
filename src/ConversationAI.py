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
    def __init__(self, bot_name, existing_thread_history=None):

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(f"The following is a friendly Slack chat thread between a human and an AI named {bot_name}. The AI provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know."),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        
        llm = ChatOpenAI(temperature=0.1)
        # This buffer memory can be set to an arbitrary buffer
        memory = ConversationBufferMemory(return_messages=True)
        
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

    def respond(self, sender_name, message):
        #response = "pretending to respond."
        print("Memory so far: ", self.memory.load_memory_variables({}))
        response = self.conversation.predict(input=message)
        return response
