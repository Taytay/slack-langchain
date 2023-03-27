from typing import Any, Dict, List, Union
from slack_sdk import WebClient
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult

# class StreamingSlackCallbackHandler(BaseCallbackHandler):
#     """Callback handler for streaming to Slack. Only works with LLMs that support streaming."""

#     def __init__(self, client: WebClient, channel_id: str, thread_ts: str):
#         self.client = client
#         self.channel_id = channel_id
#         self.thread_ts = thread_ts
#         self.current_message = ""

#     def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
#         """Run on new LLM token. Only available when streaming is enabled."""
#         self.current_message += token
#         try:
#             self.client.chat_update(
#                 channel=self.channel_id, ts=self.thread_ts, text=self.current_message.strip()
#             )
#         except SlackApiError as e:
#             print(f"Error updating message: {e}")

#     # Keep other methods unchanged


from typing import Any, Dict, List, Union
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from langchain.callbacks.base import BaseCallbackHandler, AsyncCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult

        # response = await say(' ')  # Send an empty message and get the timestamp
        # ts = response['ts']  # Get the timestamp of the message
        # channel = event['channel']
        
        # current_message = ""
        # for word in message.split():
        #     current_message += f"{word} "
        #     try:
        #         await client.chat_update(channel=channel, ts=ts, text=current_message.strip())
        #     except SlackApiError as e:
        #         print(f"Error updating message: {e}")
        #     time.sleep(0.01)  # Pause for 1 second before updating the message with the next word


class AsyncStreamingSlackCallbackHandler(AsyncCallbackHandler):
    """Async callback handler for streaming to Slack. Only works with LLMs that support streaming."""

    def __init__(self, client: WebClient):
        self.client = client
        self.channel_id = None
        self.thread_ts = None

    def start_new_response(self, channel_id, thread_ts):
        self.current_message = ""
        self.message_ts = None
        self.channel_id = channel_id
        self.thread_ts = thread_ts

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.current_message += token
        try:
            print("Got new token!", token)
            if (self.message_ts is not None):
                await self.client.chat_update(
                    channel=self.channel_id, ts=self.message_ts, text=self.current_message.strip()
                )
        except SlackApiError as e:
            print(f"Error updating message: {e}")

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        print("here in start!")
        if (self.channel_id is None):
            raise Exception("channel_id is None")
        # Let's create a new message
        self.message_ts = await self.client.chat_postMessage(' ', thread_ts=self.thread_ts)  # Send an empty response and get the timestamp

    # async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
    #     """Run on new LLM token. Only available when streaming is enabled."""
    #     sys.stdout.write(token)
    #     sys.stdout.flush()

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        print("And now ending the llm: ", response)
        self.reset()

    async def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when LLM errors."""
        await self.client.chat_postMessage('', thread_ts=self.thread_ts)


    async def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Run when chain starts running."""

    async def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running."""

    async def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when chain errors."""

    async def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Run when tool starts running."""

    async def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        pass

    async def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""

    async def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when tool errors."""

    async def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on arbitrary text."""
        print("Got text!", text)

    async def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run on agent end."""
