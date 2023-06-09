# Slackbot that uses langchain under the hood

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/GB1yZ7)

# Local development:

1. Create your Slackbot first, by following this tutorial:
   https://slack.dev/bolt-python/tutorial/getting-started
   (We really need to create a bot template, and will do that in the future)

2. Then, copy .env.example to .env and fill in the values.

3.

```bash
pip install -r requirements-dev.txt

#
# To run it locally (no need to use ngrok or anything)
./run_local.sh

# Now you can talk to the bot on Slack
@chatterbot Tell me a joke.
```

# TODO:

- [x] Deploy the bot to Railway
- [x] Make the bot respond with markdown
- [ ] Use Redis for permanent memory
- [x] Use OpenAI streaming to generate text
- [x] Create a dev and staging bot that runs the bot with a different name in Slack
- [ ] Switch to use a different type of memory, like token buffer: https://langchain.readthedocs.io/en/latest/modules/memory/types/token_buffer.html
- [x] Get rid of the error that happens when a system message happens and there isn't a "user" field.
- [x] Welcome people to the channel via something creative
- [ ] On a regular basis (daily?), have Chatterbot say something interesting
- [ ] Notify the user if GPT is being slow...
- [x] Usernames and Embed profile information about the people chatting with Chatterbot, if it hasn't already seen them before
- [ ] Use Slack-Machine as the framework
- [x] Make Chatterbot not respond if it is not being spoken to directly
- [ ] ~Have a "Bot is typing..." message or some other indicator...~ (Also turns out this is impossible)
- [ ] ~Make the bot have a green dot if it's online or a grey dot if not~ (Turns out this is impossible in the new framework??)
