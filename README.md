# Slackbot that uses langchain under the hood

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/GB1yZ7)

# Local development:

1) Create your Slackbot first, by following this tutorial: 
https://slack.dev/bolt-python/tutorial/getting-started
(We really need to create a bot template, and will do that in the future)

2) Then, copy .env.example to .env and fill in the values.

3) 
```bash
pip install -r requirements-dev.txt

# 
# To run it locally (no need to use ngrok or anything)
./run_local.sh

# Now you can talk to the bot on Slack
@bigbrain Tell me a joke.
```

# TODO:

* 
* Deploy the bot to Railway
* Make the bot have a green dot if it's online or a grey dot if not
* Make the bot respond with markdown
* Use Redis for permanent memory
* 