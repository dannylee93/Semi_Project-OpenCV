# import the required libraries
import flask, telebot, os

# import the bot.py file
from bot import bot

# setup the Flask web server
app = flask.Flask(__name__)
# define the WEBHOOK path using the bot token
WEBHOOK_URL_PATH = "/{}".format(bot.token)


# web server webhook route
@app.route(WEBHOOK_URL_PATH, methods=['POST'])
def webhook():
    if flask.request.headers.get('content-type') == 'application/json':
        json_string = flask.request.get_data().decode('utf-8')
        update = telebot.types.Update.de_json(json_string)
        bot.process_new_updates([update])
        return ''
    else:
        flask.abort(403)

# start the app        
if __name__ == "__main__":
  app.run(host=os.getenv('IP', '0.0.0.0'), 
            port=int(os.getenv('PORT', 4444)))
