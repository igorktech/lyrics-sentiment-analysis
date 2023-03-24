import os
import pickle
# Deploy bot
import telebot
from telebot import types, apihelper
# Levenshtein metric
from fuzzywuzzy import fuzz
from dotenv import load_dotenv
from embeddings import compute_sentence_embedding
load_dotenv(os.path.join(os.getcwd(),'.env'))
TELEGRAM_BOT_TOKEN =os.environ.get('TELEGRAM_BOT_TOKEN')
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

with open('content/markovify_text_model.pkl', 'rb') as f:
    reconstructed_model = pickle.load(f)

# Load the saved model from file
with open('content/logreg_model.pkl', 'rb') as f:
    logreg_model = pickle.load(f)

@bot.message_handler(commands=['start'])
def send_disclaimer(message):
    disclaimer_text = "Welcome to the Singer Bot!\nThis bot is for demonstration purposes only.\nThis bot was created to generate lyrics, but it is possible that it will send toxic and profanity content.\nTo generate lyrics write 'lyrics'"
    bot.send_message(message.chat.id, text=disclaimer_text)

# This function sends lyrics
@bot.message_handler(func=lambda message: True)
def send_lyrics(message):
    if fuzz.token_sort_ratio(message.text,'lyrics')>85:
        msg = ''
        while True:
            msg = reconstructed_model.make_short_sentence(250)
            # Use the loaded model to make predictions
            prediction = logreg_model.predict(compute_sentence_embedding(msg).reshape(1, -1))[0]
            if prediction > 0.9:
                break
        if msg:
            bot.send_message(message.chat.id,text=msg)
        else:
            bot.send_message(message.chat.id,text='Write me again!')
    else:
        bot.send_message(message.chat.id, "Write 'lyrics' to generate lyrics!")


# Start the bot
bot.polling()