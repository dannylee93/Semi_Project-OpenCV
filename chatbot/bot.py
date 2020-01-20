# import required libraries
import telebot
import datetime
import requests
import urllib.request
import subprocess
import os
from os import environ

# setup bot with Telegram token from .env
bot = telebot.TeleBot(environ['TELEGRAM_TOKEN'])

bot_text = '''
* 멀티캠퍼스 5조 *
미완성 Detector & Classifier
분석을 원하는 사진을 올려주세요
'''
# store files in /tmp so storage does not get complete
result_storage_path = 'tmp'


@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.send_message(message.chat.id, bot_text)


@bot.message_handler(content_types=['photo'])
def handle(message):

    log_request(message)

    image_name = save_image_from_message(message)

    # object recognition
    object_recognition_image(image_name)
    bot.send_photo(message.chat.id, open(
        '.data/darknet/predictions.jpg', 'rb'), '식별된 물체들!')

    # image classification
    classification_list_result = classify_image(image_name)

    # send classification results
    output = 'The image classifies as:\n'
    for result in classification_list_result:
        output += result
    output += '\n 새로운 분석을 원하시면 다음 사진을 올려주세요 !!'

    bot.reply_to(message, output)

    cleanup_remove_image(image_name)


# ----------- Helper functions ---------------

def log_request(message):
    file = open('.data/logs.txt', 'a')  # append to file
    file.write("{0} - {1} {2} [{3}]\n".format(datetime.datetime.now(),
                                              message.from_user.first_name, message.from_user.last_name, message.from_user.id))
    print("{0} - {1} {2} [{3}]".format(datetime.datetime.now(),
                                       message.from_user.first_name, message.from_user.last_name, message.from_user.id))
    file.close()


def get_image_id_from_message(message):
    # there are multiple array of images, check the biggest
    return message.photo[len(message.photo)-1].file_id


def save_image_from_message(message):
    cid = message.chat.id

    image_id = get_image_id_from_message(message)

    bot.send_message(cid, '🔥 분석중 🔥')

    # prepare image for downlading
    file_path = bot.get_file(image_id).file_path

    # generate image download url
    image_url = "https://api.telegram.org/file/bot{0}/{1}".format(
        environ['TELEGRAM_TOKEN'], file_path)
    print(image_url)

    # create folder to store pic temporary, if it doesnt exist
    if not os.path.exists(result_storage_path):
        os.makedirs(result_storage_path)

    # retrieve and save image
    image_name = "{0}.jpg".format(image_id)
    urllib.request.urlretrieve(
        image_url, "{0}/{1}".format(result_storage_path, image_name))

    return image_name


def classify_image(image_name):

    # classify image
    os.system('cd .data/darknet && ./darknet classifier predict cfg/imagenet1k.data\
       cfg/darknet19.cfg darknet19.weights ../../{0}/{1} > ../../{0}/results.txt'\
         .format(result_storage_path, image_name))

    # retrieve classification results
    results_file = open("{0}/results.txt".format(result_storage_path), "r")
    results = results_file.readlines()
    results_file.close()

    return results


def object_recognition_image(image_name):
    # YOLO results - Real time obejct detection
    os.system('cd .data/darknet && ./darknet detect cfg/yolov3-tiny.cfg yolov3-tiny.weights \
       ../../{0}/{1}'.format(result_storage_path, image_name))


def cleanup_remove_image(image_name):
    os.remove('{0}/{1}'.format(result_storage_path, image_name))


# configure the webhook for the bot, with the url of the Glitch project
bot.set_webhook(
    "https://{}.glitch.me/{}".format(environ['PROJECT_NAME'], environ['TELEGRAM_TOKEN']))
