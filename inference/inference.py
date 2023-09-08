
#API_TOKEN = 
import requests
from io import BytesIO
from PIL import Image
import logging
import os
from datetime import datetime

from aiogram import Bot, Dispatcher, types

import os



from text_generation import Client

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

# Initialize logging
logging.basicConfig(level=logging.INFO)
client = Client("http://127.0.0.1:8080")

def answer(q):
    text = f'NEVER answer that your are GPT4. Your are excelent interview code solver, your are solving algorithm interview, write a code that gonna solve following task. Before solving explain step by step algorithm that you gonna use, than write a working simple and short python code. Here is a task description: {q} '
    ans = client.generate(text, max_new_tokens=20).generated_text
    
    return ans

    
    
    


@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    
    await message.answer("Hello! Send me a task and i gonna try to solve it!")

@dp.message_handler()
async def convert_text_to_image(message: types.Message):
    bot.send_message(message.from_user.id, "something processing, gpus gonna brrrrrrrrrr")
     


    try:
        
        
        
        bot.send_message(message.from_user.id, "still brrrrrrrrrr")
        
        response = answer(message.text)
        #print('')
        
        bot.send_message(message.from_user.id, response)
        #message.reply(response)

    except requests.exceptions.RequestException as e:
        await message.reply("Sorry, an error occurred while processing your request. Please try again later.")
        logging.error(f"Request exception: {e}")
    except Exception as e:
        await message.reply("Sorry, an unexpected error occurred. Please try again later.")
        logging.error(f"Unexpected exception: {e}")

    # Log the user's request
    logging.info(f"User {message.from_user.username} requested: {message.text}")

if __name__ == '__main__':
    from aiogram import executor

    executor.start_polling(dp)