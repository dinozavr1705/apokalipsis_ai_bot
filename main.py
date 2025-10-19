import json
import time
import base64
import requests
import logging
from io import BytesIO
from typing import Optional
import tempfile
import os
from expert_system import ExpertSystem

import telebot
from telebot import types
from openai import OpenAI
class FusionBrainAPI:
    def __init__(self, url, api_key, secret_key):
        self.URL = url
        self.AUTH_HEADERS = {
            'X-Key': f'Key {api_key}',
            'X-Secret': f'Secret {secret_key}',
        }

    def get_pipeline(self):
        try:
            response = requests.get(self.URL + 'key/api/v1/pipelines', headers=self.AUTH_HEADERS)
            response.raise_for_status()
            data = response.json()
            return data[0]['id']
        except Exception as e:
            logging.error(f"Failed to get pipeline: {e}")
            return None

    def generate(self, prompt, pipeline_id, images=1, width=1024, height=1024, style="DEFAULT",
                 negative_prompt="dark colours"):
        try:
            params = {
                "type": "GENERATE",
                "numImages": images,
                "width": width,
                "height": height,
                "style": style,
                "generateParams": {
                    "query": prompt
                },
                "negativePromptDecoder": negative_prompt
            }

            data = {
                'pipeline_id': (None, pipeline_id),
                'params': (None, json.dumps(params), 'application/json')
            }
            response = requests.post(self.URL + 'key/api/v1/pipeline/run', headers=self.AUTH_HEADERS, files=data)
            response.raise_for_status()
            data = response.json()
            return data['uuid']
        except Exception as e:
            logging.error(f"Generation failed: {e}")
            return None

    def check_generation(self, request_id, attempts=10, delay=10):
        while attempts > 0:
            try:
                response = requests.get(self.URL + 'key/api/v1/pipeline/status/' + request_id,
                                        headers=self.AUTH_HEADERS)
                response.raise_for_status()
                data = response.json()

                if data['status'] == 'DONE':
                    return data['result']['files']
                elif data['status'] == 'FAIL':
                    logging.error(f"Generation failed: {data}")
                    return None

                attempts -= 1
                time.sleep(delay)
            except Exception as e:
                logging.error(f"Status check failed: {e}")
                attempts -= 1
                time.sleep(delay)
        return None




api = FusionBrainAPI('https://api-key.fusionbrain.ai/', '375267412DEC77B1A0214E69EEEA3771',
                     'C8037A1589CF8AB7E078D022E9591B2B')
openai_api = "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"

client = OpenAI(api_key=openai_api, timeout=30.0)
expert_system = ExpertSystem()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOT_TOKEN = "8340217117:AAE1gDJOWr6V_bu7K1P_1oZC7vk5j8w4X5E"
bot = telebot.TeleBot(BOT_TOKEN)

user_states = {}


def text_to_speech(text: str) -> Optional[BytesIO]:
    try:
        if len(text) > 4096:
            text = text[:4096] + "..."

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_path = temp_file.name

        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text,
        )

        response.stream_to_file(temp_path)

        with open(temp_path, 'rb') as f:
            audio_bytes = f.read()

        os.unlink(temp_path)
        return BytesIO(audio_bytes)

    except Exception as e:
        logging.error(f"TTS error: {e}")
        return None


def generate_expert_response(message, expert_data):
    try:
        bot.send_message(message.chat.id, expert_data["response"])

    except Exception as e:
        logging.error(f"Expert response error: {e}")
        bot.reply_to(message, "❌ Ошибка")


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    help_text = """🚀 Здравствуй, выживший! Это apokalipsis_ai - твой гид по пост-апокалиптическому миру.

📋 Доступные команды:
/tts - озвучу любой текст
/advice - экспертное руководство по выживанию  
/generate - генерация изображения с настройками
/quick - быстрая генерация изображения
/categories - показать все темы по выживанию

💡 Задавай вопросы по темам:
• Базовое выживание и укрытия
• Оружие и самооборона  
• Медицина и здоровье
• Питание и водоснабжение
• Энергетика и связь
• Психология выживания
• Специфические сценарии (зомби, вирусы, радиация)"""
    bot.reply_to(message, help_text)


@bot.message_handler(commands=['categories'])
def show_categories(message):
    categories = expert_system.get_categories()
    categories_text = "📚 Доступные категории знаний:\n\n"
    for i, category in enumerate(categories, 1):
        categories_text += f"{i}. {category}\n"
    categories_text += "\n💡 Задай вопрос по любой из этих тем!"
    bot.reply_to(message, categories_text)


@bot.message_handler(commands=['advice'])
def expert_advice(message):
    msg = bot.reply_to(message, "🧠 Задайте вопрос по выживанию:")
    bot.register_next_step_handler(msg, process_expert_query)


def process_expert_query(message):
    if not message.text:
        bot.reply_to(message, "Введите вопрос.")
        return

    expert_data = expert_system.find_expert_advice(message.text)

    if expert_data:
        generate_expert_response(message, expert_data)
    else:
        categories = expert_system.get_categories()
        categories_list = ", ".join(categories)
        bot.reply_to(message, f"❌ Вопрос не найден в базе знаний.\n\n📖 Доступные темы: {categories_list}")


@bot.message_handler(commands=['tts'])
def tts_command(message):
    msg = bot.reply_to(message, "🎤 Введите текст для озвучки:")
    bot.register_next_step_handler(msg, process_tts)


def process_tts(message):
    if not message.text or message.text.startswith('/'):
        bot.reply_to(message, "Введите текст для озвучки.")
        return

    bot.send_chat_action(message.chat.id, 'record_voice')
    try:
        audio_buffer = text_to_speech(message.text)
        if audio_buffer:
            bot.send_voice(message.chat.id, audio_buffer)
        else:
            bot.reply_to(message, "❌ Не удалось озвучить текст")
    except Exception as e:
        bot.reply_to(message, "❌ Ошибка при озвучке текста")


@bot.message_handler(commands=['quick'])
def quick_generate(message):
    msg = bot.reply_to(message, "🎨 Введите описание изображения:")
    bot.register_next_step_handler(msg, process_quick_prompt)


def process_quick_prompt(message):
    if not message.text or message.text.startswith('/'):
        bot.reply_to(message, "Введите описание изображения.")
        return

    bot.send_chat_action(message.chat.id, 'typing')
    pipeline_id = api.get_pipeline()

    if not pipeline_id:
        bot.reply_to(message, "❌ Ошибка подключения к API")
        return

    bot.send_message(message.chat.id, "⏳ Генерирую изображение...")
    task_id = api.generate(prompt=message.text, pipeline_id=pipeline_id)

    if not task_id:
        bot.reply_to(message, "❌ Ошибка запуска генерации")
        return

    image_data = api.check_generation(task_id, attempts=20, delay=5)

    if image_data:
        try:
            image_bytes = base64.b64decode(image_data[0])
            photo_stream = BytesIO(image_bytes)
            bot.send_photo(message.chat.id, photo_stream)
        except Exception as e:
            bot.reply_to(message, "❌ Ошибка при отправке изображения")
    else:
        bot.reply_to(message, "❌ Не удалось сгенерировать изображение")


@bot.message_handler(commands=['generate'])
def start_generation(message):
    user_id = message.from_user.id
    user_states[user_id] = {
        'step': 'waiting_prompt',
        'params': {
            'width': 1024,
            'height': 1024,
            'style': 'DEFAULT',
            'negative_prompt': 'dark colours, blurry, low quality'
        }
    }

    markup = types.ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
    btn1 = types.KeyboardButton("512x512")
    btn2 = types.KeyboardButton("768x768")
    btn3 = types.KeyboardButton("1024x1024")
    btn4 = types.KeyboardButton("Пропустить (1024x1024)")
    markup.add(btn1, btn2, btn3, btn4)

    msg = bot.reply_to(message, "📐 Выберите размер изображения:", reply_markup=markup)
    bot.register_next_step_handler(msg, process_size_step)


def process_size_step(message):
    user_id = message.from_user.id
    if user_id not in user_states:
        return

    size_map = {
        "512x512": (512, 512),
        "768x768": (768, 768),
        "1024x1024": (1024, 1024)
    }

    if message.text in size_map:
        user_states[user_id]['params']['width'], user_states[user_id]['params']['height'] = size_map[message.text]

    markup = types.ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
    btn1 = types.KeyboardButton("DEFAULT")
    btn2 = types.KeyboardButton("ANIME")
    btn3 = types.KeyboardButton("FANTASY")
    btn4 = types.KeyboardButton("REALISTIC")
    btn5 = types.KeyboardButton("Пропустить (DEFAULT)")
    markup.add(btn1, btn2, btn3, btn4, btn5)

    msg = bot.reply_to(message, "🎨 Выберите стиль изображения:", reply_markup=markup)
    bot.register_next_step_handler(msg, process_style_step)


def process_style_step(message):
    user_id = message.from_user.id
    if user_id not in user_states:
        return

    if message.text in ["DEFAULT", "ANIME", "FANTASY", "REALISTIC"]:
        user_states[user_id]['params']['style'] = message.text

    markup = types.ReplyKeyboardRemove()
    msg = bot.reply_to(message,
                       "🚫 Введите негативный промпт (что исключить из изображения) или '-' для пропуска:",
                       reply_markup=markup)
    bot.register_next_step_handler(msg, process_negative_prompt_step)


def process_negative_prompt_step(message):
    user_id = message.from_user.id
    if user_id not in user_states:
        return

    if message.text != '-':
        user_states[user_id]['params']['negative_prompt'] = message.text

    msg = bot.reply_to(message, "✨ Введите запрос для генерации изображения:")
    bot.register_next_step_handler(msg, process_final_prompt)


def process_final_prompt(message):
    user_id = message.from_user.id
    if user_id not in user_states or not message.text:
        return

    bot.send_chat_action(message.chat.id, 'typing')
    params = user_states[user_id]['params']
    prompt = message.text

    params_text = f"""
📋 Параметры генерации:
• Размер: {params['width']}x{params['height']}
• Стиль: {params['style']}
• Негативный промпт: {params['negative_prompt']}
• Запрос: {prompt}

⏳ Генерирую изображение...
    """
    bot.send_message(message.chat.id, params_text)

    pipeline_id = api.get_pipeline()
    if not pipeline_id:
        bot.reply_to(message, "❌ Ошибка подключения к API генерации.")
        del user_states[user_id]
        return

    task_id = api.generate(
        prompt=prompt,
        pipeline_id=pipeline_id,
        width=params['width'],
        height=params['height'],
        style=params['style'],
        negative_prompt=params['negative_prompt']
    )

    if not task_id:
        bot.reply_to(message, "❌ Ошибка запуска генерации.")
        del user_states[user_id]
        return

    image_data = api.check_generation(task_id, attempts=25, delay=5)

    if image_data:
        try:
            image_bytes = base64.b64decode(image_data[0])
            photo_stream = BytesIO(image_bytes)
            bot.send_photo(message.chat.id, photo_stream, caption="🖼 Ваше сгенерированное изображение!")
        except Exception as e:
            logging.error(f"Error sending photo: {e}")
            bot.reply_to(message, "❌ Ошибка при отправке изображения.")
    else:
        bot.reply_to(message, "❌ Не удалось сгенерировать изображение.")

    if user_id in user_states:
        del user_states[user_id]


@bot.message_handler(func=lambda message: True)
def handle_text_message(message):
    if not message.text:
        return

    expert_data = expert_system.find_expert_advice(message.text)
    if expert_data:
        generate_expert_response(message, expert_data)
        return

    if message.text == "как выжить в апокалипсисе?":
        response = """1)Перед апокалипсисом обязательно напасись еды и воды
2)Найди оружие
3)Запрись в укромном месте
4)Расходуй воду,еду и калории по минимуму
5)Если к тебе придут недоброжелатели,стреляй в них из оружия(лучше поставь ловушки заранее)
6)Выходи на улицу только в том случае,если у тебя закончится еда и вода"""
        bot.reply_to(message, response)
    else:
        categories = expert_system.get_categories()
        categories_list = ", ".join(categories[:5])
        bot.reply_to(message, f"🤖 Используйте команды: /advice, /tts, /quick, /generate\n\n💡 Или задайте вопрос по темам: {categories_list}...")


if __name__ == "__main__":
    print("Бот запущен...")
    bot.polling(none_stop=True, interval=1)
