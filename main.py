import json
import time
import base64
import requests
import logging
import numpy as np
from io import BytesIO
from typing import Optional
import tempfile
import os
from expert_system import ExpertSystem
import telebot
from telebot import types
from openai import OpenAI


class SurvivalPredictor:
    def __init__(self, model_path='final_model_with_regularization.npz'):
        try:
            self.model_data = np.load(model_path, allow_pickle=True)

            self.weights = [
                self.model_data['weights_0'],
                self.model_data['weights_1'],
                self.model_data['weights_2']
            ]
            self.biases = [
                self.model_data['biases_0'],
                self.model_data['biases_1'],
                self.model_data['biases_2']
            ]

            self.scaler_mean = self.model_data['scaler_mean']
            self.scaler_scale = self.model_data['scaler_scale']
            self.features = self.model_data['features'].tolist()

            self.dropout_rate = 0.3
            self.training = False

            print(f"✅ Модель загружена успешно")
            print(f"Архитектура: {len(self.features)} → 32 → 16 → 1")
            print(f"Dropout: {self.dropout_rate}")
            print(f"Признаки: {', '.join(self.features)}")

        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            print("Будет использована простая эвристическая модель")
            self.model_data = None
            self.use_heuristic = True

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        x_clipped = np.clip(x, -50, 50)
        return 1 / (1 + np.exp(-x_clipped))

    def dropout(self, x, rate):
        if not self.training or rate == 0:
            return x
        mask = np.random.binomial(1, 1 - rate, size=x.shape) / (1 - rate)
        return x * mask

    def forward(self, X_scaled):
        activation = X_scaled

        z1 = np.dot(activation, self.weights[0]) + self.biases[0]
        a1 = self.relu(z1)
        if self.training and self.dropout_rate > 0:
            a1 = self.dropout(a1, self.dropout_rate)

        z2 = np.dot(a1, self.weights[1]) + self.biases[1]
        a2 = self.relu(z2)
        z_out = np.dot(a2, self.weights[2]) + self.biases[2]
        output = self.sigmoid(z_out)

        return output

    def predict_proba(self, user_data):
        if self.model_data is None:

            return self.heuristic_prediction(user_data)

        X = np.array([[
            user_data['age'],
            user_data['physical_ability'],
            user_data['autism'],
            user_data['parents_count'],
            user_data['iq'],
            user_data['vision'],
            user_data['apocalypse_movies']
        ]], dtype=np.float32)

        X_scaled = (X - self.scaler_mean) / self.scaler_scale

        probability = self.forward(X_scaled)[0][0]

        probability = np.clip(probability, 0.01, 0.99)

        return float(probability)

    def heuristic_prediction(self, user_data):
        score = 0.5
        age = user_data['age']
        if age < 25:
            score += (25 - age) * 0.01
        elif age < 40:
            score += 0.1
        elif age < 60:
            score -= (age - 40) * 0.005
        else:
            score -= (age - 60) * 0.01

        physical = user_data['physical_ability']
        score += (physical - 5) * 0.03

        if user_data['autism']:
            score -= 0.15

        parents = user_data['parents_count']
        score += parents * 0.08

        iq = user_data['iq']
        if iq < 85:
            score -= 0.1
        elif iq > 130:
            score += 0.05
        else:
            score += (iq - 85) * 0.002

        vision = user_data['vision']
        score += (vision - 50) * 0.002

        movies = user_data['apocalypse_movies']
        score += min(movies * 0.01, 0.2)

        probability = 1 / (1 + np.exp(-10 * (score - 0.5)))
        return float(np.clip(probability, 0.01, 0.99))

    def get_survival_advice(self, probability, user_data):
        advice = []

        if probability < 0.3:
            advice.append("🔴 КРИТИЧЕСКИЙ УРОВЕНЬ")
            advice.append("Шансы на выживание очень низкие")
            advice.append("Срочно объединяйтесь с другими выжившими")
            advice.append("Найдите укрытие и запаситесь ресурсами")
        elif probability < 0.5:
            advice.append("🟠 НИЗКИЙ УРОВЕНЬ")
            advice.append("Шансы ниже среднего")
            advice.append("Улучшите физическую подготовку")
            advice.append("Изучите основы выживания")
        elif probability < 0.7:
            advice.append("🟡 СРЕДНИЙ УРОВЕНЬ")
            advice.append("У вас хорошие базовые шансы")
            advice.append("Создайте запас еды и воды на 2-3 месяца")
            advice.append("Научитесь обращаться с оружием")
        elif probability < 0.9:
            advice.append("🟢 ВЫСОКИЙ УРОВЕНЬ")
            advice.append("Отличные шансы на выживание")
            advice.append("Вы хорошо подготовлены")
            advice.append("Помогайте другим выжившим")
        else:
            advice.append("✅ ОТЛИЧНЫЙ УРОВЕНЬ")
            advice.append("Вы - идеальный кандидат на выживание")
            advice.append("Станьте лидером в группе выживших")
            advice.append("Передавайте свои знания другим")

        if user_data['age'] > 60:
            advice.append(f"🎯 В возрасте {user_data['age']} лет важно найти молодых помощников")

        if user_data['physical_ability'] < 5:
            advice.append(f"💪 Ваша физическая форма ({user_data['physical_ability']}/10) требует улучшения")
        elif user_data['physical_ability'] >= 8:
            advice.append(f"💪 Отличная физическая форма ({user_data['physical_ability']}/10) - ваш козырь")

        if user_data['autism']:
            advice.append("🧠 Используйте свои сильные стороны: внимание к деталям, системное мышление")

        if user_data['parents_count'] == 0:
            advice.append("👪 Рассмотрите возможность создания новой 'семьи' с другими выжившими")
        elif user_data['parents_count'] == 2:
            advice.append("👪 Вы имеете хорошую социальную поддержку")

        if user_data['iq'] < 90:
            advice.append("🧠 Развивайте практические навыки выживания")
        elif user_data['iq'] > 120:
            advice.append("🧠 Используйте свой интеллект для стратегического планирования")

        if user_data['vision'] < 50:
            advice.append("👁️ Позаботьтесь о запасных очках/линзах")

        if user_data['apocalypse_movies'] < 10:
            advice.append("🎬 Посмотрите больше фильмов про апокалипсис для психологической подготовки")
        elif user_data['apocalypse_movies'] > 20:
            advice.append("🎬 Ваша подготовленность к сценариям апокалипсиса на высоте")

        return advice

    def get_detailed_analysis(self, probability, user_data):
        """Детальный анализ факторов выживания"""
        analysis = {
            'probability': probability,
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }

        if user_data['physical_ability'] >= 7:
            analysis['strengths'].append(f"Хорошая физическая форма ({user_data['physical_ability']}/10)")

        if user_data['iq'] >= 110:
            analysis['strengths'].append(f"Высокий интеллект (IQ {user_data['iq']})")

        if user_data['parents_count'] >= 1:
            analysis['strengths'].append("Социальная поддержка (живые родители)")

        if user_data['apocalypse_movies'] >= 15:
            analysis['strengths'].append("Хорошая подготовленность к сценариям апокалипсиса")

        if user_data['age'] > 50:
            analysis['weaknesses'].append(f"Возраст {user_data['age']} лет может быть помехой")

        if user_data['physical_ability'] <= 4:
            analysis['weaknesses'].append(f"Слабая физическая форма ({user_data['physical_ability']}/10)")

        if user_data['autism']:
            analysis['weaknesses'].append("Расстройство аутистического спектра требует особого подхода")

        if user_data['vision'] < 70:
            analysis['weaknesses'].append(f"Сниженное зрение ({user_data['vision']}%)")

        if probability < 0.6:
            analysis['recommendations'].append("Немедленно начать физическую подготовку")
            analysis['recommendations'].append("Создать стратегический запас ресурсов")
            analysis['recommendations'].append("Найти группу выживших")

        if probability >= 0.6:
            analysis['recommendations'].append("Развивать лидерские качества")
            analysis['recommendations'].append("Создать систему взаимопомощи")
            analysis['recommendations'].append("Разработать план на случай разных сценариев")

        return analysis


survival_predictor = SurvivalPredictor()


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
user_survival_data = {}


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
/survival - оценить шансы на выживание
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


@bot.message_handler(commands=['survival'])
def start_survival_test(message):
    user_id = message.from_user.id
    user_survival_data[user_id] = {}

    msg = bot.reply_to(message, "🧬 ОЦЕНКА ШАНСОВ НА ВЫЖИВАНИЕ\n\nСколько вам лет?")
    bot.register_next_step_handler(msg, process_age_step)


def process_age_step(message):
    try:
        user_id = message.from_user.id
        age = int(message.text)

        if age < 0 or age > 120:
            bot.reply_to(message, "⚠️ Введите корректный возраст (0-120)")
            return start_survival_test(message)

        user_survival_data[user_id]['age'] = age

        markup = types.ReplyKeyboardMarkup(row_width=5, resize_keyboard=True)
        for i in range(1, 11):
            markup.add(types.KeyboardButton(str(i)))

        msg = bot.reply_to(message,
                           "💪 Оцените свои физические способности (1-10):\n1 - очень слабый\n10 - отличная форма",
                           reply_markup=markup)
        bot.register_next_step_handler(msg, process_physical_step)
    except:
        bot.reply_to(message, "⚠️ Введите число")
        start_survival_test(message)


def process_physical_step(message):
    try:
        user_id = message.from_user.id
        physical = int(message.text)

        if physical < 1 or physical > 10:
            bot.reply_to(message, "⚠️ Введите число от 1 до 10")
            return process_age_step(message)

        user_survival_data[user_id]['physical_ability'] = physical

        markup = types.ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
        markup.add(types.KeyboardButton("Нет"), types.KeyboardButton("Да"))

        msg = bot.reply_to(message, "🧠 Есть ли у вас расстройство аутистического спектра?", reply_markup=markup)
        bot.register_next_step_handler(msg, process_autism_step)
    except:
        bot.reply_to(message, "⚠️ Введите число")
        process_age_step(message)


def process_autism_step(message):
    user_id = message.from_user.id
    autism = 1 if message.text.lower() in ['да', 'yes'] else 0
    user_survival_data[user_id]['autism'] = autism

    markup = types.ReplyKeyboardMarkup(row_width=3, resize_keyboard=True)
    markup.add(types.KeyboardButton("0"), types.KeyboardButton("1"), types.KeyboardButton("2"))

    msg = bot.reply_to(message, "👪 Сколько родителей у вас есть в живых?", reply_markup=markup)
    bot.register_next_step_handler(msg, process_parents_step)


def process_parents_step(message):
    try:
        user_id = message.from_user.id
        parents = int(message.text)

        if parents < 0 or parents > 2:
            bot.reply_to(message, "⚠️ Введите 0, 1 или 2")
            return process_autism_step(message)

        user_survival_data[user_id]['parents_count'] = parents

        markup = types.ReplyKeyboardRemove()
        msg = bot.reply_to(message, "🧠 Какой у вас IQ (примерно)?", reply_markup=markup)
        bot.register_next_step_handler(msg, process_iq_step)
    except:
        bot.reply_to(message, "⚠️ Введите число")
        process_autism_step(message)


def process_iq_step(message):
    try:
        user_id = message.from_user.id
        iq = int(message.text)

        if iq < 40 or iq > 200:
            bot.reply_to(message, "⚠️ Введите корректный IQ (40-200)")
            return process_parents_step(message)

        user_survival_data[user_id]['iq'] = iq

        msg = bot.reply_to(message, "👁️ Какой процент зрения у вас остался? (0-100)")
        bot.register_next_step_handler(msg, process_vision_step)
    except:
        bot.reply_to(message, "⚠️ Введите число")
        process_parents_step(message)


def process_vision_step(message):
    try:
        user_id = message.from_user.id
        vision = int(message.text)

        if vision < 0 or vision > 100:
            bot.reply_to(message, "⚠️ Введите процент от 0 до 100")
            return process_iq_step(message)

        user_survival_data[user_id]['vision'] = vision

        msg = bot.reply_to(message, "🎬 Сколько фильмов про апокалипсис вы смотрели?")
        bot.register_next_step_handler(msg, process_movies_step)
    except:
        bot.reply_to(message, "⚠️ Введите число")
        process_iq_step(message)


def process_movies_step(message):
    try:
        user_id = message.from_user.id
        movies = int(message.text)

        if movies < 0:
            bot.reply_to(message, "⚠️ Введите положительное число")
            return process_vision_step(message)

        user_survival_data[user_id]['apocalypse_movies'] = movies

        calculate_survival_probability(message)
    except:
        bot.reply_to(message, "⚠️ Введите число")
        process_vision_step(message)


def calculate_survival_probability(message):
    user_id = message.from_user.id

    if user_id not in user_survival_data:
        bot.reply_to(message, "❌ Данные не найдены")
        return

    user_data = user_survival_data[user_id]

    probability = survival_predictor.predict_proba(user_data)

    advice = survival_predictor.get_survival_advice(probability, user_data)

    analysis = survival_predictor.get_detailed_analysis(probability, user_data)

    result_text = f"""
🎯 РЕЗУЛЬТАТЫ ОЦЕНКИ ВЫЖИВАНИЯ

📊 ВАШИ ДАННЫЕ:
• Возраст: {user_data['age']} лет
• Физические способности: {user_data['physical_ability']}/10
• Аутизм: {'Да' if user_data['autism'] else 'Нет'}
• Родители в живых: {user_data['parents_count']}
• IQ: {user_data['iq']}
• Зрение: {user_data['vision']}%
• Фильмы про апокалипсис: {user_data['apocalypse_movies']}

🔥 ВЕРОЯТНОСТЬ ВЫЖИВАНИЯ: {probability:.1%}

💡 ОСНОВНЫЕ РЕКОМЕНДАЦИИ:
"""

    for item in advice[:5]:
        result_text += f"• {item}\n"

    result_text += "\n🏆 ВАШИ СИЛЬНЫЕ СТОРОНЫ:\n"
    if analysis['strengths']:
        for strength in analysis['strengths']:
            result_text += f"✅ {strength}\n"
    else:
        result_text += "Нет выраженных сильных сторон\n"

    result_text += "\n⚠️ ОБЛАСТИ ДЛЯ УЛУЧШЕНИЯ:\n"
    if analysis['weaknesses']:
        for weakness in analysis['weaknesses']:
            result_text += f"🔧 {weakness}\n"
    else:
        result_text += "Минимальные слабые стороны\n"

    result_text += "\n🎯 ПРИОРИТЕТНЫЕ ДЕЙСТВИЯ:\n"
    for rec in analysis['recommendations']:
        result_text += f"📌 {rec}\n"

    if probability > 0.8:
        result_text += "\n🎉 ВЫСОКИЙ УРОВЕНЬ! Вы отлично подготовлены к апокалипсису!"
    elif probability > 0.6:
        result_text += "\n🟡 ХОРОШИЕ ШАНСЫ! У вас есть хороший потенциал для выживания."
    elif probability > 0.4:
        result_text += "\n🟠 СРЕДНИЙ УРОВЕНЬ! Вам нужно серьезно подготовиться."
    else:
        result_text += "\n🔴 ТРЕБУЕТСЯ ДЕЙСТВИЕ! Начните подготовку немедленно."

    if survival_predictor.model_data is not None:
        result_text += f"\n\n🤖 Прогноз сделан нейросетью (точность модели: ~96%)"
    else:
        result_text += f"\n\n⚠️ Используется эвристическая модель"

    bot.send_message(message.chat.id, result_text)

    markup = types.InlineKeyboardMarkup()
    btn1 = types.InlineKeyboardButton("🎧 Озвучить результат", callback_data='tts_result')
    btn2 = types.InlineKeyboardButton("📊 Подробная статистика", callback_data='detailed_stats')
    markup.add(btn1, btn2)

    bot.send_message(message.chat.id, "Выберите дополнительные опции:", reply_markup=markup)

    user_survival_data[user_id]['probability'] = probability


@bot.callback_query_handler(func=lambda call: True)
def handle_callback(call):
    user_id = call.from_user.id

    if call.data == 'tts_result' and user_id in user_survival_data:
        user_data = user_survival_data[user_id]
        probability = user_data['probability']

        tts_text = f"Ваша вероятность выживания составляет {probability:.1%}. "
        if probability > 0.8:
            tts_text += "Отличный результат! Вы хорошо подготовлены к апокалипсису."
        elif probability > 0.6:
            tts_text += "Хорошие шансы на выживание. Продолжайте подготовку."
        elif probability > 0.4:
            tts_text += "Средние шансы. Рекомендуется усилить подготовку."
        else:
            tts_text += "Требуется срочная подготовка. Начните немедленно."

        audio_buffer = text_to_speech(tts_text)
        if audio_buffer:
            bot.send_voice(call.message.chat.id, audio_buffer)
        else:
            bot.answer_callback_query(call.id, "❌ Не удалось озвучить результат")

    elif call.data == 'detailed_stats':
        if user_id in user_survival_data:
            user_data = user_survival_data[user_id]

            stats_text = f"""
📈 ДЕТАЛЬНАЯ СТАТИСТИКА:

Возраст ({user_data['age']} лет):
• Оптимальный возраст: 25-40 лет
• Ваш показатель: {'✅ Оптимальный' if 25 <= user_data['age'] <= 40 else '⚠️ Неоптимальный'}

Физическая форма ({user_data['physical_ability']}/10):
• Минимальный порог: 5/10
• Ваш показатель: {'✅ Выше минимального' if user_data['physical_ability'] >= 5 else '⚠️ Ниже минимального'}

IQ ({user_data['iq']}):
• Средний показатель: 100
• Ваш показатель: {'✅ Выше среднего' if user_data['iq'] >= 100 else '⚠️ Ниже среднего'}

Зрение ({user_data['vision']}%):
• Критический уровень: <30%
• Ваш показатель: {'✅ Достаточный' if user_data['vision'] >= 30 else '⚠️ Критический'}

Подготовленность ({user_data['apocalypse_movies']} фильмов):
• Базовый уровень: 10 фильмов
• Ваш показатель: {'✅ Хорошая подготовка' if user_data['apocalypse_movies'] >= 10 else '⚠️ Недостаточная подготовка'}
"""
            bot.send_message(call.message.chat.id, stats_text)

    bot.answer_callback_query(call.id)


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

    if message.text.lower() in ['выживание', 'шансы', 'прогноз']:
        bot.reply_to(message, "📊 Для оценки шансов на выживание используйте команду /survival")
    elif message.text.lower() in ['совет', 'рекомендации']:
        bot.reply_to(message, "💡 Для получения экспертных советов используйте команду /advice")
    elif message.text.lower() in ['картинка', 'изображение', 'генерация']:
        bot.reply_to(message,
                     "🎨 Для генерации изображений используйте команды:\n/quick - быстрая генерация\n/generate - с настройками")
    elif message.text.lower() in ['озвучка', 'аудио', 'tts']:
        bot.reply_to(message, "🎤 Для озвучки текста используйте команду /tts")
    else:
        help_text = """🤖 Я не распознал ваш запрос. Попробуйте одну из команд:
/survival - оценка шансов на выживание
/advice - экспертные советы
/quick - быстрая генерация изображения
/tts - озвучка текста
/categories - все темы по выживанию

Или задайте вопрос по выживанию напрямую!"""
        bot.reply_to(message, help_text)


if __name__ == "__main__":
    print("🤖 Бот запущен...")
    print(f"✅ Модель выживания: {'Загружена' if survival_predictor.model_data is not None else 'Эвристическая'}")
    bot.polling(none_stop=True, interval=1)
