import json
import time
import base64
import requests
import logging
from io import BytesIO
from typing import Optional
import tempfile
import os
import re

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


class ExpertSystem:
    def __init__(self):
        self.knowledge_base = {
            # БАЗОВЫЕ ПОТРЕБНОСТИ
            r".*очистить.*вод.*|.*фильтр.*вод.*|.*пить.*вод.*": {
                "response": """💧 **ОЧИСТКА ВОДЫ БЕЗ ФИЛЬТРА:**
1. **Кипячение** - 10+ минут убивает бактерии
2. **Солнечная дезинфекция** - в прозрачной бутылке на солнце 6 часов
3. **Песчаный фильтр** - песок + уголь + ткань
4. **Химическая очистка** - йод (5 капель на литр) или хлор
5. **Дистилляция** - кипячение с сбором пара""",
                "category": "water"
            },

            r".*растен.*есть.*|.*съедобн.*растен.*|.*ягод.*есть.*": {
                "response": """🌿 **СЪЕДОБНЫЕ РАСТЕНИЯ ЛЕСА:**
✅ **Безопасные:**
- Крапива (варить 5 минут)
- Одуванчик (листья и корни)
- Подорожник (молодые листья)
- Щавель (сырой или вареный)
- Клевер (цветки и листья)

❌ **ОПАСНО:**
- Волчье лыко
- Болиголов
- Вороний глаз
- Ландыш

⚠️ **Правило:** Не знаешь - не ешь! Сначала пробуй маленький кусочек""",
                "category": "food"
            },

            r".*укрыти.*|.*убежищ.*|.*построить.*укрыти.*": {
                "response": """🏠 **ПРОСТОЕ УКРЫТИЕ ИЗ ПОДРУЧНЫХ СРЕДСТВ:**
1. **Навес** - палка на дереве + ветки + листья
2. **Шалаш** - каркас из веток + лапник
3. **Землянка** - яма + бревна + дерн
4. **В зданиях** - подвалы, чердаки, внутренние комнаты

**Материалы:** пластик, доски, ветки, ткань, земля""",
                "category": "shelter"
            },

            r".*разжечь.*огонь.*|.*костер.*без.*спичек.*": {
                "response": """🔥 **РАЗЖИГАНИЕ ОГНЯ БЕЗ СПИЧЕК:**
1. **Линза** - лупа, очки, бутылка с водой
2. **Огниво** - камень и металл
3. **Трение** - лук-сверло или ручное сверление
4. **Батарейка** + фольга/вата
5. **Химия** - марганцовка + глицерин

**Растопка:** сухая трава, береста, пух, смола""",
                "category": "fire"
            },

            r".*ошибк.*город.*|.*нельзя.*делать.*город.*": {
                "response": """🚫 **ЧАСТЫЕ ОШИБКИ В ГОРОДЕ:**
1. Шуметь и привлекать внимание
2. Ходить днем по открытым местам
3. Доверять незнакомцам
4. Носить яркую одежду
5. Оставаться в высотках (риск обрушения)
6. Не проверять здания на безопасность
7. Не иметь запасных путей отступления""",
                "category": "urban_survival"
            },

            # БЕЗОПАСНОСТЬ И ЗДОРОВЬЕ
            r".*обеззаразить.*ран.*|.*лечить.*ран.*|.*порез.*": {
                "response": """🩹 **ОБЕЗЗАРАЖИВАНИЕ РАН БЕЗ ЛЕКАРСТВ:**
1. **Кипяченая вода** - промыть рану
2. **Солевой раствор** - 1 чайная ложка на литр
3. **Травы** - подорожник, тысячелистник, ромашка
4. **Древесный уголь** - растолочь и присыпать
5. **Мед** - природный антисептик

**Важно:** менять повязки ежедневно""",
                "category": "medical"
            },

            r".*передвигаться.*город.*|.*не.*замечен.*|.*скрыт.*движен.*": {
                "response": """🚶 **СКРЫТНОЕ ПЕРЕДВИЖЕНИЕ ПО ГОРОДУ:**
1. **Только ночью** - использовать темноту
2. **Тени и укрытия** - вдоль стен, в подворотнях
3. **Медленно и тихо** - слушать каждый звук
4. **Избегать:** 
   - Главные улицы
   - Открытые площади
   - Стеклянные здания
5. **Проверять маршрут** через бинокль""",
                "category": "security"
            },

            r".*опасн.*здан.*|.*обрушен.*|.*разрушен.*здан.*": {
                "response": """🏚️ **ПРИЗНАКИ ОПАСНОГО ЗДАНИЯ:**
🔴 **ОПАСНО:**
- Трещины в стенах
- Наклоненные конструкции
- Обрушенные перекрытия
- Запах газа
- Скрипы и шумы

🟢 **БЕЗОПАСНО:**
- Низкие этажи
- Бетонные конструкции
- Целые окна и двери
- Нет запахов""",
                "category": "shelter"
            },

            r".*защитить.*укрыти.*|.*мародер.*|.*защита.*от.*людей.*": {
                "response": """🛡️ **ЗАЩИТА УКРЫТИЯ ОТ МАРОДЕРОВ:**
1. **Скрытый вход** - за маскировкой
2. **Ловушки** - шумовые, световые
3. **Наблюдение** - бинокль, зеркала
4. **Баррикады** - мебель, доски, проволока
5. **Запасной выход** - всегда имей план Б
6. **Ночные дежурства** - смена каждые 2 часа""",
                "category": "security"
            },

            r".*радиац.*|.*химическ.*заражен.*|.*отравлен.*": {
                "response": """☢️ **ДЕЙСТВИЯ ПРИ ЗАРАЖЕНИИ:**
1. **Немедленно укрыться** - подвал, метро
2. **Закрыть все отверстия** - окна, двери, вентиляцию
3. **Принять йод** - в первые 6 часов
4. **Снять одежду** - оставить снаружи
5. **Промыть кожу** - водой с мылом
6. **Не есть местную пищу** - только консервы""",
                "category": "threats"
            },

            # ПЛАНИРОВАНИЕ И ДЕЙСТВИЯ
            r".*оставаться.*на.*месте.*|.*двигаться.*|.*убежищ.*": {
                "response": """📍 **ОСТАВАТЬСЯ ИЛИ ДВИГАТЬСЯ?**
🟢 **ОСТАВАТЬСЯ ЕСЛИ:**
- Укрытие безопасное
- Есть запас ресурсов
- Местность знакома
- Нет непосредственной угрозы

🔴 **ДВИГАТЬСЯ ЕСЛИ:**
- Угроза приближается
- Кончились ресурсы
- Местность опасна
- Есть информация о безопасном месте""",
                "category": "planning"
            },

            r".*набор.*вещ.*|.*рюкзак.*|.*тревожн.*чемодан.*": {
                "response": """🎒 **МИНИМАЛЬНЫЙ НАБОР ВЫЖИВАНИЯ:**
1. **Вода** - 2 литра + фильтр
2. **Еда** - консервы, шоколад, орехи
3. **Аптечка** - бинты, антисептик, обезболивающее
4. **Оружие** - нож, дубинка
5. **Инструменты** - мультитул, фонарь, спички
6. **Одежда** - теплая, непромокаемая
7. **Документы** - копии важных бумаг""",
                "category": "equipment"
            },

            r".*сигнал.*бедств.*|.*подать.*сигнал.*|.*рация.*": {
                "response": """🆘 **СИГНАЛЫ БЕДСТВИЯ БЕЗ РАЦИИ:**
1. **Дым** - три костра треугольником
2. **Зеркало** - солнечные зайчики
3. **Свисток** - три коротких, три длинных
4. **Фонарь** - SOS (...---...)
5. **Яркая ткань** - на открытом месте
6. **Камень/палка** - выложить HELP или SOS""",
                "category": "communication"
            },

            r".*ориентироваться.*|.*карт.*|.*компас.*": {
                "response": """🧭 **ОРИЕНТИРОВАНИЕ БЕЗ КАРТЫ И КОМПАСА:**
1. **Солнце** - встает на востоке, садится на западе
2. **Звезды** - Полярная звезда указывает север
3. **Мох** - растет с северной стороны деревьев
4. **Муравейники** - с южной стороны деревьев
5. **Церкви** - алтарь на востоке
6. **Телефон** - GPS без интернета""",
                "category": "navigation"
            },

            r".*сохранить.*продукт.*|.*холодильник.*|.*еда.*надолго.*": {
                "response": """🥫 **СОХРАНЕНИЕ ПРОДУКТОВ БЕЗ ХОЛОДИЛЬНИКА:**
1. **Консервация** - соль, уксус, сахар
2. **Сушка** - мясо, фрукты, овощи
3. **Копчение** - дым защищает от бактерий
4. **Закапывание** - в земле прохладнее
5. **Водяное охлаждение** - в колодце или реке
6. **Соление** - особенно для мяса и рыбы""",
                "category": "food"
            },

            # ДОЛГОСРОЧНОЕ ВЫЖИВАНИЕ
            r".*хранен.*вод.*|.*больш.*вод.*|.*запас.*вод.*": {
                "response": """💦 **ДОЛГОСРОЧНОЕ ХРАНЕНИЕ ВОДЫ:**
1. **Чистые емкости** - пластиковые бутыли, бочки
2. **Темное место** - защита от водорослей
3. **Прохлада** - подвал или земля
4. **Консервация** - хлорка (1/8 ч.л. на 20л)
5. **Ротация** - менять воду каждые 6 месяцев
6. **Запас** - 4 литра на человека в день""",
                "category": "water"
            },

            r".*сбор.*дождев.*вод.*|.*дождев.*вод.*": {
                "response": """🌧️ **СИСТЕМА СБОРА ДОЖДЕВОЙ ВОДЫ:**
1. **Крыша** - желоб + труба + бочка
2. **Тент/плащ** - воронка в емкость
3. **Яма** - выложить пленкой
4. **Деревья** - накинуть ткань на ветки
5. **Фильтрация** - песок + уголь перед употреблением

**Важно:** кипятить перед питьем!""",
                "category": "water"
            },

            r".*еда.*зимой.*|.*зим.*пищ.*|.*добыть.*еду.*зимой.*": {
                "response": """❄️ **ДОБЫЧА ПИЩИ ЗИМОЙ:**
1. **Охота** - следы на снегу, засады у водопоев
2. **Рыбалка** - проруби, подледная ловля
3. **Сушеные запасы** - грибы, ягоды, травы
4. **Кора деревьев** - внутренний слой сосны, березы
5. **Хвойные иголки** - чай с витамином С
6. **Зимние ягоды** - рябина, калина, шиповник""",
                "category": "food"
            },

            r".*ловушк.*|.*поймать.*животн.*|.*дич.*": {
                "response": """🪤 **ПРОСТЫЕ ЛОВУШКИ ДЛЯ ДИЧИ:**
1. **Петля** - на звериных тропах
2. **Западня** - яма с кольями
3. **Давилка** - бревно с приманкой
4. **Силок** - для птиц и мелких животных
5. **Рыболовная** - из пластиковой бутылки

**Приманка:** ягоды, зерно, соль""",
                "category": "food"
            },

            r".*сохранить.*мясо.*|.*мясо.*без.*соли.*": {
                "response": """🥩 **СОХРАНЕНИЕ МЯСА БЕЗ СОЛИ И ХОЛОДИЛЬНИКА:**
1. **Вяление** - нарезать тонко, сушить на ветру
2. **Копчение** - холодный дым несколько дней
3. **Жир** - залить растопленным жиром
4. **Мед** - залить медом как консервантом
5. **Закапывание** - в сухой песок или золу
6. **Мороз** - зимой использовать естественный холод""",
                "category": "food"
            },

            r".*вырастить.*овощ.*|.*семен.*|.*огород.*": {
                "response": """🌱 **ВЫРАЩИВАНИЕ ОВОЩЕЙ ИЗ СЕМЯН:**
1. **Быстрые культуры:** редис (20 дней), салат (30 дней)
2. **Питательные:** картофель, морковь, капуста
3. **Контейнеры:** ведра, ящики, мешки с землей
4. **Удобрения:** зола, компост, рыбные отходы
5. **Полив:** дождевая вода, роса
6. **Защита:** сетки от птиц, зола от насекомых""",
                "category": "food"
            },

            r".*мыло.*|.*чистящ.*средств.*|.*гигиен.*": {
                "response": """🧼 **САМОДЕЛЬНОЕ МЫЛО И ЧИСТЯЩИЕ СРЕДСТВА:**
1. **Зольное мыло** - зола + жир + вода
2. **Мыльные орехи** - природные моющие
3. **Сода** - для чистки и дезинфекции
4. **Уксус** - против накипи и запахов
5. **Хвоя** - отвар для мытья посуды
6. **Песок** - абразив для чистки""",
                "category": "hygiene"
            },

            r".*паник.*|.*страх.*|.*безысходност.*": {
                "response": """😨 **БОРЬБА С ПАНИКОЙ И СТРАХОМ:**
1. **Дыхание** - глубокий вдох, медленный выдох
2. **Фокусировка** - на конкретных задачах
3. **Режим дня** - структура помогает
4. **Физическая активность** - сжигает стресс
5. **Разговор** - обсуждение с другими
6. **Отдых** - спать минимум 6 часов""",
                "category": "psychology"
            },

            r".*отношен.*выживш.*|.*альянс.*|.*групп.*": {
                "response": """🤝 **СОЗДАНИЕ АЛЬЯНСОВ С ВЫЖИВШИМИ:**
1. **Осторожный контакт** - на расстоянии
2. **Обмен ресурсами** - еда на лекарства
3. **Общие цели** - защита, добыча пищи
4. **Четкие правила** - распределение обязанностей
5. **Испытательный срок** - 2 недели наблюдения
6. **Выходные пути** - план если что-то пойдет не так""",
                "category": "community"
            },

            r".*правил.*сообществ.*|.*закон.*групп.*": {
                "response": """📜 **ПРАВИЛА ДЛЯ СООБЩЕСТВА ВЫЖИВШИХ:**
1. **Безопасность прежде всего** - ночные дежурства
2. **Равный труд** - каждый работает
3. **Общие ресурсы** - справедливое распределение
4. **Коллективные решения** - голосование по важным вопросам
5. **Конфликты** - третейский суд
6. **Изгнание** - за воровство или насилие""",
                "category": "community"
            },

            r".*распределить.*задач.*|.*обязанност.*групп.*": {
                "response": """👥 **РАСПРЕДЕЛЕНИЕ ЗАДАЧ В ГРУППЕ:**
1. **Охранники** - безопасность, наблюдение
2. **Добытчики** - еда, вода, ресурсы
3. **Медики** - здоровье, гигиена
4. **Строители** - укрытие, укрепления
5. **Повара** - приготовление пищи
6. **Разведчики** - информация о местности

**Смена:** менять обязанности каждую неделю""",
                "category": "community"
            },

            r".*одиночеств.*|.*рассудок.*|.*психологическ.*здоров.*": {
                "response": """🧠 **БОРЬБА С ОДИНОЧЕСТВОМ:**
1. **Режим дня** - структурирует время
2. **Хобби** - чтение, рисование, музыка
3. **Дневник** - записывать мысли и события
4. **Физические упражнения** - поддерживают тонус
5. **Разговоры с собой** - вслух, чтобы слышать голос
6. **Цели** - ставить маленькие achievable цели""",
                "category": "psychology"
            },

            # МЕДИЦИНА И ГИГИЕНА
            r".*инфекц.*|.*антибиотик.*|.*заражен.*ран.*": {
                "response": """🦠 **ЛЕЧЕНИЕ ИНФЕКЦИИ БЕЗ АНТИБИОТИКОВ:**
1. **Чеснок** - природный антибиотик
2. **Мед** - на раны, против воспаления
3. **Хрен** - компрессы на раны
4. **Кора ивы** - содержит салициловую кислоту
5. **Чай tree oil** - если есть в запасах
6. **Чистота** - главное средство против инфекций""",
                "category": "medical"
            },

            r".*обезболивающ.*|.*боль.*|.*природн.*средств.*": {
                "response": """💊 **ПРИРОДНЫЕ ОБЕЗБОЛИВАЮЩИЕ:**
1. **Ива** - кора (аспирин)
2. **Мята** - чай от головной боли
3. **Ромашка** - противовоспалительное
4. **Крапива** - компрессы от суставной боли
5. **Чеснок** - при зубной боли
6. **Холод/тепло** - компрессы в зависимости от боли""",
                "category": "medical"
            },

            r".*гигиен.*|.*чистот.*|.*болезн.*антисанитар.*": {
                "response": """🚿 **ГИГИЕНА В АНТИСАНИТАРНЫХ УСЛОВИЯХ:**
1. **Мытье рук** - после туалета, перед едой
2. **Вода + зола** - замена мылу
3. **Отдельное место** - для туалета вдали от жилья
4. **Смена одежды** - регулярно стирать
5. **Чистка зубов** - уголь + соль
6. **Борьба с насекомыми** - дым, травы

**Профилактика:** лучше чем лечение!""",
                "category": "hygiene"
            }
        }

    def find_expert_advice(self, query):
        query_lower = query.lower()
        for pattern, knowledge in self.knowledge_base.items():
            if re.search(pattern, query_lower):
                return knowledge
        return None

    def get_categories(self):
        categories = set()
        for knowledge in self.knowledge_base.values():
            categories.add(knowledge["category"])
        return sorted(categories)


api = FusionBrainAPI('https://api-key.fusionbrain.ai/', '375267412DEC77B1A0214E69EEEA3771',
                     'C8037A1589CF8AB7E078D022E9591B2B')
openai_api = "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"

client = OpenAI(api_key=openai_api)
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
