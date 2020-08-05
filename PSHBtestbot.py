import config
import telebot
from telebot import types  # кнопки
from string import Template
import telebot
from deeppavlov.utils.telegram import interact_model_by_telegram
from deeppavlov import configs
from deeppavlov.core.common.file import read_json
from deeppavlov.dataset_readers.dstc2_reader import SimpleDSTC2DatasetReader
from deeppavlov.dataset_iterators.dialog_iterator import DialogDatasetIterator
from pprint import pprint
from deeppavlov import configs
from deeppavlov.core.common.file import read_json
from deeppavlov import train_model
from deeppavlov import build_model

bot = telebot.TeleBot('1302667182:AAHMoB8cQOZReu2bjEKugEf9Q--7QFJLbi0')

class AssistantDatasetReader(SimpleDSTC2DatasetReader):

    @staticmethod
    def _data_fname(datatype):
        assert datatype in ('val', 'trn', 'tst'), "wrong datatype name"
        return f"assistant-{datatype}.json"

data = AssistantDatasetReader().read('assistant_data')

iterator = DialogDatasetIterator(data)

for dialog in iterator.gen_batches(batch_size=1, data_type='train'):
    turns_x, turns_y = dialog

    print("User utterances:\n----------------\n")
    pprint(turns_x[0], indent=4)
    print("\nSystem responses:\n-----------------\n")
    pprint(turns_y[0], indent=4)

    break

print("\n-----------------")
print(f"{len(iterator.get_instances('train')[0])} dialog(s) in train.")
print(f"{len(iterator.get_instances('valid')[0])} dialog(s) in valid.")
print(f"{len(iterator.get_instances('test')[0])} dialog(s) in test.")

gobot_config = read_json(configs.go_bot.gobot_dstc2_minimal)

gobot_config['chainer']['pipe'][-1]['embedder'] = {
    "class_name": "glove",
    "load_path": "assistant_bot/small.txt"
}

gobot_config['chainer']['pipe'][-1]['nlg_manager']['template_path'] = 'assistant_data/assistant-templates.txt'
gobot_config['chainer']['pipe'][-1]['nlg_manager']['api_call_action'] = None

gobot_config['dataset_reader']['class_name'] = '__main__:AssistantDatasetReader'
gobot_config['metadata']['variables']['DATA_PATH'] = 'assistant_data'

gobot_config['metadata']['variables']['MODEL_PATH'] = 'assistant_bot'

gobot_config['train']['batch_size'] = 4  # set batch size
gobot_config['train']['max_batches'] = 30  # maximum number of training batches
gobot_config['train']['val_every_n_batches'] = 30  # evaluate on full 'valid' split every 30 epochs
gobot_config['train']['log_every_n_batches'] = 5  # evaluate on full 'train' split every 5 batches

train_model(gobot_config)

bot_model = build_model(gobot_config)

# gobot_config = read_json(configs.go_bot.gobot_dstc2_minimal)
# interact_model_by_telegram(model_config=gobot_config, token='1153548935:AAFIZkbBaYKjzlpum6wVM6oTHviL4VYlPY8')




#def train(message):


    # return bot_model

def get_answer(bot_model):
    answer = bot_model([[{"text": "Привет"}]])
    return answer
# from deeppavlov.utils.telegram import interact_model_by_telegram

# interact_model_by_telegram(model_config=gobot_config, token='1153548935:AAFIZkbBaYKjzlpum6wVM6oTHviL4VYlPY8')


# если /help, /start
@bot.message_handler(commands=['help', 'start'])
def send_welcome(message):
    # markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    # itembtn1 = types.KeyboardButton('/about')
    # itembtn2 = types.KeyboardButton('/reg')
    # key = types.InlineKeyboardMarkup()
    # markup.add(itembtn1, itembtn2)
    # key.add(itembtn2)
    keyboard = types.InlineKeyboardMarkup(row_width=1)
    first_button = types.InlineKeyboardButton(text="Начать", callback_data="first")
    keyboard.add(first_button)

    bot.send_message(message.chat.id, "Здравствуйте "
                     + message.from_user.first_name
                     + ",я - помощник «Россельхозбанк»!, чтобы вы хотели узнать?")
    bot.send_message(message.chat.id,
                     "Могу рассказать вам о продуктах Банка, помогу в выборе и оформлю их за пару минут.")
    bot.send_message(message.chat.id,
                     "P.S. Всегда можно перейти в «Главное меню» или попросить «Помощь», только напишите мне об этом!",
                     reply_markup=keyboard)


@bot.callback_query_handler(func=lambda call: True)
def callback_inline(call):
    chat_id = call.message.chat.id
    if call.data == "first":
        # bot.send_message(chat_id,"Какой продукт вас интересует?")
        keyboardmain = types.InlineKeyboardMarkup(row_width=1)
        first_button = types.InlineKeyboardButton(text="Кредит", callback_data="credit")
        second_button = types.InlineKeyboardButton(text="Ипотека", callback_data="second")
        fird_button = types.InlineKeyboardButton(text="Кредитная карта", callback_data="credit_card")
        four_button = types.InlineKeyboardButton(text="Дебетовая карта", callback_data="debet_card")
        keyboardmain.add(first_button, second_button, fird_button, four_button)
        # bot.edit_message_text(chat_id=call.message.chat.id,message_id=call.message.message_id, text="menu",reply_markup=keyboardmain)
        bot.send_message(chat_id, "Какой продукт вас интересует?", parse_mode='HTML', reply_markup=keyboardmain)

    if call.data == "credit":
        # bot.send_message(chat_id,"Какой продукт вас интересует?")
        keyboard = types.InlineKeyboardMarkup(row_width=1)
        first_button = types.InlineKeyboardButton(text="Взять новый кредит", callback_data="new_credit")
        second_button = types.InlineKeyboardButton(text="Рефинансировать кредит", callback_data="refresh")
        keyboard.add(first_button, second_button)
        # bot.edit_message_text(chat_id=call.message.chat.id,message_id=call.message.message_id, text="menu",reply_markup=keyboardmain)
        bot.send_message(chat_id, "Я расскажу о наших лучших предложениях и подберу наиболее выгодную программу!")
        bot.send_message(chat_id, "Планируете взять новый кредит или рефинансировать существующий кредит?",
                         parse_mode='HTML', reply_markup=keyboard)
        # bot.register_next_step_handler(msg, process_period_step)
    if call.data == "new_credit":
        # bot.send_message(chat_id,"Какой продукт вас интересует?")
        keyboard = types.InlineKeyboardMarkup(row_width=1)
        first_button = types.InlineKeyboardButton(text="Кредит наличными", callback_data="cash_credit")
        second_button = types.InlineKeyboardButton(text="Пенсионный кредит", callback_data="old_credit")
        keyboard.add(first_button, second_button)
        # bot.edit_message_text(chat_id=call.message.chat.id,message_id=call.message.message_id, text="menu",reply_markup=keyboardmain)
        bot.send_message(chat_id, "Я расскажу о наших лучших предложениях и подберу наиболее выгодную программу!")
        bot.send_message(chat_id, "Какая программа вам подходит?", parse_mode='HTML', reply_markup=keyboard)
    if call.data == "cash_credit":
        # bot.send_message(chat_id,"Какой продукт вас интересует?")
        keyboard = types.InlineKeyboardMarkup(row_width=1)
        first_button = types.InlineKeyboardButton(text="Подать заявку", callback_data="cash_credit")
        second_button = types.InlineKeyboardButton(text="Рассчитать платеж", callback_data="old_credit")
        fird_button = types.InlineKeyboardButton(text="Требования", callback_data="cash_credit")
        four_button = types.InlineKeyboardButton(text="Документы", callback_data="old_credit")
        keyboard.add(first_button, second_button, fird_button, four_button)
        # bot.edit_message_text(chat_id=call.message.chat.id,message_id=call.message.message_id, text="menu",reply_markup=keyboardmain)
        bot.send_photo(chat_id, photo=open('photo.jpg', 'rb'))
        bot.send_message(chat_id,
                         "Исполни мечту! Без залогов и поручительства\n— Кредит на любые цели\n— Сумма кредита до 5 000 000 руб.\n— Отсутствие комиссий по кредиту\n— Возможность выбора схемы погашения кредита")
        bot.send_message(chat_id, "retail.rshb.fil-it.ru/loans/bez_op", parse_mode='HTML', reply_markup=keyboard)


@bot.message_handler(content_types=["text"])
def send_help(message):
    # bot.send_message(message.chat.id, 'Я только учусь и очень стараюсь, воспользуйтесь меню')
    print("get some text")
    print(bot_model(message))


# произвольное фото
@bot.message_handler(content_types=["photo"])
def send_help_text(message):
    bot.send_message(message.chat.id, 'Напишите текст')


bot.enable_save_next_step_handlers(delay=2)

#bot.load_next_step_handlers()

if __name__ == '__main__':
    bot.polling(none_stop=True)

