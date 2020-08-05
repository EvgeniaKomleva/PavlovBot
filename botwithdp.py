from deeppavlov.dataset_readers.dstc2_reader import SimpleDSTC2DatasetReader
from deeppavlov.dataset_iterators.dialog_iterator import DialogDatasetIterator
from pprint import pprint
from deeppavlov import configs
from deeppavlov.core.common.file import read_json
from deeppavlov import train_model
from deeppavlov import build_model


class AssistantDatasetReader(SimpleDSTC2DatasetReader):

    @staticmethod
    def _data_fname(datatype):
        assert datatype in ('val', 'trn', 'tst'), "wrong datatype name"
        return f"assistant-{datatype}.json"


if __name__ == '__main__':
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

# bot([[{"text": "Привет"}]])

# from deeppavlov.utils.telegram import interact_model_by_telegram

# interact_model_by_telegram(model_config=gobot_config, token='1153548935:AAFIZkbBaYKjzlpum6wVM6oTHviL4VYlPY8')
