from data_loader.data_loader import ModelDataLoader
from models.model_model import Model
from trainers.trainer import ModelTrainer
from predictor.predictor import ModelPredict
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config['callbacks']['tensorboard_log_dir'], config['callbacks']['checkpoint_dir']])

    print('Create the data generator.')
    data_loader = ModelDataLoader(config)
    data_loader.load_data()
    #print(data_loader.get_test_data())

    print('Create the model.')
    model = Model(config)

    ###print('Create the trainer')
    ###trainer = ModelTrainer(model, data_loader.get_train_data(), config)

    ###print('Start training the model.')
    ###trainer.train()
    
    print('Start predict.')
    predictor = ModelPredict(model, data_loader.get_train_data(), config)


if __name__ == '__main__':
    main()
