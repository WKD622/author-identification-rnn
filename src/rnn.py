import sys
sys.path.append('/net/people/plgjakubziarko/author-identification-rnn/')
from library.network.train import Train

rnn = Train(hidden_size=int(sys.argv[1]),
            num_layers=int(sys.argv[2]),
            num_epochs=int(sys.argv[3]),
            batch_size=int(sys.argv[4]),
            timesteps=int(sys.argv[5]),
            learning_rate=float(sys.argv[6]),
            authors_size=int(sys.argv[7]),
            vocab_size=int(sys.argv[8]),
            save_path=sys.argv[9],
            training_tensors_path=sys.argv[10],
            testing_tensors_path=sys.argv[11],
            language=sys.argv[12],
            truth_file_path=sys.argv[13])

rnn.train()
